"""Temporary disk indexes for exact audits of large window manifests."""
from collections import OrderedDict
from contextlib import contextmanager
import json
import math
import sqlite3
from tempfile import TemporaryDirectory

from .storage import storage_environment


def _key(value):
    if isinstance(value, int):
        return f"n{int(value)}/1"
    if isinstance(value, float) and math.isfinite(value):
        numerator, denominator = value.as_integer_ratio()
        return f"n{numerator}/{denominator}"
    return "s" + value if isinstance(value, str) else "j" + json.dumps(value, ensure_ascii=False, sort_keys=True)


class ManifestIndex:
    """Keep uniqueness, split membership and sorted coverage off the Python heap."""
    def __init__(self, path):
        self.db = sqlite3.connect(str(path))
        try:
            self.db.executescript("""
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            PRAGMA temp_store=FILE;
            PRAGMA cache_size=-4096;
            CREATE TABLE seen(scope TEXT, value TEXT, PRIMARY KEY(scope,value)) WITHOUT ROWID;
            CREATE TABLE groups(id INTEGER PRIMARY KEY, kind TEXT, value TEXT, payload TEXT, UNIQUE(kind,value));
            CREATE TABLE splits(group_id INTEGER, split TEXT, PRIMARY KEY(group_id,split)) WITHOUT ROWID;
            CREATE TABLE sources(id INTEGER PRIMARY KEY, pool TEXT, source TEXT, UNIQUE(pool,source));
            CREATE TABLE intervals(source_id INTEGER, start REAL, end REAL);
            CREATE INDEX interval_order ON intervals(source_id,start,end);
            CREATE TABLE file_info(path TEXT, sha TEXT, frames INTEGER, duration REAL, PRIMARY KEY(path,sha)) WITHOUT ROWID;
            CREATE TABLE admissions(path TEXT, sha TEXT, data TEXT, PRIMARY KEY(path,sha)) WITHOUT ROWID;
            CREATE TABLE admitted_hashes(path TEXT, sha TEXT, value TEXT, PRIMARY KEY(path,sha,value)) WITHOUT ROWID;
            """)
        except BaseException:
            self.db.close()
            raise
        self.groups = OrderedDict()
        self.sources = OrderedDict()
        self.files = OrderedDict()
        self.partitions = OrderedDict()
        self.seen = OrderedDict()
        self.admission_metadata = OrderedDict()
        self.admission_membership = OrderedDict()

    @staticmethod
    def _cache(cache, key, value):
        cache[key] = value
        cache.move_to_end(key)
        if len(cache) > 256:
            cache.popitem(last=False)
        return value

    def unique(self, scope, value):
        key = scope, value
        if key in self.seen:
            return False
        try:
            self.db.execute("INSERT INTO seen VALUES (?,?)", (scope, _key(value)))
            self._cache(self.seen, key, True)
            return True
        except sqlite3.IntegrityError:
            self._cache(self.seen, key, True)
            return False

    def partition(self, kind, value, split):
        key = kind, value
        group_id = self.groups.get(key)
        if group_id is None:
            encoded = _key(value)
            self.db.execute("INSERT OR IGNORE INTO groups(kind,value,payload) VALUES (?,?,?)",
                            (kind, encoded, json.dumps(value, ensure_ascii=False)))
            group_id = self.db.execute("SELECT id FROM groups WHERE kind=? AND value=?", (kind, encoded)).fetchone()[0]
            self._cache(self.groups, key, group_id)
        pair = group_id, split
        if pair not in self.partitions:
            self.db.execute("INSERT OR IGNORE INTO splits VALUES (?,?)", (group_id, json.dumps(split)))
            self._cache(self.partitions, pair, True)

    def leaks(self):
        records = []
        for group_id, kind, value in self.db.execute("""
            SELECT g.id,g.kind,g.payload FROM groups g JOIN splits s ON s.group_id=g.id
            GROUP BY g.id HAVING COUNT(*)>1 ORDER BY g.id
        """):
            splits = [json.loads(row[0]) for row in self.db.execute("SELECT split FROM splits WHERE group_id=?", (group_id,))]
            records.append({"key": kind, "id": json.loads(value), "splits": sorted(splits)})
        return records

    def interval(self, pool, source, start, end):
        key = pool, source
        source_id = self.sources.get(key)
        if source_id is None:
            self.db.execute("INSERT OR IGNORE INTO sources(pool,source) VALUES (?,?)", (pool, _key(source)))
            source_id = self.db.execute("SELECT id FROM sources WHERE pool=? AND source=?", (pool, _key(source))).fetchone()[0]
            self._cache(self.sources, key, source_id)
        self.db.execute("INSERT INTO intervals VALUES (?,?,?)", (source_id, start, end))

    def unique_hours(self, pool):
        seconds = 0.0
        for (source_id,) in self.db.execute("SELECT id FROM sources WHERE pool=? ORDER BY id", (pool,)):
            end = -math.inf
            for a, b in self.db.execute("SELECT start,end FROM intervals WHERE source_id=? ORDER BY start,end", (source_id,)):
                seconds += max(0, b - max(a, end))
                end = max(end, b)
        return seconds / 3600

    def file_info(self, key, inspect):
        cached = self.files.get(key)
        if cached is None:
            cached = self.db.execute("SELECT frames,duration FROM file_info WHERE path=? AND sha=?", key).fetchone()
            if cached is None:
                info = inspect()
                cached = info.frames, info.duration
                self.db.execute("INSERT INTO file_info VALUES (?,?,?,?)", (*key, *cached))
            self._cache(self.files, key, cached)
        return cached

    def admission(self, key, load):
        if key in self.admission_metadata:
            return self.admission_metadata[key]
        value = self.db.execute("SELECT data FROM admissions WHERE path=? AND sha=?", key).fetchone()
        if value is not None:
            return self._cache(self.admission_metadata, key, json.loads(value[0]))
        artifact = load()
        self.db.executemany("INSERT OR IGNORE INTO admitted_hashes VALUES (?,?,?)",
                            ((*key, _key(value)) for value in artifact.get("accepted_source_sha256", [])))
        metadata = {k: artifact.get(k) for k in ("state", "wakeword", "human_reference_used",
                    "generation_policy_approved", "content_review")}
        self.db.execute("INSERT INTO admissions VALUES (?,?,?)", (*key, json.dumps(metadata)))
        return self._cache(self.admission_metadata, key, metadata)

    def admits(self, key, sha):
        cached = key, sha
        if cached not in self.admission_membership:
            value = self.db.execute("SELECT 1 FROM admitted_hashes WHERE path=? AND sha=? AND value=?",
                                    (*key, _key(sha))).fetchone() is not None
            self._cache(self.admission_membership, cached, value)
        return self.admission_membership[cached]


@contextmanager
def manifest_index(root):
    """The selected project cache owns all temporary indexes, including SQL sorts."""
    with storage_environment(root) as storage:
        with TemporaryDirectory(prefix="koww_manifest_", dir=storage["temp"]) as directory:
            from pathlib import Path
            index = ManifestIndex(Path(directory) / "manifest.sqlite3")
            try:
                yield index
            finally:
                index.db.close()
