"""One bounded queue for independently downloadable fixed material sources.

Large automatically selected Train sources keep their bounded acquisition,
conversion and retirement transaction; they are only announced here.
"""
from collections import defaultdict, deque
from contextvars import ContextVar
from pathlib import Path
from urllib.parse import urlsplit
import os

from .downloads import VerificationCache, download_checked, download_many


_current = ContextVar("koww_material_downloads", default=None)


def _identity(path):
    info = Path(path).stat()
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def borrowed_download(path, algorithm, expected, size_bytes):
    """Borrow only this call's verified, unchanged original with identical pins."""
    owner = _current.get()
    if owner is None:
        return None
    path = Path(path).resolve()
    binding = owner.verified.get((str(path), algorithm, expected, size_bytes))
    if binding is None:
        return None
    try:
        return path if _identity(path) == binding else None
    except FileNotFoundError:
        return None


def _provider_order(items):
    """Offer different providers before more files from the same provider."""
    providers = defaultdict(deque)
    for item in items:
        providers[urlsplit(item["source_url"]).netloc.lower()].append(item)
    ordered = []
    while any(providers.values()):
        for waiting in providers.values():
            if waiting:
                ordered.append(waiting.popleft())
    return ordered


class MaterialDownloads:
    def __init__(self, workspace, progress=None):
        self.workspace = Path(workspace).resolve()
        self.shared = self.workspace / "shared"
        self.destination = self.shared / "downloads"
        self.progress = progress
        self.verification_cache = VerificationCache()
        self.verified, self.files = {}, {}
        self.base_paths = {}
        self.official_music = None
        self.mit_directory = None
        self._token = None

    def __enter__(self):
        self._token = _current.set(self)
        return self

    def __exit__(self, *args):
        _current.reset(self._token)
        self.verified.clear()
        self.verification_cache = VerificationCache()

    def report(self, event):
        event = dict(event)
        for entry in event.get("files", []):
            self.files.setdefault(entry["id"], {}).update(entry)
        # Automatic acquisition already reports component hours around each
        # bounded part; do not invent a file list or a completed download.
        if event.get("unit") == "hours":
            component = event.get("label", "").rsplit(": ", 1)[-1]
            key = "automatic:" + component
            if key in self.files:
                done, target = event.get("done", 0), event.get("total")
                self.files[key].update(done=done, total=target, unit="hours", prepared_hours=done,
                    state="complete" if target is not None and done >= target else "downloading")
        if self.files:
            event["files"] = [dict(row) for row in self.files.values()]
        if self.progress:
            self.progress(event)

    def finish_automatic(self, result):
        for component, actual in result.get("component_actual_hours", {}).items():
            row = self.files.get("automatic:" + component)
            if row:
                row.update(done=actual, total=row["target_hours"], unit="hours", prepared_hours=actual,
                           state="complete" if actual >= row["target_hours"] else "failed")
        self.report({"phase": "automatic_materials", "label": "대용량 비호출 자료 준비 결과 확인"})

    def _remember(self, path, item):
        path = Path(path).resolve()
        key = (str(path), item["algorithm"], item["expected"], item["size_bytes"])
        observed = _identity(path)
        if self.verification_cache.matches(path, item["algorithm"], item["expected"]) and _identity(path) == observed:
            self.verified[key] = observed
            return True
        return False

    def prepare(self, assets, recipe, config):
        from . import studio_data
        from .evaluation_rir import common_rir_catalog
        items, destinations = [], {}
        def add(item):
            target = item.get("target")
            if target is not None:
                key = os.path.normcase(str(Path(target).resolve()))
                pin = (item["algorithm"], item["expected"], item["size_bytes"])
                if key in destinations:
                    if destinations[key] != pin:
                        raise ValueError("Conflicting pinned downloads share one destination")
                    return
                destinations[key] = pin
            items.append(item)

        for asset in assets:
            key = asset["id"]
            group, label = (("generic_negative", "일반 비호출 · openWakeWord") if key == "generic" else
                            ("zeroth", "한국어 낭독 · Zeroth") if key == "zeroth" else
                            ("audioset", "소음 증강 · AudioSet") if key.startswith("audioset") else
                            ("official_music", "소음 증강 음악 · FMA") if key in ("fma", "fma_metadata") else
                            (key, asset["filename"]))
            add({"id": "base:" + asset["id"], "kind": "base", "asset": asset,
                 "label": label if key in ("generic", "zeroth") else asset["filename"], "group": group, "group_label": label,
                 "source_url": asset["source_url"], "size_bytes": asset["size_bytes"],
                 "target": studio_data._cached_asset(asset, self.destination),
                 "algorithm": asset["checksum_algorithm"], "expected": asset[asset["checksum_algorithm"]]})
        if recipe.get("official_noise_acquisition_policy"):
            from .official_music import official_music_catalog
            catalog = official_music_catalog()
            add({"id": "official_music", "kind": "official_music", "label": "음악 120곡 · 곡 목록",
                 "group": "official_music", "group_label": "소음 증강 음악 · FMA",
                 "source_url": catalog["archives"]["music"]["source_url"],
                 "size_bytes": sum(row["compressed_bytes"] for source in catalog["archives"].values() for row in source["members"]),
                 "file_count": 121})
        self.report({"phase": "download_plan", "label": "전체 다운로드 목록 확인"})
        self.mit_directory, mit = studio_data._mit_download_plan(self.shared)
        for request in mit:
            url, path, expected, size = request["request"]
            add({"id": "mit:" + request["id"], "kind": "mit", "label": request["id"],
                 "group": "mit_rir", "group_label": "학습용 공간 울림 · MIT RIR",
                 "source_url": url, "size_bytes": size, "target": path,
                 "algorithm": "sha256", "expected": expected})
        for asset in recipe.get("evaluation_assets", []):
            add({"id": "evaluation:" + asset["id"], "kind": "base", "asset": asset,
                 "label": asset["filename"], "group": "evaluation_legacy", "group_label": "기존 평가용 배경 음원",
                 "source_url": asset["source_url"], "size_bytes": asset["size_bytes"],
                 "target": studio_data._cached_asset(asset, self.destination),
                 "algorithm": asset["checksum_algorithm"], "expected": asset[asset["checksum_algorithm"]]})
        if recipe.get("recipe_version") in (5, 6):
            for asset in recipe["common_evaluation_catalog"]["archives"]:
                group, label = (("fsd50k", "평가용 환경음·전자음 · FSD50K") if asset["id"].startswith("fsd_") else
                                ("musan", "평가용 말소리·음악 · MUSAN") if asset["id"] == "musan" else
                                ("seoul", "평가용 한국어 대화 · Seoul") if asset["id"] == "seoul" else
                                ("zeroth", "한국어 낭독 · Zeroth") if asset["id"] == "zeroth" else
                                (asset["id"], asset["filename"]))
                add({"id": "common:" + asset["id"], "kind": "common", "asset": asset,
                     "label": asset["filename"] if group == "fsd50k" else label, "group": group, "group_label": label,
                     "source_url": asset["source_url"], "size_bytes": asset["size_bytes"],
                     "target": (self.destination / asset["filename"] if asset.get("checksum_algorithm") == "md5"
                                else studio_data._cached_asset(asset, self.destination)),
                     "algorithm": asset["checksum_algorithm"], "expected": asset[asset["checksum_algorithm"]]})
            pin = common_rir_catalog()["archive"]
            add({"id": "common:rir_archive", "kind": "common_rir", "label": pin["filename"],
                 "group": "common_rir", "group_label": "평가용 공간 울림 · OpenSLR RIR",
                 "source_url": pin["source_url"], "size_bytes": pin["size_bytes"],
                 "target": self.destination / "common_rir" / pin["filename"],
                 "algorithm": "sha256", "expected": pin["sha256"]})
        for item in items:
            self.files[item["id"]] = {key: item[key] for key in ("id", "label", "group", "group_label")}
            self.files[item["id"]].update(state="waiting", done=0, total=item["size_bytes"], network_bytes=0,
                bytes_per_second=None, eta_seconds=None, **({"file_count": item["file_count"]} if item.get("file_count") else {}))
        if recipe.get("material_acquisition_policy") in ("automatic_v1", "automatic_v2", "automatic_v3"):
            plan = studio_data._automatic_source_plan(recipe["preset"], config["targets"]["generic_hours"], config["targets"]["korean_hours"])
            for component in plan["components"]:
                key = "automatic:" + component["id"]
                self.files[key] = {"id": key, "label": component["name"] + " · 목록 선정 후 묶음별 준비",
                    "group": key, "group_label": "대용량 학습 자료", "state": "waiting", "done": 0,
                    "total": None, "network_bytes": 0, "bytes_per_second": None, "eta_seconds": None,
                    "deferred": True, "target_hours": component["target_hours"]}
        self.report({"phase": "download_plan", "label": "전체 자료 다운로드 준비"})
        ordered = _provider_order(items)
        results = download_many(ordered, self._acquire, progress=self.report)
        for item, result in zip(ordered, results):
            if item["kind"] == "official_music":
                self.official_music = result
                continue
            path = Path(result["path"] if isinstance(result, dict) else result)
            if not self._remember(path, item):
                raise ValueError("Verified material changed before preparation; existing bytes are preserved")
            if item["kind"] == "base" and item["id"].startswith("base:"):
                self.base_paths[item["asset"]["id"]] = path
        # Duplicated pins can serve another original ID without another job.
        self.base_paths = {asset["id"]: self.base_paths.get(asset["id"], studio_data._cached_asset(asset, self.destination))
                           for asset in assets}

    def _acquire(self, item, report):
        from . import studio_data
        if item["kind"] == "official_music":
            from .official_music import prepare_official_music
            return prepare_official_music(self.shared, report, aggregate_progress=True)
        if item["kind"] == "base":
            return studio_data._download(item["asset"], self.destination, report, self.verification_cache)
        if item["kind"] == "common":
            from .evaluation_materials import _download_archive
            return _download_archive(item["asset"], self.destination, report, verification_cache=self.verification_cache)
        return download_checked(item["source_url"], item["target"], item["expected"], item["size_bytes"],
                                progress=report, verification_cache=self.verification_cache)
