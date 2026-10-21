"""Canonical transaction ownership, durable recovery and retained evidence."""
import os
from pathlib import Path

import pytest

from korean_openwakeword import canonical_sources as module
from korean_openwakeword.common import digest, read_json


def bound(path):
    return {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}


class Source:
    def __init__(self, directory):
        self.directory = directory
        self.raw = directory / "audio/source.wav"
        self.metadata = directory / "originals/publisher.json"
        self.calls = []

    def acquire(self):
        self.calls.append("acquire")
        self.raw.parent.mkdir(parents=True, exist_ok=True)
        self.metadata.parent.mkdir(parents=True, exist_ok=True)
        if not self.raw.exists():
            self.raw.write_bytes(b"pinned original audio" * 15)
        self.metadata.write_bytes(b'{"publisher":"pinned"}')
        return {"source": {"directory": str(self.raw.parent), "adapter": "fixture"},
                "rows": [{"path": self.raw.name, "sha256": digest(self.raw),
                          "canonical_source_id": "provider:speaker:recording"}],
                "receipts": [{"path": str(self.metadata), "sha256": digest(self.metadata)}],
                "reserved_download_bytes": self.raw.stat().st_size,
                "retired_archive_bytes": 17, "temporary_archive_peak_bytes": 17}

    def convert(self, prepared, output):
        self.calls.append("convert")
        output.mkdir(parents=True, exist_ok=True)
        canonical = output / "source.flac"
        canonical.write_bytes(b"canonical PCM losslessly encoded")
        index = output / "index.json"
        index.write_bytes(b'{"original_sha256":"' + digest(self.raw).encode() + b'"}')
        prepared["source"]["directory"] = str(output)
        prepared["rows"][0].update({"path": canonical.name, "sha256": digest(canonical),
                                    "original_sha256": digest(self.raw)})
        return {"prepared": prepared, "converted": [
                    {"original": bound(self.raw), "canonical": bound(canonical)}],
                "bindings": [bound(index)]}

    def run(self, **kwargs):
        return module.prepare_canonical_source(self.directory, request={"fixture": "v3"},
            acquire=self.acquire, canonicalize=self.convert, **kwargs)


def test_new_raw_is_retired_only_after_durable_canonical_commit(tmp_path, monkeypatch):
    source = Source(tmp_path / "v3")
    original_retire = module._retire_original
    def inspect(directory, mapping, claim):
        pending = read_json(directory / "canonical_pending.json")
        assert pending["converted"] == [mapping]
        assert source.raw.exists() and claim["retire_eligible"]
        assert Path(mapping["canonical"]["path"]).read_bytes() == b"canonical PCM losslessly encoded"
        original_retire(directory, mapping, claim)
    monkeypatch.setattr(module, "_retire_original", inspect)
    result = source.run()
    assert not source.raw.exists() and source.metadata.exists()
    assert source.calls == ["acquire", "convert"]
    assert result["retired_original_bytes"] == len(b"pinned original audio" * 15)
    assert result["reserved_download_bytes"] == result["retired_original_bytes"]
    assert result["retired_archive_bytes"] == 17 and result["temporary_archive_peak_bytes"] == 17
    assert result["storage_policy"] == "canonical_flac_only_v1"
    assert result["rows"][0]["canonical_source_id"] == "provider:speaker:recording"
    assert result["rows"][0]["original_sha256"] != result["rows"][0]["sha256"]
    assert all(Path(record["path"]).exists() for record in result["receipts"])
    calls = list(source.calls)
    again = source.run()
    assert again["resumed"] and source.calls == calls
    assert again["canonical_receipt"] == result["canonical_receipt"]


@pytest.mark.parametrize("after_unlink", [False, True])
def test_pending_commit_recovers_locally_after_retirement_interruption(tmp_path, monkeypatch, after_unlink):
    source = Source(tmp_path / "v3")
    retire = module._retire_original
    def interrupt(directory, mapping, claim):
        if after_unlink:
            retire(directory, mapping, claim)
        raise RuntimeError("fixture crash")
    monkeypatch.setattr(module, "_retire_original", interrupt)
    with pytest.raises(RuntimeError, match="fixture crash"):
        source.run()
    assert source.raw.exists() is not after_unlink
    monkeypatch.setattr(module, "_retire_original", retire)
    result = source.run()
    assert source.calls == ["acquire", "convert"]
    assert not source.raw.exists()
    assert module.verify_canonical_source(source.directory)["canonical_receipt"] == result["canonical_receipt"]


def test_preexisting_raw_and_external_hardlinks_are_preserved(tmp_path):
    for kind in ("existing", "hardlink"):
        source = Source(tmp_path / kind)
        if kind == "existing":
            source.raw.parent.mkdir(parents=True)
            source.raw.write_bytes(b"historical original")
        else:
            original_acquire = source.acquire
            shared = tmp_path / "external.wav"
            def acquire():
                result = original_acquire()
                os.link(source.raw, shared)
                return result
            source.acquire = acquire
        result = source.run()
        assert source.raw.is_file()
        assert result["retired_original_bytes"] == 0
        assert result["original_storage_bytes"] >= source.raw.stat().st_size + result["canonical_storage_bytes"]


def test_invalid_audio_omitted_from_mapping_is_preserved_for_existing_qc(tmp_path):
    source = Source(tmp_path / "v3")
    convert = source.convert
    def acquire():
        result = Source.acquire(source)
        (source.raw.parent / "broken.wav").write_bytes(b"bad input for existing QC")
        return result
    source.acquire = acquire
    source.convert = convert
    source.run()
    assert not source.raw.exists()
    assert (source.raw.parent / "broken.wav").read_bytes() == b"bad input for existing QC"


def test_conversion_failure_preserves_inputs_and_reuses_acquisition(tmp_path):
    source = Source(tmp_path / "v3")
    convert = source.convert
    source.convert = lambda *args: (_ for _ in ()).throw(RuntimeError("conversion failed"))
    with pytest.raises(RuntimeError, match="conversion failed"):
        source.run()
    assert source.raw.exists() and not (source.directory / "canonical_pending.json").exists()
    source.convert = convert
    source.run()
    assert source.calls == ["acquire", "convert"]


def test_corrupt_canonical_binding_never_retires_raw(tmp_path):
    source = Source(tmp_path / "v3")
    convert = source.convert
    def changed(prepared, output):
        result = convert(prepared, output)
        Path(result["converted"][0]["canonical"]["path"]).write_bytes(b"corrupt")
        return result
    source.convert = changed
    with pytest.raises(ValueError, match="changed"):
        source.run()
    assert source.raw.exists()


@pytest.mark.parametrize("change", ["metadata", "canonical", "pending", "acquisition", "request"])
def test_completed_cache_detects_tampering_without_new_work(tmp_path, change):
    source = Source(tmp_path / "v3")
    source.run()
    calls = list(source.calls)
    paths = {"metadata": source.metadata,
             "canonical": source.directory / "canonical/source.flac",
             "pending": source.directory / "canonical_pending.json",
             "acquisition": source.directory / "canonical_acquisition.json",
             "request": source.directory / "canonical_request.json"}
    path = paths[change]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed|missing|no longer binds"):
        source.run()
    assert source.calls == calls and not source.raw.exists()


def test_current_byte_limit_is_enforced_before_raw_retirement(tmp_path):
    source = Source(tmp_path / "v3")
    with pytest.raises(ValueError, match="storage bound"):
        source.run(max_retained_bytes=1)
    assert source.raw.exists()
    source.run(max_retained_bytes=1024**2)
    assert source.calls == ["acquire", "convert", "convert"]
    assert not source.raw.exists()
    with pytest.raises(ValueError, match="storage bound"):
        source.run(max_retained_bytes=1)


def test_changed_request_refuses_cache_without_mutation(tmp_path):
    source = Source(tmp_path / "v3")
    source.run()
    with pytest.raises(ValueError, match="another request"):
        module.prepare_canonical_source(source.directory, request={"fixture": "other"},
            acquire=source.acquire, canonicalize=source.convert)
    assert source.calls == ["acquire", "convert"]


def test_replaced_owned_raw_is_preserved_after_pending_commit(tmp_path, monkeypatch):
    source = Source(tmp_path / "v3")
    retire = module._retire_original
    def replace(directory, mapping, claim):
        replacement = source.raw.with_suffix(".new")
        replacement.write_bytes(source.raw.read_bytes())
        os.replace(replacement, source.raw)
        retire(directory, mapping, claim)
    monkeypatch.setattr(module, "_retire_original", replace)
    with pytest.raises(ValueError, match="shared or replaced"):
        source.run()
    assert source.raw.exists() and (source.directory / "canonical_pending.json").exists()


def test_retained_binding_cannot_escape_cache(tmp_path):
    source = Source(tmp_path / "v3")
    outside = tmp_path / "external.flac"
    outside.write_bytes(b"external")
    convert = source.convert
    def escape(prepared, output):
        result = convert(prepared, output)
        result["converted"][0]["canonical"] = bound(outside)
        return result
    source.convert = escape
    with pytest.raises(ValueError, match="stay inside"):
        source.run()
    assert source.raw.exists() and outside.read_bytes() == b"external"


def test_extra_semantic_verifier_runs_before_retirement(tmp_path):
    source = Source(tmp_path / "v3")
    def reject(prepared):
        assert source.raw.exists()
        raise ValueError("PCM parity rejected")
    with pytest.raises(ValueError, match="PCM parity"):
        source.run(verify=reject)
    assert source.raw.exists() and not (source.directory / "canonical_pending.json").exists()


def test_acquisition_interruption_before_claim_never_guesses_raw_ownership(tmp_path):
    source = Source(tmp_path / "v3")
    acquire = source.acquire
    def interrupted():
        acquire()
        raise RuntimeError("acquisition stopped before claim")
    source.acquire = interrupted
    with pytest.raises(RuntimeError, match="before claim"):
        source.run()
    assert source.raw.exists() and not (source.directory / "canonical_acquisition.json").exists()
    source.acquire = acquire
    result = source.run()
    assert source.raw.exists() and result["retired_original_bytes"] == 0


@pytest.mark.parametrize("historical", [False, True])
def test_real_codec_exact_pcm_retirement_and_offline_resume(tmp_path, historical):
    import numpy as np
    import soundfile as sf
    from korean_openwakeword.canonical_audio import canonicalize_prepared_source, verify_canonical_audio
    from korean_openwakeword.preparation import _convert

    source = Source(tmp_path / ("historical" if historical else "new"))
    samples = np.random.default_rng(73).normal(0, 0.1, (4417, 2))
    def make_audio():
        source.raw.parent.mkdir(parents=True, exist_ok=True)
        sf.write(source.raw, samples, 44100, subtype="PCM_24")
    if historical:
        make_audio()
    acquire = source.acquire
    def acquire_audio():
        if not source.raw.exists():
            make_audio()
        return acquire()
    source.acquire = acquire_audio
    source.convert = canonicalize_prepared_source
    reference = tmp_path / "reference.wav"
    # Generate the same known input outside the new acquisition namespace, so
    # the existing transformation establishes a byte-for-byte reference.
    fixture = tmp_path / "original.wav"
    sf.write(fixture, samples, 44100, subtype="PCM_24")
    _convert(fixture, reference)
    result = source.run()
    assert source.raw.exists() is historical
    row = {**result["rows"][0], "path": str(Path(result["source"]["directory"]) / result["rows"][0]["path"])}
    saved = verify_canonical_audio(row)
    assert row["original_source_sha256"] == digest(fixture)
    assert row["logical_audio_sha256"] == digest(reference)
    assert saved["original_info"]["input_sample_rate"] == 44100
    assert saved["original_info"]["input_channels"] == 2
    assert np.array_equal(sf.read(reference, dtype="int16")[0], sf.read(row["path"], dtype="int16")[0])
    assert result["retired_original_bytes"] == (0 if historical else fixture.stat().st_size)
    assert source.run()["canonical_receipt"] == result["canonical_receipt"]
    assert source.calls == ["acquire"]


def test_real_codec_invalid_audio_stays_available_for_established_qc(tmp_path):
    import numpy as np
    import soundfile as sf
    from korean_openwakeword.canonical_audio import canonicalize_prepared_source

    source = Source(tmp_path / "v3")
    def acquire_audio():
        source.raw.parent.mkdir(parents=True, exist_ok=True)
        sf.write(source.raw, np.sin(np.arange(1600) / 19) * 0.1, 16000, subtype="PCM_16")
        result = Source.acquire(source)
        broken = source.raw.parent / "broken.wav"
        broken.write_bytes(b"not an audio container")
        result["rows"].append({"path": broken.name, "sha256": digest(broken),
                               "canonical_source_id": "provider:invalid"})
        return result
    source.acquire = acquire_audio
    source.convert = canonicalize_prepared_source
    result = source.run()
    assert not source.raw.exists()
    broken = source.raw.parent / "broken.wav"
    assert broken.exists()
    rejected = result["rows"][1]
    alias = Path(result["source"]["directory"]) / rejected["path"]
    assert rejected["canonical_conversion_rejected"] and alias.read_bytes() == broken.read_bytes()
    assert "canonical_audio" not in rejected
    assert source.run()["canonical_receipt"] == result["canonical_receipt"]


def _real_source(tmp_path):
    import numpy as np
    import soundfile as sf
    from korean_openwakeword.canonical_audio import canonicalize_prepared_source
    source = Source(tmp_path / "v3")
    def acquire():
        if not source.raw.exists():
            source.raw.parent.mkdir(parents=True, exist_ok=True)
            sf.write(source.raw, np.sin(np.arange(1600) / 23) * 0.2, 16000, subtype="PCM_16")
        return Source.acquire(source)
    source.acquire = acquire
    source.convert = canonicalize_prepared_source
    return source


def test_real_codec_crash_after_flac_publish_resumes_without_overwriting_output(tmp_path, monkeypatch):
    from korean_openwakeword import canonical_audio
    source = _real_source(tmp_path)
    write_json = canonical_audio.write_json
    monkeypatch.setattr(canonical_audio, "write_json", lambda *args: (_ for _ in ()).throw(
        RuntimeError("fixture crash before sidecar commit")))
    with pytest.raises(RuntimeError, match="before sidecar"):
        source.run()
    flac = source.directory / "canonical/audio/source.wav.flac"
    before, info = digest(flac), flac.stat()
    assert source.raw.exists() and not flac.with_suffix(".flac.json").exists()
    monkeypatch.setattr(canonical_audio, "write_json", write_json)
    result = source.run()
    assert not source.raw.exists() and digest(flac) == before
    assert flac.stat().st_ino == info.st_ino and flac.stat().st_mtime_ns == info.st_mtime_ns
    assert source.calls == ["acquire"]
    assert module.verify_canonical_source(source.directory)["canonical_receipt"] == result["canonical_receipt"]


@pytest.mark.parametrize("error_kind", ["disk_full", "system_io"])
def test_real_codec_operational_failures_propagate_without_invalid_audio_approval(tmp_path, monkeypatch, error_kind):
    import errno
    import soundfile as sf
    from korean_openwakeword import preparation
    source = _real_source(tmp_path)
    error = OSError(errno.ENOSPC, "No space left on device") if error_kind == "disk_full" else sf.LibsndfileError(2)
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(preparation, "_convert", fail)
    with pytest.raises(type(error)):
        source.run()
    assert source.raw.exists() and not (source.directory / "canonical_pending.json").exists()
    assert not (source.directory / "canonical_receipt.json").exists()
