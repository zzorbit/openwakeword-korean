"""Native sample conditioning is separate from human review and call QC."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import os
from pathlib import Path, PureWindowsPath
import shutil

import numpy as np
import pytest
import soundfile as sf

from korean_openwakeword import clone_reference as preparation
from korean_openwakeword.common import digest, read_json, write_json


def tone(duration, rate=24000, amplitude=.04, frequency=173):
    return (amplitude * np.sin(2 * np.pi * frequency * np.arange(round(duration * rate)) / rate)).astype("float32")


def reference(root, samples, rate=24000, subtype="FLOAT"):
    root.mkdir(parents=True, exist_ok=True)
    source = root / "source.wav"
    sf.write(source, samples, rate, subtype=subtype)
    return {"path": "source.wav", "sha256": digest(source), "text": "보존할 기준 문장", "voice_family_id": "voice-one"}


def files(root):
    return {path.relative_to(root).as_posix(): digest(path) for path in root.rglob("*") if path.is_file()}


def padded_reference(root, rate=24000):
    wave = np.concatenate((np.zeros(round(rate * 1.2), dtype="float32"), tone(1.5, rate),
                           np.full(round(rate * 2.0), .00002, dtype="float32")))
    return reference(root, wave, rate), wave


@pytest.mark.parametrize("subtype", ["FLOAT", "PCM_16", "PCM_24"])
def test_normal_voice_keeps_original_bytes_without_creating_a_copy(tmp_path, subtype):
    original = reference(tmp_path, tone(1.6), subtype=subtype)
    before, supplied = files(tmp_path), deepcopy(original)
    path, metadata = preparation.prepare_clone_reference(tmp_path, original)
    assert path.samefile(tmp_path / "source.wav") and not metadata["trimmed"]
    assert metadata["source"] == metadata["prepared"]
    assert "derivation_receipt" not in metadata
    assert preparation.verify_clone_reference(tmp_path, original, metadata) == path
    assert files(tmp_path) == before and original == supplied
    json.dumps(metadata, allow_nan=False)


@pytest.mark.parametrize("rate", [16000, 24000, 48000])
def test_only_long_edges_are_removed_with_native_samples_and_context(tmp_path, rate):
    quiet_consonant = tone(.3, rate, amplitude=.0003)
    wave = np.concatenate((np.zeros(round(1.4 * rate), dtype="float32"), quiet_consonant,
                           tone(.8, rate), np.zeros(rate, dtype="float32"), tone(.6, rate),
                           quiet_consonant, np.full(2 * rate, .00002, dtype="float32")))
    original = reference(tmp_path, wave, rate)
    original_hash = digest(tmp_path / "source.wav")
    path, metadata = preparation.prepare_clone_reference(tmp_path, original)
    actual, actual_rate = sf.read(path, dtype="float32")
    start, end = metadata["start_frame"], metadata["end_frame"]
    assert actual_rate == rate and np.array_equal(actual, wave[start:end])
    assert start <= round(1.2 * rate) and end >= len(wave) - round(1.8 * rate)
    # Both quiet consonants and the one-second pause between voiced parts survive.
    assert np.array_equal(actual[round(1.4 * rate) - start:round(1.7 * rate) - start], quiet_consonant)
    assert np.array_equal(actual[round(2.5 * rate) - start:round(3.5 * rate) - start], np.zeros(rate, dtype="float32"))
    assert metadata["prepared"]["sha256"] == digest(path)
    assert metadata["source"]["sha256"] == original_hash == digest(tmp_path / "source.wav")
    assert metadata["source_scope"].endswith("not human review or voice replacement")
    assert not any(key in metadata for key in ("listened", "decision", "accepted"))
    assert preparation.verify_clone_reference(tmp_path, original, metadata) == path


def test_short_normal_edges_and_quiet_speech_are_not_tightened(tmp_path):
    wave = np.concatenate((np.zeros(12000, dtype="float32"), tone(2, amplitude=.004), np.zeros(12000, dtype="float32")))
    original = reference(tmp_path, wave)
    path, metadata = preparation.prepare_clone_reference(tmp_path, original)
    assert not metadata["trimmed"] and path.samefile(tmp_path / "source.wav")


def test_bulk_prompt_compacts_only_long_internal_quiet_and_verifies_old_receipt(tmp_path):
    rate = 24000
    wave = np.concatenate((np.zeros(rate), tone(.8), np.full(16 * rate, .00002),
                           tone(.6, frequency=211), np.zeros(rate), tone(.8, frequency=275), np.zeros(rate)))
    wave = wave.astype("float32")
    original = reference(tmp_path, wave)
    legacy_path, legacy = preparation.prepare_clone_reference(tmp_path, original)
    before = files(tmp_path)

    path, metadata = preparation.prepare_clone_reference(tmp_path, original, compact_internal_silence=True)
    assert metadata["policy"] == preparation.INTERNAL_SILENCE_POLICY
    assert metadata["retained_frame_ranges"] == [[round(.8 * rate), 2 * rate], [round(17.6 * rate), round(20.4 * rate)]]
    assert metadata["internal_removed_s"] == pytest.approx(15.6)
    actual, actual_rate = sf.read(path, dtype="float32")
    assert actual_rate == rate
    assert np.array_equal(actual, np.concatenate((wave[round(.8 * rate):2 * rate],
                                                  wave[round(17.6 * rate):round(20.4 * rate)])))
    assert preparation.verify_clone_reference(tmp_path, original, legacy) == legacy_path
    assert preparation.verify_clone_reference(tmp_path, original, metadata) == path
    assert all(files(tmp_path)[name] == sha for name, sha in before.items())

    tampered = deepcopy(metadata)
    tampered["retained_frame_ranges"][0][1] -= 1
    with pytest.raises(ValueError):
        preparation.verify_clone_reference(tmp_path, original, tampered)
    assert preparation.prepare_clone_reference(tmp_path, original, compact_internal_silence=True) == (path, metadata)


def test_bulk_prompt_without_long_quiet_retains_original_conditioning_record(tmp_path):
    original = reference(tmp_path, np.concatenate((tone(1), np.zeros(24000, dtype="float32"), tone(1))))
    before = files(tmp_path)
    assert (preparation.prepare_clone_reference(tmp_path, original, compact_internal_silence=True)
            == preparation.prepare_clone_reference(tmp_path, original))
    assert files(tmp_path) == before


@pytest.mark.parametrize("kind", ["silence", "nonfinite", "stereo", "clipping", "near_silence"])
def test_invalid_sources_are_not_repaired_by_gain_downmix_or_relaxed_quality(tmp_path, kind):
    wave = tone(1.2)
    if kind == "silence":
        wave[:] = 0
    elif kind == "nonfinite":
        wave[100] = np.nan
    elif kind == "stereo":
        wave = np.column_stack((wave, wave))
    elif kind == "clipping":
        wave[:] = 1.1
    elif kind == "near_silence":
        wave *= .0001
    original = reference(tmp_path, wave)
    before = files(tmp_path)
    with pytest.raises(ValueError):
        preparation.prepare_clone_reference(tmp_path, original)
    assert files(tmp_path) == before


def test_cache_reuse_and_readonly_verification_do_not_republish_evidence(tmp_path):
    original, _ = padded_reference(tmp_path)
    path, metadata = preparation.prepare_clone_reference(tmp_path, original)
    before, supplied = files(tmp_path), deepcopy(original)
    again, repeated = preparation.prepare_clone_reference(tmp_path, original)
    assert (again, repeated) == (path, metadata)
    assert preparation.verify_clone_reference(tmp_path, original, repeated) == path
    assert files(tmp_path) == before and original == supplied


def test_source_binding_cannot_change_between_analysis_and_receipt_plan(tmp_path, monkeypatch):
    original, _ = padded_reference(tmp_path)
    actual_digest = preparation.digest
    reads = 0
    def changing_digest(path):
        nonlocal reads
        if Path(path) == tmp_path / "source.wav":
            reads += 1
            if reads == 2:
                return "f" * 64
        return actual_digest(path)
    monkeypatch.setattr(preparation, "digest", changing_digest)
    before = files(tmp_path)
    with pytest.raises(ValueError, match="원본이 변경"):
        preparation.prepare_clone_reference(tmp_path, original)
    assert files(tmp_path) == before


def test_derived_quality_cannot_hide_clipping_diluted_by_a_long_silent_tail(tmp_path):
    wave = np.concatenate((tone(1), np.zeros(10 * 24000, dtype="float32")))
    wave[12000:12960] = 1.0  # <1% of original, >1% of the retained speech.
    original = reference(tmp_path, wave)
    from korean_openwakeword.reference_qc import _check_wave
    assert _check_wave(tmp_path / "source.wav")[0] == []
    before = files(tmp_path)
    with pytest.raises(ValueError, match="severe_clipping"):
        preparation.prepare_clone_reference(tmp_path, original)
    assert files(tmp_path) == before


@pytest.mark.parametrize("part", ["source", "derived", "receipt", "metadata", "policy"])
def test_tampered_source_derived_receipt_or_policy_is_rejected_without_replacement(tmp_path, part):
    original, _ = padded_reference(tmp_path)
    path, metadata = preparation.prepare_clone_reference(tmp_path, original)
    if part == "source":
        with (tmp_path / "source.wav").open("ab") as stream:
            stream.write(b"changed source")
    elif part == "derived":
        values, rate = sf.read(path, dtype="float32")
        values[100] = .123
        sf.write(path, values, rate, subtype="FLOAT")
    elif part == "receipt":
        record_path = tmp_path / metadata["derivation_receipt"]["path"]
        record = read_json(record_path)
        record["start_frame"] += 1
        write_json(record_path, record)
    elif part == "metadata":
        metadata["end_frame"] -= 1
    elif part == "policy":
        metadata["policy"] = "unknown"
    before = files(tmp_path)
    with pytest.raises(ValueError):
        preparation.verify_clone_reference(tmp_path, original, metadata)
    if part not in ("metadata", "policy"):
        with pytest.raises(ValueError):
            preparation.prepare_clone_reference(tmp_path, original)
    assert files(tmp_path) == before


def test_interrupted_audio_publish_can_recover_only_its_exact_expected_receipt(tmp_path):
    original, _ = padded_reference(tmp_path)
    path, metadata = preparation.prepare_clone_reference(tmp_path, original)
    record = tmp_path / metadata["derivation_receipt"]["path"]
    receipt_bytes = record.read_bytes()
    record.unlink()
    before = files(tmp_path)
    with pytest.raises(ValueError):
        preparation.verify_clone_reference(tmp_path, original, metadata)
    assert files(tmp_path) == before  # Read verification cannot perform recovery.
    recovered_path, recovered = preparation.prepare_clone_reference(tmp_path, original)
    assert recovered_path == path and recovered == metadata and record.read_bytes() == receipt_bytes
    path.unlink()
    before = files(tmp_path)
    with pytest.raises(ValueError, match="사본 음원"):
        preparation.prepare_clone_reference(tmp_path, original)
    assert files(tmp_path) == before


def test_concurrent_threads_share_one_complete_immutable_preparation(tmp_path):
    original, _ = padded_reference(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: preparation.prepare_clone_reference(tmp_path, original), range(8)))
    assert all(result == results[0] for result in results)
    assert len(list((tmp_path / "data/clone_references").rglob("*.wav"))) == 1
    assert not list(tmp_path.rglob(".writer.lock")) and not list(tmp_path.rglob("*.tmp.wav"))


def test_windows_verbatim_drive_and_unc_names_preserve_workspace_containment():
    names = preparation._path_identity
    for ordinary, verbatim in ((r"F:\project\data\clone_references", r"\\?\F:\project\data\clone_references"),
                               (r"\\server\share\project\data", r"\\?\UNC\server\share\project\data")):
        a, b = PureWindowsPath(ordinary), PureWindowsPath(verbatim)
        assert names(a) == names(b)
        assert names(a / "child").is_relative_to(names(b))
        assert names(a / "child").relative_to(names(b)).as_posix() == "child"
    assert not names(PureWindowsPath(r"\\?\F:\outside\data")).is_relative_to(names(PureWindowsPath(r"F:\project")))
    assert names(PureWindowsPath(r"\\.\PhysicalDrive0")) == PureWindowsPath(r"\\.\PhysicalDrive0")


@pytest.mark.skipif(os.name != "nt", reason="Windows resolve namespace regression")
def test_verbatim_resolution_during_cache_creation_keeps_stable_relative_receipts(tmp_path, monkeypatch):
    original, _ = padded_reference(tmp_path)
    actual_resolve = Path.resolve
    mode = "verbatim"
    def namespace_resolve(path, *args, **kwargs):
        resolved = actual_resolve(path, *args, **kwargs)
        if "clone_references" in resolved.parts:
            name = str(resolved)
            if name.startswith("\\\\?\\"):
                name = name[4:]
            return Path("\\\\?\\" + name if mode == "verbatim" else name)
        return resolved
    monkeypatch.setattr(Path, "resolve", namespace_resolve)
    first, metadata = preparation.prepare_clone_reference(tmp_path, original)
    assert not Path(metadata["prepared"]["path"]).is_absolute()
    assert not Path(metadata["derivation_receipt"]["path"]).is_absolute()
    before = files(tmp_path)
    mode = "ordinary"
    second, repeated = preparation.prepare_clone_reference(tmp_path, original)
    assert second == first and metadata == repeated
    assert str(second).startswith("\\\\?\\")
    assert preparation.verify_clone_reference(tmp_path, original, metadata) == first
    assert files(tmp_path) == before


def test_physically_resolved_cache_escape_is_still_rejected(tmp_path, monkeypatch):
    original, _ = padded_reference(tmp_path)
    actual_resolve = Path.resolve
    escaped = tmp_path.parent / "outside-clone-cache"
    def outside_resolve(path, *args, **kwargs):
        resolved = actual_resolve(path, *args, **kwargs)
        if resolved.name == "clone_references":
            return escaped
        return resolved
    monkeypatch.setattr(Path, "resolve", outside_resolve)
    before = files(tmp_path)
    with pytest.raises(ValueError, match="프로젝트 밖"):
        preparation.prepare_clone_reference(tmp_path, original)
    assert files(tmp_path) == before and not escaped.exists()


def test_actual_reported_reference_is_only_read_and_its_copy_has_shorter_quiet_tail(tmp_path):
    supplied = os.environ.get("KOWW_CLONE_REFERENCE_REAL_FIXTURE")
    if not supplied:
        pytest.skip("Optional read-only real reference fixture not supplied")
    source = Path(supplied)
    before = digest(source)
    shutil.copyfile(source, tmp_path / "source.wav")
    original = {"path": "source.wav", "sha256": before}
    prepared, metadata = preparation.prepare_clone_reference(tmp_path, original)
    assert metadata["source_duration_s"] == pytest.approx(20.4)
    assert 3 < metadata["prepared_duration_s"] < 6
    assert metadata["trailing_removed_s"] > 14
    assert sf.info(prepared).samplerate == 24000
    assert digest(source) == before
