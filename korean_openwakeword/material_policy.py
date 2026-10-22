"""Hash-bound technical admission for the library's fixed public sources.

This policy records no human answers or listening. Custom material keeps the
ordinary review path. Classification is repeated when evidence is consumed;
names and a policy marker alone never establish a fixed-library source.
"""
from pathlib import Path
import os
import re

from .common import bound_file, digest, identity, iter_rows, read_json, resolved, write_json
from .project import load_project

POLICY_ID = "library_fixed_sources_v1"
ARTIFACT_TYPE = "library_fixed_catalog_technical_validation"
CONTENT_REVIEW = "library_fixed_sources_technical_validation"
_KIND = "library_fixed_sources_evidence_v1"
_TRAIN = {"generic_negative", "korean_negative", "noise", "rir"}
_BASE = {"zeroth": {"korean_negative", "noise"}, "audioset": {"noise"},
         "fma": {"noise"}, "mit_rir": {"rir"}}
_AUTO = {"audioset_unbalanced": {"generic_negative", "noise"},
         "mls": {"generic_negative", "noise"}, "fma_full": {"generic_negative", "noise"},
         "yodas2": {"korean_negative", "noise"}}


def _require(condition, message):
    if not condition:
        raise ValueError("Fixed library material policy: " + message)


def _binding(path):
    return {"path": str(Path(path).resolve()), "sha256": digest(path)}


def _hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _path_key(path):
    # Contract paths are already absolute. Lexical lookup avoids a filesystem
    # resolve for every unselected recording; selected rows are resolved below.
    return os.path.normcase(os.path.abspath(path))


def _catalog_pins():
    from . import studio_data
    from .commercial_sources import _PROVIDERS
    package = Path(__file__).parent
    return {"resources": {p.name: digest(p) for p in sorted((package / "resources").glob("*.json"))},
            "official_assets": identity(studio_data._official_assets()),
            "official_revisions": {name: getattr(studio_data, name) for name in
                                   ("AS_REV", "MIT_REV", "FMA_REV", "DNS_REV")},
            "automatic_providers": identity(_PROVIDERS)}


def _workspace(root, prepared):
    selected = prepared.get("workspace")
    if selected is None:
        workflow = read_json(root / ".studio/workflow.json")
        _require(Path(workflow.get("project", "")).resolve() == root / "project.json",
                 "workspace owner differs from the project")
        selected = workflow.get("storage_workspace")
    _require(isinstance(selected, str) and bool(selected), "selected workspace is missing")
    path = Path(selected).resolve()
    _require(not any(part.lower() in (".codex", ".git") for part in path.parts), "invalid workspace")
    return path


def _record(root, binding, lineage):
    path = bound_file(root, binding)
    actual = {"path": str(path), "sha256": binding["sha256"]}
    lineage[actual["path"]] = actual
    return read_json(path)


def _metadata_file(path, lineage):
    binding = _binding(path)
    lineage[binding["path"]] = binding
    return read_json(path)


def _asset_receipt(asset, downloads, lineage, archive=None):
    archive = Path(archive) if archive else downloads / asset["filename"]
    choices = [archive.with_name(archive.name + ".receipt.json"),
               downloads / (asset["filename"] + ".reused.receipt.json")]
    path = next((p for p in choices if p.is_file()), None)
    _require(path is not None, "pinned download receipt is missing: " + asset["id"])
    record = read_json(path)
    _require(Path(record.get("path", "")).resolve() == archive.resolve()
             and record.get("bytes") == asset["size_bytes"]
             and record.get("source_url") == asset["source_url"] and _hash(record.get("sha256")),
             "download receipt differs from the pinned asset")
    algorithm = asset["checksum_algorithm"]
    if algorithm == "sha256":
        _require(record["sha256"] == asset["sha256"], "archive SHA256 pin changed")
    elif algorithm == "md5":
        _require(record.get("publisher_md5") == asset["md5"], "archive MD5 pin changed")
    else:
        _require(record.get("verified_upstream_algorithm") == algorithm
                 and record.get("verified_upstream_checksum") == asset[algorithm], "upstream archive pin changed")
    # Downloaders legitimately refresh completed_at/reused in this shared
    # summary. Bind its verified, immutable source fields, so another project's
    # normal cache reuse cannot invalidate an existing experiment's evidence.
    fields = {key: record[key] for key in ("path", "sha256", "bytes", "source_url",
              "publisher_md5", "verified_upstream_algorithm", "verified_upstream_checksum") if key in record}
    lineage[str(path.resolve())] = {"path": str(path.resolve()), "validation": "pinned_download_receipt_fields",
                                  "fields": fields, "identity": identity(fields)}
    return record


def _extraction(directory, asset, downloads, lineage):
    directory = directory.resolve()
    receipt = _metadata_file(directory / "extraction_receipt.json", lineage)
    original = receipt.get("request", {}).get("archive", {})
    archive = _asset_receipt(asset, downloads, lineage, original.get("path"))
    _require(original.get("sha256") == archive["sha256"], "extraction is not bound to the pinned download")
    files = receipt.get("files")
    _require(isinstance(files, list) and bool(files), "extraction inventory is missing")
    result = {}
    for item in files:
        relative = Path(item["path"])
        path = directory / relative
        _require(not relative.is_absolute() and ".." not in relative.parts
                 and _hash(item.get("sha256")) and _path_key(path) not in result, "invalid extraction member")
        result[_path_key(path)] = item["sha256"]
    return result


def _base_lineage(root, workspace, key, lineage):
    from .studio_data import _official_assets, MIT_REV, MIT_RIR_COUNT
    assets = {a["id"]: a for a in _official_assets()}
    shared, downloads = workspace / "shared", workspace / "shared/downloads"
    directories = {"zeroth": shared / "zeroth", "audioset": shared / "audioset",
                   "fma": shared / "fma_1h", "mit_rir": shared / "mit_rir/16khz"}
    directory = directories[key]
    metadata_pins = {}
    if key == "mit_rir":
        tree = _metadata_file(shared / "mit_rir/provider_tree.json", lineage)
        expected = f"https://huggingface.co/api/datasets/davidscripka/MIT_environmental_impulse_responses/tree/{MIT_REV}/16khz?limit=1000"
        _require(tree.get("source_url") == expected, "MIT provider revision changed")
        entries = [r for r in tree.get("entries", []) if r.get("type") == "file" and r.get("path", "").endswith(".wav")]
        _require(len(entries) == MIT_RIR_COUNT and len({r["path"] for r in entries}) == MIT_RIR_COUNT,
                 "MIT provider inventory changed")
        originals = {}
        for entry in entries:
            path = (shared / "mit_rir" / entry["path"]).resolve()
            _require(path.is_relative_to(directory.resolve()), "MIT provider member escaped its corpus")
            asset = {"id": entry["path"], "filename": path.name, "size_bytes": entry["size"],
                     "source_url": f"https://huggingface.co/datasets/davidscripka/MIT_environmental_impulse_responses/resolve/{MIT_REV}/{entry['path']}",
                     "checksum_algorithm": "sha256", "sha256": entry.get("lfs", {}).get("oid")}
            _require(_hash(asset["sha256"]), "MIT provider file lacks an original hash")
            _asset_receipt(asset, path.parent, lineage, path)
            originals[_path_key(path)] = asset["sha256"]
    else:
        originals = _extraction(directory, assets[key], downloads, lineage)
        if key == "audioset":
            for name in ("audioset_segments", "audioset_ontology"):
                receipt = _asset_receipt(assets[name], downloads, lineage)
                metadata_pins[str(Path(receipt["path"]).resolve())] = assets[name]["sha256"]
        if key == "fma":
            metadata_pins = _extraction(shared / "fma_metadata", assets["fma_metadata"], downloads, lineage)
    index = root / ".studio/materials" / (key + "_source_index.jsonl")
    return directory, index, originals, metadata_pins


def _metadata_graph(root, values, lineage):
    """Follow hash-bound JSON receipts, never audio payload or raw archives."""
    pending = [values]
    records = []
    visited = set()
    while pending:
        value = pending.pop()
        if isinstance(value, list):
            pending.extend(value)
        elif isinstance(value, dict):
            if (set(("path", "sha256")).issubset(value) and str(value["path"]).lower().endswith(".json")
                    and (set(value) == {"path", "sha256"} or Path(value["path"]).is_absolute())):
                path = resolved(root, value["path"])
                token = (str(path), value["sha256"])
                if token not in visited:
                    visited.add(token)
                    record = _record(root, value, lineage)
                    if isinstance(record, dict):
                        records.append((path, record))
                    pending.append(record)
            else:
                pending.extend(value.values())
    return records


def _automatic_lineage(root, workspace, prepared, source_key, component, lineage):
    from .commercial_sources import _PROVIDERS, _FMA_FILES, _validate_resolution
    policy = prepared.get("material_acquisition_policy")
    _require(policy in ("automatic_v1", "automatic_v2", "automatic_v3"), "automatic acquisition policy missing")
    namespace = "automatic_sources_" + policy.removeprefix("automatic_")
    stage = root / ".studio/materials" / namespace
    request = _metadata_file(stage / "request.json", lineage)
    _, config = load_project(root / "project.json")
    _require(request.get("workspace") == str(workspace) and request.get("wakeword") == config["wakeword"]
             and request.get("project_id") == config["project_id"]
             and request.get("providers") == {k: _PROVIDERS[k]["revision"] for k in _AUTO},
             "automatic request lost its pinned provider/project binding")
    key = source_key.removeprefix("automatic_")
    _require(source_key == "automatic_" + key and re.fullmatch(r"[A-Za-z0-9_-]{1,80}", key), "unsafe automatic part")
    part = stage / "parts" / key
    record = _metadata_file(part / "source.json", lineage)
    source = record.get("source", {})
    _require(record.get("candidate", {}).get("component") == component
             and source.get("id") == source_key and source.get("component") == component
             and source.get("provider") == component, "automatic part differs from its selected provider")
    directory, index = part / "selected", part / "source_index.jsonl"
    _require(Path(source.get("directory", "")).resolve() == directory.resolve()
             and Path(source.get("source_index", "")).resolve() == index.resolve()
             and digest(index) == record.get("source_index_sha256"), "automatic source index changed")
    records = _metadata_graph(root, [record.get("receipts", []),
                                   record.get("canonical_receipt"), record.get("compact_receipt")], lineage)
    _require(bool(records), "automatic acquisition receipts are missing")
    resolutions = [r for _, r in records if r.get("kind") == "commercial_source_resolution_v1"]
    metadata = source.get("metadata")
    if metadata and Path(metadata).name == "source_resolution.json":
        resolutions.append(_metadata_file(metadata, lineage))
    if component != "fma_full":
        _require(bool(resolutions), "automatic source lacks pinned provider resolution")
        for resolution in resolutions:
            _validate_resolution(resolution)
            _require(resolution["source_id"] == component, "automatic provider resolution differs")
    else:
        _require(any(r.get("kind") in ("fma_selective_download_v1", "fma_streaming_source_v1") for _, r in records),
                 "FMA needs selective original-track acquisition evidence")
        plans = [r for _, r in records if r.get("kind") == "fma_selective_plan_v1"]
        _require(bool(plans), "FMA pinned selection plan is missing")
        for plan in plans:
            audio = plan.get("sources", {}).get("audio", {})
            metadata_pin = plan.get("sources", {}).get("metadata", {})
            _require(audio.get("url") == "https://os.unil.cloud.switch.ch/fma/fma_full.zip"
                     and audio.get("size_bytes") == _FMA_FILES["fma_full.zip"][0]
                     and plan.get("whole_archive_sha1_reference") == _FMA_FILES["fma_full.zip"][1]
                     and metadata_pin.get("url") == "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
                     and metadata_pin.get("size_bytes") == _FMA_FILES["fma_metadata.zip"][0],
                     "FMA selection has no original provider URL")
    index_rows = list(iter_rows(index))
    originals = {_path_key(directory / r["path"]): r.get("original_source_sha256", r.get("sha256")) for r in index_rows}
    _require(all(_hash(v) for v in originals.values()), "automatic source original hashes are missing")
    return directory, index, originals, {}


def _train(root, workspace, prepared, queue, entry, component, lineage):
    key = entry["id"]
    allowed = _AUTO[component] if component else _BASE[key]
    pools = queue.get("pools")
    _require(isinstance(pools, list) and bool(pools) and len(set(pools)) == len(pools)
             and set(pools).issubset(allowed) and pools == entry.get("pools"), "unsupported training pool")
    prep_binding = entry["preparation"]
    prep = _record(root, prep_binding, lineage)
    _, config = load_project(root / "project.json")
    _require(prep.get("kind") == "public_audio_preparation_v1" and prep.get("wakeword") == config["wakeword"],
             "training preparation differs from this project")
    contract = _record(root, prep["contract"], lineage)
    _require(contract.get("kind") == "public_audio_preparation_contract_v2"
             and contract.get("wakeword") == config["wakeword"], "preparation contract changed")
    source = prep["artifacts"]["source_manifest"]
    previews = prep["artifacts"]["representative_queue"]
    _require(resolved(root, queue["pool_manifest"]) == resolved(root, source["path"])
             and resolved(root, queue["queue"]) == resolved(root, previews["path"]), "review queue/source manifest changed")
    bound_file(root, previews)
    rows = list(iter_rows(bound_file(root, source)))
    _require(bool(rows), "training source manifest is empty")
    directory, index, originals, metadata_pins = (_automatic_lineage(root, workspace, prepared, key, component, lineage)
                                                  if component else _base_lineage(root, workspace, key, lineage))
    lineage[str(index.resolve())] = _binding(index)
    index_rows = {r.get("canonical_source_id"): r for r in iter_rows(index) if not r.get("excluded")}
    contracts = {(r.get("canonical_source_id"), _path_key(r["path"])): r for r in contract.get("sources", [])}
    prefix = {"zeroth": "zeroth:", "audioset": "audioset:", "fma": "fma:", "mit_rir": "mit_ir:",
              "audioset_unbalanced": "audioset:", "mls": "mls:", "fma_full": "fma:", "yodas2": "yodas:"}[component or key]
    checked_metadata = set()
    for row in rows:
        original_path = Path(row.get("original_path", "")).resolve()
        canonical = row.get("canonical_source_id", "")
        original = contracts.get((canonical, _path_key(row.get("original_path", ""))))
        indexed = index_rows.get(canonical)
        _require(row.get("provenance") == "public_dataset" and row.get("split") == "train"
                 and row.get("technical_pass") is True and canonical.startswith(prefix)
                 and original_path.is_relative_to(directory.resolve()) and original is not None and indexed is not None,
                 "source is not a technically checked fixed-provider Train recording")
        _require((directory / indexed["path"]).resolve() == original_path
                 and originals.get(_path_key(original_path)) == row.get("original_sha256") == original.get("original_sha256"),
                 "source original hash is not linked to its provider acquisition")
        for field in ("dataset_id", "canonical_recording_id", "source_url", "license", "category", "language", "speech"):
            _require(row.get(field) == original.get(field) == indexed.get(field), "provider source metadata changed: " + field)
        for binding in row.get("provider_metadata_bindings", []):
            token = (binding.get("path"), binding.get("sha256"))
            if token in checked_metadata:
                continue
            bound_file(root, binding)
            path = str(resolved(root, binding["path"]))
            if metadata_pins:
                _require(metadata_pins.get(_path_key(path), metadata_pins.get(path)) == binding["sha256"], "provider metadata is not in the pinned extraction/download")
            lineage[path] = {"path": path, "sha256": binding["sha256"]}
            checked_metadata.add(token)
        if metadata_pins:
            _require(bool(row.get("provider_metadata_bindings")), "source lacks original provider metadata bindings")
    return "train", sorted(pools), prep_binding, source, previews


def _evaluation(root, workspace, prepared, queue, lineage):
    from .evaluation_materials import evaluation_material_catalog
    from .evaluation_rir import common_rir_catalog
    catalog, rir_catalog = evaluation_material_catalog(), common_rir_catalog()
    common_catalog_id = identity(catalog)[:16]
    common_stage = (workspace / "shared" / ("common_evaluation_30h_" + common_catalog_id)).resolve()
    catalog_id = identity([catalog, rir_catalog, {"evaluation_source_scope": "common_30h_only"}])[:16]
    binding = prepared["evaluation_preparation"]
    prep = _record(root, binding, lineage)
    _, config = load_project(root / "project.json")
    _require(prep.get("kind") == "studio_evaluation_noise_preparation_v6" and prep.get("recipe_version") == 6
             and prep.get("catalog_id") == catalog_id and prep.get("wakeword") == config["wakeword"]
             and prep.get("evaluation_source_scope") == "common_30h_only" and prep.get("evaluation_only") is True
             and prep.get("training_admitted") is False and prep.get("intended_use") == "held_out_development_noise"
             and not prep.get("inherited_admissions"), "evaluation preparation is outside the fixed common recipe")
    _require(queue.get("id") == "materials_evaluation_v6_" + catalog_id and queue.get("pools") == []
             and queue.get("evaluation_only") is True, "evaluation queue scope changed")
    source, previews = (prep["artifacts"][k] for k in ("source_manifest", "representative_queue"))
    _require(resolved(root, queue["pool_manifest"]) == resolved(root, source["path"])
             and resolved(root, queue["queue"]) == resolved(root, previews["path"]), "evaluation queue/source changed")
    bound_file(root, previews)
    rows = list(iter_rows(bound_file(root, source)))
    _require(bool(rows), "evaluation source manifest is empty")
    known = {r["canonical_source_id"]: r for values in catalog["selections"].values() for r in values}
    rir_pins = {r["canonical_source_id"]: r for r in rir_catalog["assets"]}
    common_records = {}
    zeroth_originals = None
    for row in rows:
        _require(row.get("provenance") == "public_dataset" and row.get("split") == "validation"
                 and row.get("technical_pass") is True and row.get("evaluation_only") is True
                 and row.get("training_admitted") is False and row.get("evaluation_source_scope") == "common_30h_only"
                 and _hash(row.get("original_sha256")), "invalid fixed evaluation source")
        if row.get("category") == "rir":
            pin = rir_pins.get(row.get("canonical_source_id"))
            _require(pin is not None and row.get("sha256") == pin["sha256"]
                     and row.get("canonical_room_id") == pin["canonical_room_id"]
                     and row.get("source_url") == rir_catalog["source_url"], "evaluation RIR pin changed")
            continue
        _require(row.get("evaluation_material_catalog_id") == common_catalog_id, "evaluation material catalog changed")
        common_path = Path(row["path"]).parent.parent / "preparation.json"
        _require(common_path.resolve().is_relative_to(common_stage), "evaluation recording escaped its fixed preparation")
        if str(common_path) not in common_records:
            record = _metadata_file(common_path, lineage)
            _require(record.get("kind") == "common_evaluation_materials_v2"
                     and record.get("source_pool_scope") == "common_30h_only"
                     and record.get("catalog_id") == common_catalog_id and record.get("target_met") is True,
                     "evaluation source preparation changed")
            common_records[str(common_path)] = {r["sample_id"]: r for r in record["source_rows"]}
        original = common_records[str(common_path)].get(row["sample_id"])
        _require(original is not None and all(row.get(k) == value for k, value in original.items()),
                 "evaluation source differs from its technical preparation")
        pin = known.get(row.get("canonical_source_id"))
        if row.get("eval_bucket") == "korean_read":
            _require(row.get("canonical_source_id", "").startswith("zeroth:")
                     and row.get("source_url") == "https://www.openslr.org/40/", "evaluation Zeroth provider changed")
            if zeroth_originals is None:
                zeroth_originals = _extraction(workspace / "shared/zeroth", next(a for a in catalog["archives"] if a["id"] == "zeroth"),
                                               workspace / "shared/downloads", lineage)
            _require(zeroth_originals.get(_path_key(Path(row["original_path"]).resolve())) == row["original_sha256"], "evaluation Zeroth original hash changed")
        else:
            _require(pin is not None and all(row.get(k) == pin.get(k) for k in
                     ("dataset_id", "canonical_recording_id", "source_url", "license", "category", "eval_bucket", "eval_type")),
                     "evaluation recording is absent from the fixed catalog")
            if pin.get("sha256"):
                _require(row["original_sha256"] == pin["sha256"], "evaluation original pin changed")
    for asset in catalog["archives"]:
        _asset_receipt(asset, workspace / "shared/downloads", lineage)
    rir_receipt = _metadata_file(resolved(root, binding["path"]).parent / "rir/receipt.json", lineage)
    _require(rir_receipt.get("kind") == "common_rir_preparation_v1"
             and rir_receipt.get("catalog_sha256") == identity(rir_catalog), "common RIR preparation changed")
    _asset_receipt({**rir_catalog["archive"], "id": "common_rir", "checksum_algorithm": "sha256"},
                   workspace / "shared/downloads/common_rir", lineage)
    return "evaluation", ["evaluation"], binding, source, previews


def _payload(project, prepared, queue):
    project = Path(project).resolve()
    root, config = load_project(project)
    if prepared.get("kind") != "studio_material_preparation_v1" or Path(prepared.get("project", "")).resolve() != project:
        return None
    if not any(all(q.get(k) == queue.get(k) for k in ("id", "queue", "pool_manifest", "preparation", "pools", "evaluation_only", "automatic_component"))
               for q in prepared.get("review_queues", [])):
        return None
    evaluation = queue.get("evaluation_only") is True
    entry = None
    component = queue.get("automatic_component")
    if evaluation:
        if not prepared.get("evaluation_preparation"):
            return None
        binding = prepared["evaluation_preparation"]
        source_key = "common_evaluation_v6"
    else:
        entry = next((e for e in prepared.get("preparations", [])
                      if queue.get("id") == "materials_" + e.get("id", "")), None)
        if entry is None or (entry["id"] not in _BASE and component not in _AUTO):
            return None
        _require(not component or component in _AUTO and entry["id"].startswith("automatic_"), "invalid automatic source key")
        binding, source_key = entry["preparation"], entry["id"]
        prep_path = resolved(root, binding["path"])
        stage = root / ".studio/materials"
        expected_name = source_key if not component else source_key + "_"
        if (prep_path.name != "preparation_receipt.json" or prep_path.parent.parent != stage
                or (prep_path.parent.name != expected_name if not component else not prep_path.parent.name.startswith(expected_name))):
            return None
    _require(resolved(root, queue["preparation"]) == resolved(root, binding["path"]), "queue preparation binding changed")
    workspace = _workspace(root, prepared)
    lineage = {}
    scope, pools, preparation, source, previews = (_evaluation(root, workspace, prepared, queue, lineage) if evaluation
                                                 else _train(root, workspace, prepared, queue, entry, component, lineage))
    selected_queue = {k: queue[k] for k in ("id", "kind", "queue", "pool_manifest", "preparation", "pools", "evaluation_only", "automatic_component") if k in queue}
    recipe = {"kind": prepared["kind"], "project": str(project), "workspace": str(workspace), "review_queues": [selected_queue]}
    if evaluation:
        recipe["evaluation_preparation"] = preparation
    else:
        recipe["preparations"] = [{"id": source_key, "preparation": preparation, "pools": queue["pools"]}]
        if component:
            recipe["material_acquisition_policy"] = prepared["material_acquisition_policy"]
    return {"kind": _KIND, "policy_id": POLICY_ID, "project": str(project), "wakeword": config["wakeword"],
            "scope": scope, "allowed_pools": pools, "source_key": source_key,
            "preparation": preparation, "source_manifest": source, "queue": previews,
            "catalog_pins": _catalog_pins(), "lineage_bindings": sorted(lineage.values(), key=lambda b: b["path"]),
            "human_listening": False, "human_listened_count": 0, "human_answers": None, "recipe": recipe}


def fixed_queue_evidence(project, prepared, queue):
    """Return immutable technical evidence for one actually pinned queue."""
    try:
        payload = _payload(project, prepared, queue)
    except (KeyError, TypeError, OSError) as error:
        raise ValueError("Fixed library material policy: incomplete or changed source evidence") from error
    if payload is None:
        return None
    root = Path(project).resolve().parent
    path = root / "review/library_fixed_sources" / (identity(payload) + ".json")
    if path.exists():
        _require(read_json(path) == payload, "immutable library evidence changed")
    else:
        write_json(path, payload)
    return {"path": path.relative_to(root).as_posix(), "sha256": digest(path)}


def validate_library_policy_evidence(root, artifact, role=None):
    """Validate compact admission metadata against current catalog and lineage."""
    root = Path(root).resolve()
    _require(artifact.get("type") == ARTIFACT_TYPE and artifact.get("policy_id") == POLICY_ID
             and artifact.get("content_review") == CONTENT_REVIEW and artifact.get("human_listening") is False
             and type(artifact.get("human_listened_count")) is int and artifact["human_listened_count"] == 0
             and artifact.get("human_answers") is None, "artifact is not a technical library-policy admission")
    _require(isinstance(artifact.get("library_evidence"), dict), "hash-bound library evidence is missing")
    evidence_path = bound_file(root, artifact["library_evidence"])
    _require(evidence_path.is_relative_to(root / "review/library_fixed_sources"), "evidence escaped this project's policy directory")
    evidence = read_json(evidence_path)
    _require(evidence.get("kind") == _KIND and evidence.get("policy_id") == POLICY_ID, "unsupported policy evidence")
    _require(role is None or role in _TRAIN | {"evaluation"}, "unsupported material role")
    scope = evidence.get("scope")
    _require(scope in ("train", "evaluation") and artifact.get("state") == ("accepted" if scope == "train" else "accepted_for_evaluation")
             and artifact.get("wakeword") == evidence.get("wakeword")
             and artifact.get("source_manifest") == evidence.get("source_manifest"), "admission scope/source binding changed")
    if role is not None:
        _require(role in evidence.get("allowed_pools", []) and (role == "evaluation") == (scope == "evaluation"), "material role is outside its admitted scope")
    if artifact.get("preparation") is not None:
        _require(artifact["preparation"] == evidence.get("preparation"), "admission preparation changed")
    recipe = evidence.get("recipe", {})
    queues = recipe.get("review_queues")
    _require(isinstance(queues, list) and len(queues) == 1, "evidence has no exact source queue")
    try:
        expected = _payload(root / "project.json", recipe, queues[0])
    except (KeyError, TypeError, OSError) as error:
        raise ValueError("Fixed library material policy: incomplete or changed source evidence") from error
    _require(expected is not None and evidence == expected, "policy evidence/catalog/preparation/source lineage changed")
    return evidence
