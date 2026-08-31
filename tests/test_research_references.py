"""Integrity checks for the shipped, non-prescriptive research references."""

import json
import math
import re
import shutil
import subprocess
import sys
import zipfile
from importlib.resources import files
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIRECTORY = ROOT / "amt_augmentor" / "research_references"
GALDR_CARD_PATH = REFERENCE_DIRECTORY / "galdr-hardanger-v1.json"
MAESTRO_CARD_PATH = REFERENCE_DIRECTORY / "maestro-transkun-v1.json"
SCHEMA_PATH = REFERENCE_DIRECTORY / "schema-v1.json"
GALDR_METRICS = {"tune_macro_onset_f1", "tune_macro_onset_offset_f1"}
MAESTRO_METRICS = {
    "recording_macro_onset_f1",
    "recording_macro_onset_offset_f1",
}
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _observations(card):
    return {
        observation["id"]: observation
        for result_set in card["result_sets"]
        for observation in result_set["observations"]
    }


def _evaluate_formula(formula, observations, metrics):
    """Evaluate the card's deliberately small ``observation +/- ...`` grammar."""
    tokens = formula.split()
    assert len(tokens) % 2 == 1
    assert all(tokens[index] in {"+", "-"} for index in range(1, len(tokens), 2))
    assert all(tokens[index] in observations for index in range(0, len(tokens), 2))

    totals = dict(observations[tokens[0]]["metrics"])
    for index in range(1, len(tokens), 2):
        sign = 1.0 if tokens[index] == "+" else -1.0
        values = observations[tokens[index + 1]]["metrics"]
        for metric in metrics:
            totals[metric] += sign * values[metric]
    return totals


def test_galdr_card_is_complete_consistent_and_non_prescriptive():
    schema = _load(SCHEMA_PATH)
    card = _load(GALDR_CARD_PATH)

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert (
        schema["properties"]["package_policy"]["properties"][
            "changes_runtime_defaults"
        ]["const"]
        is False
    )
    assert card["$schema"] == "./schema-v1.json"
    assert card["schema_version"] == 1
    assert card["kind"] == "amt_augmentor_research_reference"
    assert card["status"] == "complete"
    assert card["package_policy"]["changes_runtime_defaults"] is False
    assert card["package_policy"]["recipe_status"] == "dataset_specific_evidence"
    assert card["scope"]["evidence_level"] == "single_seed_validation"
    assert card["scope"]["random_seeds"] == [424242]
    assert card["scope"]["validation_units"]["count"] == 6
    assert card["scope"]["test_data_accessed"] is False
    assert "30 prespecified candidate epochs" in card["scope"]["checkpoint_selection"]

    artifact_bases = {base["id"]: base for base in card["artifact_bases"]}
    assert set(artifact_bases) == {
        "galdr-protocol-source",
        "galdr-runtime-artifacts",
    }
    assert artifact_bases["galdr-protocol-source"]["revision"] == (
        "fa8028e13b90145c569624a1162ccfd3e46186a1"
    )
    assert artifact_bases["galdr-protocol-source"]["availability"] == "unpublished"
    assert artifact_bases["galdr-runtime-artifacts"]["availability"] == "unpublished"
    assert any(
        software["name"] == "AMT-Augmentor" and software["revision"] is None
        for software in card["software_provenance"]
    )

    metric_ids = [definition["id"] for definition in card["metric_definitions"]]
    assert len(metric_ids) == len(set(metric_ids))
    assert set(metric_ids) == GALDR_METRICS

    protocols = {protocol["id"]: protocol for protocol in card["protocols"]}
    assert len(protocols) == len(card["protocols"]) == 6
    for protocol in protocols.values():
        arm_ids = [arm["id"] for arm in protocol["arms"]]
        assert len(arm_ids) == len(set(arm_ids))
        for artifact in protocol["source_artifacts"]:
            assert not Path(artifact["locator"]).is_absolute()
            assert SHA256.fullmatch(artifact["sha256"])
            assert artifact["included"] is False
            matches = [
                base
                for base in artifact_bases.values()
                if any(
                    artifact["locator"].startswith(prefix)
                    for prefix in base["locator_prefixes"]
                )
            ]
            assert len(matches) == 1

    result_set_ids = [result_set["id"] for result_set in card["result_sets"]]
    assert len(result_set_ids) == len(set(result_set_ids))
    for result_set in card["result_sets"]:
        protocol = protocols[result_set["protocol_id"]]
        valid_arms = {arm["id"] for arm in protocol["arms"]}
        for observation in result_set["observations"]:
            assert observation["arm_id"] in valid_arms
            assert set(observation["metrics"]) == GALDR_METRICS
            assert all(0.0 <= value <= 1.0 for value in observation["metrics"].values())
            artifact = observation["artifact"]
            assert not Path(artifact["locator"]).is_absolute()
            assert SHA256.fullmatch(artifact["sha256"])
            assert artifact["included"] is False
            matches = [
                base
                for base in artifact_bases.values()
                if any(
                    artifact["locator"].startswith(prefix)
                    for prefix in base["locator_prefixes"]
                )
            ]
            assert len(matches) == 1

    observations = _observations(card)
    assert len(observations) == 38
    derived_ids = [result["id"] for result in card["derived_results"]]
    assert len(derived_ids) == len(set(derived_ids))
    for result in card["derived_results"]:
        assert result["protocol_id"] in protocols
        assert set(result["observation_ids"]) <= observations.keys()
        assert set(result["values"]) == GALDR_METRICS
        calculated = _evaluate_formula(result["formula"], observations, GALDR_METRICS)
        for metric in GALDR_METRICS:
            assert math.isclose(
                result["values"][metric],
                calculated[metric],
                rel_tol=0.0,
                abs_tol=1e-15,
            )

    assert observations["o-e150"]["metrics"] == {
        "tune_macro_onset_f1": 0.9060370717264375,
        "tune_macro_onset_offset_f1": 0.8467086488723599,
    }
    assert observations["archival-noise-e150"]["metrics"] == {
        "tune_macro_onset_f1": 0.9280128756942408,
        "tune_macro_onset_offset_f1": 0.8749630986874856,
    }
    assert observations["t-global-e150"]["metrics"] == {
        "tune_macro_onset_f1": 0.9273503321191682,
        "tune_macro_onset_offset_f1": 0.8719101518074468,
    }

    preference = protocols["preference-screen-e100-e150"]
    exposure = {arm["id"]: arm["logical_exposure"] for arm in preference["arms"]}
    for arm_id in ("p1", "p4", "p4-wide", "p12"):
        assert exposure[arm_id] == {
            "original": 48,
            "archival_noise_v1": 12,
            "pitch_shift_v1": 12,
        }
    assert exposure["rf"] == {
        "original": 36,
        "archival_noise_v1": 12,
        "pitch_shift_v1": 12,
        "reverb_filters_v1": 12,
    }
    assert exposure["t-global"] == {
        "original": 36,
        "archival_noise_v1": 12,
        "pitch_shift_v1": 12,
        "time_stretch_v1": 12,
    }

    oc_evaluation = protocols["compact-original-vs-conventional-e150"]["evaluation"]
    assert oc_evaluation["candidate_epochs"] == list(range(5, 151, 5))
    assert "checkpoint_search" not in oc_evaluation
    assert oc_evaluation["reported_epoch"] == 150


def test_maestro_card_is_complete_consistent_and_non_prescriptive():
    card = _load(MAESTRO_CARD_PATH)

    assert card["$schema"] == "./schema-v1.json"
    assert card["schema_version"] == 1
    assert card["kind"] == "amt_augmentor_research_reference"
    assert card["reference_id"] == "maestro-transkun-v1"
    assert card["status"] == "complete"
    assert card["package_policy"]["changes_runtime_defaults"] is False
    assert card["package_policy"]["recipe_status"] == "no_recipe"
    assert len(card["package_policy"]["guidance"]) == 3
    assert card["scope"]["evidence_level"] == "single_seed_validation"
    assert card["scope"]["training_source_count"] == 962
    assert card["scope"]["validation_units"] == {
        "kind": "recordings",
        "count": 137,
    }
    assert card["scope"]["random_seeds"] == [424242]
    assert card["scope"]["test_data_accessed"] is False
    assert "untouched" in card["scope"]["checkpoint_selection"]
    assert "50000-update" in card["scope"]["checkpoint_selection"]

    revisions = {
        software["name"]: software["revision"]
        for software in card["software_provenance"]
    }
    assert revisions["AMT-Augmentor"] == ("4dc2028d4c127f0e3bd4d22b389ade4703631a9e")
    assert revisions["Transkun"] == ("c219479911f641db6710a04c6d6f6dfab6bb1404")
    assert revisions["Transkun bundle campaign control"] == (
        "d3197aafacbc176ddf1cb854ccb46648b019bb25"
    )

    metric_ids = {definition["id"] for definition in card["metric_definitions"]}
    assert metric_ids == MAESTRO_METRICS

    assert len(card["artifact_bases"]) == 1
    artifact_base = card["artifact_bases"][0]
    assert artifact_base["availability"] == "unpublished"
    assert artifact_base["revision"] is None

    protocols = {protocol["id"]: protocol for protocol in card["protocols"]}
    assert len(protocols) == len(card["protocols"]) == 2
    protocol = protocols["maestro-transkun-50k-finetune"]
    assert protocol["id"] == "maestro-transkun-50k-finetune"
    assert protocol["training"]["fine_tuning_updates"] == 50000
    assert protocol["training"]["paper_training_budget_updates"] == 500000
    assert protocol["training"]["fresh_optimizer_per_arm"] is True
    assert protocol["training"]["pedal_extension_applied"] is True
    assert protocol["evaluation"]["checkpoint_search"] is False
    assert protocol["evaluation"]["recording_count"] == 137
    assert protocol["evaluation"]["aggregation"] == "macro_by_recording"
    assert protocol["evaluation"]["segment_size_seconds"] == 16.0
    assert protocol["evaluation"]["segment_hop_seconds"] == 8.0
    assert protocol["evaluation"]["test_media_or_labels_opened"] is False

    arms = {arm["id"]: arm for arm in protocol["arms"]}
    assert set(arms) == {
        "paper-baseline",
        "clean",
        "builtin",
        "noise",
        "reverb",
        "time",
    }
    assert "logical_exposure" not in arms["paper-baseline"]
    assert arms["clean"]["logical_exposure"] == {"original_metadata_fraction": 1.0}
    assert arms["builtin"]["logical_exposure"] == {"original_metadata_fraction": 1.0}
    for arm_id, family in (
        ("noise", "noise_snr_v1"),
        ("reverb", "reverb_only_v1"),
        ("time", "time_stretch_v1"),
    ):
        assert arms[arm_id]["logical_exposure"] == {
            "original_metadata_fraction": 0.5,
            f"{family}_metadata_fraction": 0.5,
        }

    for checked_protocol in protocols.values():
        arm_ids = [arm["id"] for arm in checked_protocol["arms"]]
        assert len(arm_ids) == len(set(arm_ids))
        for artifact in checked_protocol["source_artifacts"]:
            assert not Path(artifact["locator"]).is_absolute()
            assert SHA256.fullmatch(artifact["sha256"])
            assert artifact["included"] is False
            matches = [
                prefix
                for prefix in artifact_base["locator_prefixes"]
                if artifact["locator"].startswith(prefix)
            ]
            assert len(matches) == 1

    source_hashes = {
        artifact["role"]: artifact["sha256"]
        for artifact in protocol["source_artifacts"]
    }
    assert source_hashes["official model configuration"] == (
        "b5f55fdd6fcca19871510bc68151eb1d09ea5e4db59aeeb7cdc59f5977ff950c"
    )
    assert source_hashes["official paper checkpoint"] == (
        "76469be487911e110e0e661fadb583645e89c35364faa1ef8897d5600ba6ce20"
    )
    assert source_hashes["noise materialization completion receipt"] == (
        "c581283b365a2b19674a2fc1429b748702f6ab2e7500f9f263af977f4142bb59"
    )
    assert source_hashes["reverb materialization completion receipt"] == (
        "764f63580b40b9bbc15569913daa130be6f8f64481c33bcbc6080fe66b28b8b0"
    )
    assert source_hashes["time-stretch materialization completion receipt"] == (
        "c64602496f131cce035362d839096e5159233f3af5c69d2a914c33b6c231eb4c"
    )
    expected_checkpoint_hashes = {
        "clean": "fef22f4722e46c771602164030fcf6f72534cd5dc2c5f10252700cebfc8fa92a",
        "builtin": "5ed5864ad61cb2dedae40ae9fe25fe805bd5d688a60c1fcd1209ae8f96f13285",
        "noise": "585f954103df0ada111a57dad8f292e5ffdcbdbbb2df98f780d9d6c721c9e182",
        "reverb": "bbf45f2c8977ad4bd491b3fbfeb23ec55a72000996886daf81059aaac8eaf71a",
        "time": "a886a7177c536abafc888b7039d177720e7a125380b42e93ca6ead17ffdcaaee",
    }
    for arm_id, checkpoint_hash in expected_checkpoint_hashes.items():
        assert source_hashes[f"{arm_id} final checkpoint"] == checkpoint_hash

    expanded = protocols["maestro-transkun-expanded-bundles-50k"]
    assert expanded["training"]["fine_tuning_updates"] == 50000
    assert expanded["training"]["fresh_optimizer_per_arm"] is True
    assert expanded["training"]["pedal_extension_applied"] is True
    assert expanded["training"]["physical_dataset_expansion"] is True
    assert expanded["training"]["new_dsp_performed"] is False
    assert expanded["training"]["builtin_augmentation"] is False
    assert expanded["training"]["fixed_compute_single_seed"] is True
    assert expanded["training"]["rt_selected_post_hoc"] is True
    assert (
        expanded["training"]["diversity_and_original_fraction_jointly_changed"] is True
    )
    assert expanded["training"]["metadata_entries_by_arm"] == {
        "rt-expanded": 2886,
        "nrt-expanded": 3848,
    }
    assert expanded["training"]["logical_views_per_source_by_arm"] == {
        "rt-expanded": 3,
        "nrt-expanded": 4,
    }
    assert expanded["training"]["scheduled_component_draw_total_per_arm"] == 600000
    assert expanded["training"]["scheduled_component_draws_by_arm"] == {
        "rt-expanded": {
            "original": 199332,
            "reverb": 199483,
            "time": 201185,
        },
        "nrt-expanded": {
            "original": 149632,
            "noise": 149766,
            "reverb": 149641,
            "time": 150961,
        },
    }
    assert expanded["training"]["scheduled_component_draw_fractions_by_arm"] == {
        "rt-expanded": {
            "original": 0.33222,
            "reverb": 0.33247166666666667,
            "time": 0.3353083333333333,
        },
        "nrt-expanded": {
            "original": 0.24938666666666667,
            "noise": 0.24961,
            "reverb": 0.24940166666666666,
            "time": 0.25160166666666667,
        },
    }
    expanded_arms = {arm["id"]: arm for arm in expanded["arms"]}
    assert set(expanded_arms) == {"rt-expanded", "nrt-expanded"}
    assert expanded_arms["rt-expanded"]["logical_exposure"] == {
        "original_metadata_fraction": 0.3333333333333333,
        "reverb_only_v1_metadata_fraction": 0.3333333333333333,
        "time_stretch_v1_metadata_fraction": 0.3333333333333333,
    }
    assert expanded_arms["nrt-expanded"]["logical_exposure"] == {
        "original_metadata_fraction": 0.25,
        "noise_snr_v1_metadata_fraction": 0.25,
        "reverb_only_v1_metadata_fraction": 0.25,
        "time_stretch_v1_metadata_fraction": 0.25,
    }
    assert expanded["evaluation"]["checkpoint_search"] is False
    assert expanded["evaluation"]["recording_count"] == 137
    assert expanded["evaluation"]["test_media_or_labels_opened"] is False
    assert expanded["evaluation"]["prediction_count_by_arm"] == {
        "rt-expanded": 137,
        "nrt-expanded": 137,
    }
    assert expanded["evaluation"]["all_prediction_hashes_verified"] is True
    assert expanded["evaluation"]["recordings_sha256_by_arm"] == {
        "rt-expanded": "46604d026b648a987609b91f93f31e394ab8dba6bced9e35beb9174697fb17c1",
        "nrt-expanded": "6b1074be0f48a51f5df0a56100f8d764c79da4cfee94b35d133835e63b6a5d3f",
    }

    expanded_source_hashes = {
        artifact["role"]: artifact["sha256"]
        for artifact in expanded["source_artifacts"]
    }
    assert len(expanded_source_hashes) == len(expanded["source_artifacts"]) == 14
    assert expanded_source_hashes == {
        "expanded-bundle training launch protocol": "b061da18a232ab6f1bb14fcf68223bd0072e0b42fcbe17fad6bc9bcf16d6bf7e",
        "expanded-bundle training release receipt": "15b999b029798ab8873fd0d40ecdbb994462b5c06edd6ea5cecf57a54d432677",
        "expanded-bundle terminal verification receipt": "2c3e784440af84307cbc1aa996200d8623a955d7e8fa80c1ed61239254a6cfad",
        "expanded-bundle loader memory receipt": "dcc0836abfb845c6b4dc81e931418fe94fd6590a24d7609adb506bbb7604ba57",
        "RT expanded metadata manifest": "84939309fc9cd3699c024a0ebdcc7a0b2993b6a35cdaa29d472c708abedf4120",
        "RT expanded metadata payload": "13a6bf923a0bd1886fd273935af3449d34882de90b784cac009fdb38f547b3f2",
        "NRT expanded metadata manifest": "dd7bebd20aa8f368c19d1061a4dbfe857b1061fe2f7df6733ff128243237a2e9",
        "NRT expanded metadata payload": "c79afa6a7b5dfa25356c4a48cac3cd68e9cc3534a60d98df9593a07ca565741c",
        "expanded-bundle validation metadata manifest": "71223bb536d37ad6b5e8d29d55cb258df75e947683847c212b9e0764fc95be07",
        "expanded-bundle validation metadata payload": "e5a548b6ab2102014ebaf7849a2c909f756aa3dd4607f97dee847a57d068cb9b",
        "RT expanded training result": "0d51baca4f326be68911a847270b19a83696853c39b56138905d9c1f843e3825",
        "RT expanded final checkpoint": "c87ed65d23cb78c5ac7136f36f0eb2332b67e353362bc29bc4efb4734d2fb75b",
        "NRT expanded training result": "83e818f8fc48673663069a68d731fa3f620323e0d4d68fd886c4645732e447ee",
        "NRT expanded final checkpoint": "a3ab2d187b4ad0a15b4a9af7e79a37173b74d04102eb41493ad7f5b4b19269e5",
    }

    result_set_ids = [result_set["id"] for result_set in card["result_sets"]]
    assert len(result_set_ids) == len(set(result_set_ids)) == 2
    for result_set in card["result_sets"]:
        valid_arms = {arm["id"] for arm in protocols[result_set["protocol_id"]]["arms"]}
        for observation in result_set["observations"]:
            assert observation["arm_id"] in valid_arms
            assert set(observation["metrics"]) == MAESTRO_METRICS
            assert all(0.0 <= value <= 1.0 for value in observation["metrics"].values())
            artifact = observation["artifact"]
            assert not Path(artifact["locator"]).is_absolute()
            assert SHA256.fullmatch(artifact["sha256"])
            assert artifact["included"] is False
            assert (
                sum(
                    artifact["locator"].startswith(prefix)
                    for prefix in artifact_base["locator_prefixes"]
                )
                == 1
            )

    observations = _observations(card)
    expected_metrics = {
        "paper-baseline": (0.9816826530904008, 0.932441981955764),
        "clean": (0.9816614478147303, 0.9322854252174543),
        "builtin": (0.9792357943657011, 0.923509606997237),
        "noise": (0.981678700714919, 0.9320669956940891),
        "reverb": (0.9817032120322566, 0.9322743458926179),
        "time": (0.9815850579806292, 0.932239119077584),
        "rt-expanded": (0.9816605804254718, 0.9322877381110631),
        "nrt-expanded": (0.9816149843649851, 0.9320641948953048),
    }
    expected_result_hashes = {
        "paper-baseline": "2d0dc30b1ab1070759266d49dd3360d93f7a67c5be60f56bc7e33c931f71a4c9",
        "clean": "637ba4cc457739dd3899bae622827e9c856d22e1b6b1f6078652e4cf1cb13a4d",
        "builtin": "7b8014c745c1f4b1306dbd299c3a2a34c30328ba2337768e69e8658d55438fe1",
        "noise": "ead1c0429862c8a119d4d4b438a18ce83030289d45d33e2427b00a922bef37f7",
        "reverb": "5b780e55eefacec31f06601e2961f7c457dc9748371ffcf1d892cc83094cfc58",
        "time": "d847ca1221b18fa02237ccba2817828901df1c0220bd4dbbeb640099ff70ee27",
        "rt-expanded": "7bf1d1e6e99261a6876814e32799754a9f41369138a3426430d239e57a814b2f",
        "nrt-expanded": "d254c872025159609e626bd6ed9832ac2f20bea42309ae7b1427dc53e1d9aa06",
    }
    assert set(observations) == set(expected_metrics)
    for arm_id, (onset, onset_offset) in expected_metrics.items():
        assert observations[arm_id]["metrics"] == {
            "recording_macro_onset_f1": onset,
            "recording_macro_onset_offset_f1": onset_offset,
        }
        assert (
            observations[arm_id]["artifact"]["sha256"]
            == (expected_result_hashes[arm_id])
        )
        assert observations[arm_id]["artifact"]["included"] is False
    assert observations["paper-baseline"]["checkpoint"] == {
        "label": "untouched-official-paper-checkpoint"
    }
    for arm_id in ("clean", "builtin", "noise", "reverb", "time"):
        assert observations[arm_id]["checkpoint"]["optimizer_step"] == 50000
        assert observations[arm_id]["checkpoint"]["label"] == "fixed-final"
    for arm_id in ("rt-expanded", "nrt-expanded"):
        assert observations[arm_id]["checkpoint"] == {
            "epoch": 2,
            "optimizer_step": 50000,
            "label": "fixed-final",
        }

    assert len(card["derived_results"]) == 15
    for result in card["derived_results"]:
        assert result["protocol_id"] in protocols
        assert set(result["observation_ids"]) <= observations.keys()
        assert set(result["values"]) == MAESTRO_METRICS
        calculated = _evaluate_formula(result["formula"], observations, MAESTRO_METRICS)
        for metric in MAESTRO_METRICS:
            assert math.isclose(
                result["values"][metric],
                calculated[metric],
                rel_tol=0.0,
                abs_tol=1e-15,
            )

    derived = {result["id"]: result["values"] for result in card["derived_results"]}
    assert derived["builtin-minus-clean"] == {
        "recording_macro_onset_f1": -0.0024256534490292303,
        "recording_macro_onset_offset_f1": -0.008775818220217224,
    }
    assert derived["reverb-minus-clean"] == {
        "recording_macro_onset_f1": 0.00004176421752632997,
        "recording_macro_onset_offset_f1": -0.000011079324836349436,
    }
    assert derived["rt-expanded-minus-clean"] == {
        "recording_macro_onset_f1": -8.673892585298049e-07,
        "recording_macro_onset_offset_f1": 2.312893608857891e-06,
    }
    assert derived["nrt-expanded-minus-clean"] == {
        "recording_macro_onset_f1": -4.646344974523764e-05,
        "recording_macro_onset_offset_f1": -0.00022123032214949045,
    }
    assert derived["rt-expanded-minus-builtin"] == {
        "recording_macro_onset_f1": 0.0024247860597707005,
        "recording_macro_onset_offset_f1": 0.008778131113826082,
    }
    assert derived["nrt-expanded-minus-builtin"] == {
        "recording_macro_onset_f1": 0.0023791899992839927,
        "recording_macro_onset_offset_f1": 0.008554587898067734,
    }
    assert derived["nrt-expanded-minus-rt-expanded"] == {
        "recording_macro_onset_f1": -4.5596060486707835e-05,
        "recording_macro_onset_offset_f1": -0.00022354321575834835,
    }
    nrt_rt = next(
        result
        for result in card["derived_results"]
        if result["id"] == "nrt-expanded-minus-rt-expanded"
    )
    assert "not an isolated noise effect" in nrt_rt["interpretation"]
    assert "All eight primary rows" in card["provenance"]["verification"]
    assert "terminal verification receipt" in card["provenance"]["verification"]


def test_reference_index_and_distribution_manifest_are_in_sync():
    index = (ROOT / "docs" / "RESEARCH_REFERENCES.md").read_text(encoding="utf-8")
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    assert "galdr-hardanger-v1.json" in index
    assert "maestro-transkun-v1.json" in index
    assert "schema-v1.json" in index
    assert "MAESTRO" in index and "Forthcoming" not in index
    assert MAESTRO_CARD_PATH.exists()
    assert "include docs/RESEARCH_REFERENCES.md" in manifest
    assert "include docs/MAESTRO_TRANSKUN_CASE_STUDY.md" in manifest
    assert "recursive-include amt_augmentor/research_references *.json" in manifest


def test_reference_cards_are_accessible_as_package_resources():
    resources = files("amt_augmentor").joinpath("research_references")
    packaged_card = json.loads(
        resources.joinpath("galdr-hardanger-v1.json").read_text(encoding="utf-8")
    )
    packaged_schema = json.loads(
        resources.joinpath("schema-v1.json").read_text(encoding="utf-8")
    )
    packaged_maestro = json.loads(
        resources.joinpath("maestro-transkun-v1.json").read_text(encoding="utf-8")
    )

    assert packaged_card["reference_id"] == "galdr-hardanger-v1"
    assert packaged_maestro["reference_id"] == "maestro-transkun-v1"
    assert (
        packaged_schema["title"] == "AMT-Augmentor research reference card, version 1"
    )


def test_pep517_wheel_contains_reference_cards(tmp_path):
    source = tmp_path / "source"
    wheel_directory = tmp_path / "wheel"
    shutil.copytree(
        ROOT,
        source,
        ignore=shutil.ignore_patterns(
            ".git", ".pytest_cache", "__pycache__", "build", "dist", "*.egg-info"
        ),
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--wheel-dir",
            str(wheel_directory),
            str(source),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    wheel = next(wheel_directory.glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    assert "amt_augmentor/research_references/schema-v1.json" in names
    assert "amt_augmentor/research_references/galdr-hardanger-v1.json" in names
    assert "amt_augmentor/research_references/maestro-transkun-v1.json" in names
