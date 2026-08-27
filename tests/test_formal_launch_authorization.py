"""Shared-trainer authorization for upstream-selected formal recipes."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from scripts import train
from scripts.train import validate_formal_recipe_authorization


EXP22_OBJECTIVE = "treewm_v2_grounded_gauge_formal_v1"
REPAIR_OBJECTIVE = "treewm_v2_grounded_repair_formal_v1"
PREREQUISITE = "a" * 64
RECIPE = "b" * 64
PROTOCOL = "c" * 64
RUN_NAME = "grounded-gauge-formal-scene-seed220"
CANONICAL_ARGV = [
    "objective_version=treewm_v2_grounded_gauge_formal_v1",
    "+campaign_prerequisite_binding_sha256=" + PREREQUISITE,
    "+campaign_selected_recipe_sha256=" + RECIPE,
]


def authorized_config() -> dict[str, str]:
    return {
        "campaign_prerequisite_binding_sha256": PREREQUISITE,
        "campaign_selected_recipe_sha256": RECIPE,
    }


def authorized_environment() -> dict[str, str]:
    return {
        "TREEWM_PREREQUISITE_BINDING_SHA256": PREREQUISITE,
        "TREEWM_SELECTED_RECIPE_SHA256": RECIPE,
        "TREEWM_PROTOCOL_SHA256": PROTOCOL,
        "TREEWM_RUN_NAME": RUN_NAME,
    }


class CanonicalExp22Authority:
    def load_manifest(self, _path):
        return {"campaign_id": "treewm-grounded-gauge-formal-v1"}

    def verify_protocol_lock(self, _package):
        return PROTOCOL

    def load_prerequisite_bindings(
        self, _manifest, _path, *, verify_external_files: bool
    ):
        assert verify_external_files is False
        return {
            "binding_sha256": PREREQUISITE,
            "selected_recipe_sha256": RECIPE,
        }

    def expand_runs(self, _manifest):
        return [SimpleNamespace(run_name=RUN_NAME)]

    def trainer_command(self, _manifest, _run, *, repo_root):
        assert repo_root == train.Path(train.__file__).resolve().parents[1]
        return {
            "argv": ["/pinned/python", "/fixed/train_entry.py", *CANONICAL_ARGV],
            "environment": authorized_environment(),
            "hashes": {
                "package_protocol_sha256": PROTOCOL,
                "prerequisite_binding_sha256": PREREQUISITE,
                "selected_recipe_sha256": RECIPE,
            },
        }


@pytest.fixture
def canonical_authority(monkeypatch):
    authority = CanonicalExp22Authority()
    monkeypatch.setattr(train, "_load_exp22_campaign_authority", lambda _path: authority)
    return authority


def test_exp22_bare_direct_invocation_is_rejected() -> None:
    with pytest.raises(ValueError, match="Exp22 gauge formal objective requires sealed"):
        validate_formal_recipe_authorization(EXP22_OBJECTIVE, {}, {}, argv=[])


def test_exp22_authorization_is_wired_before_dataset_or_model_construction() -> None:
    source = inspect.getsource(train.main.__wrapped__)
    authorization = source.index("validate_formal_recipe_authorization(objective_version, cfg)")
    dataset = source.index("build_datasets(")
    model = source.index("build_model(")
    assert authorization < dataset < model


@pytest.mark.parametrize(
    ("config_key", "environment_key"),
    [
        (
            "campaign_prerequisite_binding_sha256",
            "TREEWM_PREREQUISITE_BINDING_SHA256",
        ),
        ("campaign_selected_recipe_sha256", "TREEWM_SELECTED_RECIPE_SHA256"),
    ],
)
def test_exp22_rejects_each_config_environment_digest_mismatch(
    canonical_authority,
    config_key: str,
    environment_key: str,
) -> None:
    config = authorized_config()
    environment = authorized_environment()
    environment[environment_key] = "d" * 64
    assert config[config_key] != environment[environment_key]
    with pytest.raises(ValueError, match="Exp22 gauge formal prerequisite hashes differ"):
        validate_formal_recipe_authorization(
            EXP22_OBJECTIVE, config, environment, argv=CANONICAL_ARGV
        )


def test_exp22_rejects_equal_well_formed_but_noncanonical_hashes(
    canonical_authority,
) -> None:
    fabricated_config = {
        "campaign_prerequisite_binding_sha256": "d" * 64,
        "campaign_selected_recipe_sha256": "e" * 64,
    }
    fabricated_environment = authorized_environment()
    fabricated_environment["TREEWM_PREREQUISITE_BINDING_SHA256"] = "d" * 64
    fabricated_environment["TREEWM_SELECTED_RECIPE_SHA256"] = "e" * 64
    with pytest.raises(ValueError, match="canonical Exp22 launch"):
        validate_formal_recipe_authorization(
            EXP22_OBJECTIVE,
            fabricated_config,
            fabricated_environment,
            argv=CANONICAL_ARGV,
        )


def test_exp22_canonical_package_launch_is_accepted(canonical_authority) -> None:
    validate_formal_recipe_authorization(
        EXP22_OBJECTIVE,
        authorized_config(),
        authorized_environment(),
        argv=CANONICAL_ARGV,
    )


def test_exp22_rejects_unsealed_package_even_with_matching_copies(monkeypatch) -> None:
    authority = CanonicalExp22Authority()

    def reject_unsealed(*_args, **_kwargs):
        raise RuntimeError("formal submission is not sealed")

    authority.load_prerequisite_bindings = reject_unsealed  # type: ignore[method-assign]
    monkeypatch.setattr(train, "_load_exp22_campaign_authority", lambda _path: authority)
    with pytest.raises(ValueError, match="canonical Exp22 launch authorization failed"):
        validate_formal_recipe_authorization(
            EXP22_OBJECTIVE,
            authorized_config(),
            authorized_environment(),
            argv=CANONICAL_ARGV,
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("argv", "arguments differ"),
        ("run-name", "does not identify one canonical"),
        ("protocol-environment", "environment differs"),
    ],
)
def test_exp22_rejects_noncanonical_invocation_fields(
    canonical_authority,
    mutation: str,
    match: str,
) -> None:
    environment = authorized_environment()
    argv = list(CANONICAL_ARGV)
    if mutation == "argv":
        argv[-1] = "+campaign_selected_recipe_sha256=" + "f" * 64
    elif mutation == "run-name":
        environment["TREEWM_RUN_NAME"] = "fabricated-run"
    else:
        environment["TREEWM_PROTOCOL_SHA256"] = "f" * 64
    with pytest.raises(ValueError, match=match):
        validate_formal_recipe_authorization(
            EXP22_OBJECTIVE, authorized_config(), environment, argv=argv
        )


def test_repair_formal_keeps_its_distinct_existing_handshake() -> None:
    validate_formal_recipe_authorization(
        REPAIR_OBJECTIVE,
        authorized_config(),
        authorized_environment(),
    )
    with pytest.raises(ValueError, match="repaired formal objective requires sealed"):
        validate_formal_recipe_authorization(REPAIR_OBJECTIVE, {}, {})


@pytest.mark.parametrize(
    "objective",
    [
        "treewm_v1",
        "treewm_v2_grounded_formal_v1",
        "treewm_v2_grounded_gauge_pilot_v2",
    ],
)
def test_other_objectives_do_not_acquire_selected_recipe_authorization(
    objective: str,
) -> None:
    validate_formal_recipe_authorization(objective, {}, {})
