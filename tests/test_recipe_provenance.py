from __future__ import annotations

from treewm.data.ogbench_dataset import recipe_producer_identity_from_env


def test_recipe_producer_identity_defaults_to_trainer(monkeypatch) -> None:
    monkeypatch.setenv("TREEWM_CODE_SHA256", "a" * 64)
    monkeypatch.setenv("TREEWM_RUNTIME_SHA256", "b" * 64)
    monkeypatch.delenv("TREEWM_RECIPE_CODE_SHA256", raising=False)
    monkeypatch.delenv("TREEWM_RECIPE_RUNTIME_SHA256", raising=False)

    assert recipe_producer_identity_from_env() == ("a" * 64, "b" * 64)


def test_recipe_producer_identity_can_be_pinned_independently(monkeypatch) -> None:
    monkeypatch.setenv("TREEWM_CODE_SHA256", "a" * 64)
    monkeypatch.setenv("TREEWM_RUNTIME_SHA256", "b" * 64)
    monkeypatch.setenv("TREEWM_RECIPE_CODE_SHA256", "c" * 64)
    monkeypatch.setenv("TREEWM_RECIPE_RUNTIME_SHA256", "d" * 64)

    assert recipe_producer_identity_from_env() == ("c" * 64, "d" * 64)
