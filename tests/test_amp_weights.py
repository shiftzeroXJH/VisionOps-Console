from pathlib import Path

import pytest

from backend.core.model_catalog import configure_amp_weights


@pytest.mark.parametrize("source", ["run", "models", "missing", "truncated"])
def test_amp_uses_existing_weight_without_changing_persistent_settings(tmp_path, monkeypatch, source):
    import ultralytics.utils as utils

    run_dir, models_dir = tmp_path / "run", tmp_path / "models"
    run_dir.mkdir()
    models_dir.mkdir()
    original = tmp_path / "default_weights"
    monkeypatch.setattr(utils, "WEIGHTS_DIR", original)
    settings_before = dict(utils.SETTINGS)
    if source in {"run", "models"}:
        target = run_dir if source == "run" else models_dir
        (target / "yolo26n.pt").write_bytes(b"x" * 2048)
    elif source == "truncated":
        (run_dir / "yolo26n.pt").write_bytes(b"partial")
    configure_amp_weights(run_dir, models_dir)
    # Mirrors the import performed inside Ultralytics check_amp.
    from ultralytics.utils import WEIGHTS_DIR
    assert WEIGHTS_DIR == (target if source in {"run", "models"} else original)
    assert dict(utils.SETTINGS) == settings_before
    assert not original.exists()
