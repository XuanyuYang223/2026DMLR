"""Fail-closed installed-package data interfaces."""
from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path


def require_disposable_config(config):
    if not isinstance(config, Mapping):
        raise TypeError("An explicit data configuration is required")
    if config.get("historical_exclusion") is not False:
        raise ValueError(
            "The installed release requires historical_exclusion=false in an explicit "
            "disposable-data config. Historical object pools are not bundled in the wheel. "
            "Use the documented repository research workflow to certify historical exclusion; "
            "this interface does not silently relax that policy."
        )


def generate_dataset(out, config):
    """Generate explicitly scoped data; historical research isolation is rejected.

    The frozen low-level oracle/generator remains byte-identical. It is suitable
    for research when used in its complete repository workflow, which also
    supplies and verifies the historical pools. A standalone wheel lacks them.
    """
    require_disposable_config(config)
    if out is None:
        raise TypeError("An explicit output directory is required")
    from permworld_platform.data import generate_dataset as frozen_generate
    return frozen_generate(out, config)


def verify_dataset(out, *, recompute_labels=True):
    """Replay disposable data without claiming historical-pool coverage."""
    if out is None:
        raise TypeError("An explicit output directory is required")
    path = Path(out)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf8"))
    require_disposable_config(manifest["config"])
    from permworld_platform.data import verify_dataset as frozen_verify
    result = frozen_verify(path, recompute_labels=recompute_labels)
    return {
        **result, "historical_exclusion": False,
        "historical_orbit_overlap": None,
        "historical_overlap": "not checked; historical pools are not part of the wheel",
    }
