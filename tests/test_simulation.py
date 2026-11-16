from __future__ import annotations

import asyncio
import json

from digital_life.simulation import run_simulation


def test_two_day_simulation_meets_all_assertions(tmp_path):
    report = asyncio.run(
        run_simulation(days=2, seed=42, data_dir=tmp_path / "sim")
    )
    report.assert_all()
    assert report.metrics["messages"] > 0
    assert report.metrics["proactive_count"] > 0
    assert report.metrics["max_ctx_tokens"] <= 8000
    assert report.metrics["safety_gate_hits"] == 0
    assert report.metrics["p_std"] > 0.1
    assert (tmp_path / "sim" / "sim_report.json").exists()
    data = json.loads(
        (tmp_path / "sim" / "sim_report.json").read_text(encoding="utf-8")
    )
    assert data["assertions"]
