"""T1.1 DoD: save_figure produces both .pdf and .png files and appends a manifest entry
containing the figure name and the list of run_ids used."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

import qapinn.viz.style as style_mod


@pytest.fixture(autouse=True)
def _isolated_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(style_mod, "FIGURES_DIR", tmp_path / "figures")
    monkeypatch.setattr(style_mod, "MANIFEST_PATH", tmp_path / "manifest.json")


def test_save_figure_writes_both_files_and_manifest_entry():
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])

    paths = style_mod.save_figure(fig, "test_fig", run_ids=["abc123", "def456"])
    plt.close(fig)

    assert paths["pdf"].is_file()
    assert paths["png"].is_file()

    with style_mod.MANIFEST_PATH.open(encoding="utf-8") as f:
        manifest = json.load(f)

    assert "test_fig" in manifest
    assert manifest["test_fig"]["run_ids"] == ["abc123", "def456"]


def test_save_figure_appends_without_overwriting_previous_entries():
    fig1, _ax1 = plt.subplots()
    style_mod.save_figure(fig1, "fig_one", run_ids=["r1"])
    plt.close(fig1)

    fig2, _ax2 = plt.subplots()
    style_mod.save_figure(fig2, "fig_two", run_ids=["r2"])
    plt.close(fig2)

    with style_mod.MANIFEST_PATH.open(encoding="utf-8") as f:
        manifest = json.load(f)

    assert set(manifest.keys()) == {"fig_one", "fig_two"}
    assert manifest["fig_one"]["run_ids"] == ["r1"]
    assert manifest["fig_two"]["run_ids"] == ["r2"]


def test_save_figure_defaults_run_ids_to_empty_list():
    fig, _ax = plt.subplots()
    style_mod.save_figure(fig, "no_runs")
    plt.close(fig)

    with style_mod.MANIFEST_PATH.open(encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["no_runs"]["run_ids"] == []


def test_family_color_known_and_unknown():
    assert style_mod.family_color("c_mlp") == style_mod.FAMILY_COLORS["c_mlp"]
    with pytest.raises(ValueError):
        style_mod.family_color("not_a_family")


def test_all_seven_families_have_distinct_colors():
    families = ["c_mlp", "c_ff", "c_rff_matched", "q_serial", "q_parallel", "q_random", "q_octave"]
    assert set(style_mod.FAMILY_COLORS.keys()) == set(families)
    colors = [style_mod.FAMILY_COLORS[f] for f in families]
    assert len(set(colors)) == 7
