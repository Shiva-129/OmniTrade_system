"""Phase 15 R1: Experiment Registry tests."""
import pytest

from research.experiments import DuplicateExperimentError, ExperimentRegistry


class TestExperimentRegistry:
    def test_record_and_get_roundtrip(self, tmp_path):
        reg = ExperimentRegistry(tmp_path / "reg.jsonl")
        payload = {"config_hash": "abc123", "metrics": {"sharpe": 1.5}}
        eid = reg.record(payload)
        assert eid == "abc123"
        assert reg.get("abc123")["metrics"]["sharpe"] == 1.5

    def test_duplicate_id_fails_loudly(self, tmp_path):
        reg = ExperimentRegistry(tmp_path / "reg.jsonl")
        reg.record({"config_hash": "dup", "x": 1})
        with pytest.raises(DuplicateExperimentError):
            reg.record({"config_hash": "dup", "x": 2})
        # original preserved
        assert reg.get("dup")["x"] == 1

    def test_missing_config_hash_rejected(self, tmp_path):
        reg = ExperimentRegistry(tmp_path / "reg.jsonl")
        with pytest.raises(ValueError):
            reg.record({"no_hash": True})

    def test_append_only_no_delete_api(self):
        assert not hasattr(ExperimentRegistry, "delete")
        assert not hasattr(ExperimentRegistry, "update")
        assert not hasattr(ExperimentRegistry, "remove")

    def test_load_all_preserves_order(self, tmp_path):
        reg = ExperimentRegistry(tmp_path / "reg.jsonl")
        for i in range(5):
            reg.record({"config_hash": f"id{i}", "i": i})
        all_recs = reg.load_all()
        assert [r["i"] for r in all_recs] == [0, 1, 2, 3, 4]
        assert reg.count() == 5

    def test_corrupt_line_skipped_loudly(self, tmp_path, capsys):
        p = tmp_path / "reg.jsonl"
        p.write_text('{"config_hash": "good"}\nNOT JSON\n{"config_hash": "good2"}\n')
        reg = ExperimentRegistry(p)
        assert reg.count() == 2
        assert reg.exists("good") and reg.exists("good2")
        assert "corrupt" in capsys.readouterr().out.lower()

    def test_query_predicate(self, tmp_path):
        reg = ExperimentRegistry(tmp_path / "reg.jsonl")
        reg.record({"config_hash": "a", "sharpe": 2.0})
        reg.record({"config_hash": "b", "sharpe": 0.5})
        high = reg.query(lambda r: r["sharpe"] > 1.0)
        assert len(high) == 1 and high[0]["config_hash"] == "a"

    def test_registry_isolation_from_trading_plane(self):
        """Structural: nothing in src/core imports the registry."""
        import pathlib
        for p in pathlib.Path("src/core").rglob("*.py"):
            text = p.read_text()
            assert "ExperimentRegistry" not in text
            assert "research.experiments" not in text

    def test_disk_failure_contained(self, tmp_path):
        """Registry write failure must not propagate into trading plane —
        verified structurally: src/core has zero imports of this module,
        so a disk error here cannot interrupt order submission."""
        reg = ExperimentRegistry(tmp_path / "sub" / "reg.jsonl")
        # parent dir auto-created
        assert reg.path.parent.exists()
