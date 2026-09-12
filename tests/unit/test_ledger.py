"""The ledger is what stops a re-run from repeating finished work."""

import json

from clinhallu.experiments.ledger import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_INTERRUPTED,
    Ledger,
    fingerprint,
)


def test_completed_units_are_reported_as_complete(tmp_path):
    ledger = Ledger(tmp_path / "l.json")
    ledger.start("B")
    ledger.complete("B")
    assert ledger.is_complete("B")
    assert not ledger.is_complete("C")


def test_state_survives_a_new_process(tmp_path):
    Ledger(tmp_path / "l.json").complete("B")
    assert Ledger(tmp_path / "l.json").is_complete("B")


def test_written_immediately_so_a_crash_cannot_erase_it(tmp_path):
    path = tmp_path / "l.json"
    ledger = Ledger(path)
    ledger.complete("B")
    # No flush, no close, no exit hook: the file is already correct on disk.
    assert json.loads(path.read_text())["units"]["B"]["status"] == STATUS_COMPLETED


def test_failed_and_interrupted_units_run_again(tmp_path):
    ledger = Ledger(tmp_path / "l.json")
    ledger.fail("C", "out of memory")
    ledger.interrupt("D")
    assert ledger.pending(["C", "D"]) == ["C", "D"]
    assert ledger.status("C") == STATUS_FAILED
    assert ledger.status("D") == STATUS_INTERRUPTED
    assert "out of memory" in ledger.entry("C")["error"]


def test_changed_configuration_invalidates_a_completed_unit(tmp_path):
    ledger = Ledger(tmp_path / "l.json")
    original = fingerprint({"lr": 2e-5})
    ledger.start("B", fingerprint_value=original)
    ledger.complete("B", fingerprint=original)

    assert ledger.is_complete("B", fingerprint_value=original)
    # Someone edits the learning rate: the old result no longer applies.
    assert not ledger.is_complete("B", fingerprint_value=fingerprint({"lr": 5e-5}))


def test_legacy_entries_without_a_fingerprint_are_trusted(tmp_path):
    """Work finished before fingerprinting existed is not thrown away."""
    path = tmp_path / "l.json"
    path.write_text(json.dumps({"name": "x", "units": {"B": {"status": "completed"}}}))
    assert Ledger(path).is_complete("B", fingerprint_value="anything")


def test_attempts_are_counted(tmp_path):
    ledger = Ledger(tmp_path / "l.json")
    ledger.start("B")
    ledger.fail("B", "first try")
    ledger.start("B")
    ledger.complete("B")
    assert ledger.entry("B")["attempts"] == 2
    assert "error" not in ledger.entry("B")


def test_reset_forces_a_rerun(tmp_path):
    ledger = Ledger(tmp_path / "l.json")
    ledger.complete("B")
    ledger.complete("C")
    ledger.reset(["B"])
    assert not ledger.is_complete("B")
    assert ledger.is_complete("C")
    ledger.reset()
    assert not ledger.is_complete("C")


def test_unreadable_ledger_does_not_crash_the_run(tmp_path):
    path = tmp_path / "l.json"
    path.write_text("{ this is not json")
    ledger = Ledger(path)
    assert ledger.pending(["B"]) == ["B"]
    ledger.complete("B")
    assert Ledger(path).is_complete("B")


def test_summary_counts_statuses(tmp_path):
    ledger = Ledger(tmp_path / "l.json")
    ledger.complete("B")
    ledger.complete("C")
    ledger.fail("D", "boom")
    assert ledger.summary()["counts"] == {STATUS_COMPLETED: 2, STATUS_FAILED: 1}
