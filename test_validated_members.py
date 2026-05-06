"""Tests for validated_members.py — store + fuzzy lookup."""
from __future__ import annotations

from datetime import date

import pytest


@pytest.fixture
def vm(tmp_path, monkeypatch):
    import validated_members as v

    target = tmp_path / "validated_members.json"
    monkeypatch.setattr(v, "VALIDATED_MEMBERS_PATH", target)
    return v


def test_empty_when_file_missing(vm):
    assert vm.list_members() == []


def test_add_member_persists(vm):
    res = vm.add_member("Maggie Cochrane", added_by="geoff",
                        today=date(2026, 5, 6))
    assert res == {
        "ok": True,
        "added": True,
        "name": "Maggie Cochrane",
        "entry": {
            "name": "Maggie Cochrane",
            "added_by": "geoff",
            "added_at": "2026-05-06",
        },
    }
    assert [m["name"] for m in vm.list_members()] == ["Maggie Cochrane"]


def test_add_member_idempotent(vm):
    vm.add_member("Maggie Cochrane", added_by="geoff")
    res = vm.add_member("maggie cochrane", added_by="geoff")
    assert res["added"] is False
    assert res["reason"] == "already_validated"
    assert len(vm.list_members()) == 1


def test_add_member_rejects_empty(vm):
    assert vm.add_member("   ").get("error") == "empty_name"


def test_lookup_uses_roster_first(vm):
    vm.add_member("Maggie Cochrane")
    res = vm.is_known_member(
        "Geoff Chapman",
        roster_names=["Geoff Chapman", "Sarah Forster"],
    )
    assert res.found is True
    assert res.canonical_name == "Geoff Chapman"
    assert res.source == "roster"


def test_lookup_falls_back_to_whitelist(vm):
    vm.add_member("Maggie Cochrane")
    res = vm.is_known_member("Maggie", roster_names=["Geoff Chapman"])
    assert res.found is True
    assert res.canonical_name == "Maggie Cochrane"
    assert res.source == "validated_members"


def test_lookup_case_insensitive(vm):
    vm.add_member("Maggie Cochrane")
    res = vm.is_known_member("maggie cochrane", roster_names=[])
    assert res.found is True
    assert res.canonical_name == "Maggie Cochrane"


def test_lookup_substring(vm):
    vm.add_member("Maggie Cochrane")
    res = vm.is_known_member("cochrane", roster_names=[])
    assert res.found is True
    assert res.canonical_name == "Maggie Cochrane"


def test_lookup_ambiguous_returns_candidates(vm):
    res = vm.is_known_member(
        "Andy",
        roster_names=["Andy Matthews", "Andy Phillips", "Sarah Forster"],
    )
    assert res.found is False
    assert res.source == "roster"
    assert set(res.candidates) == {"Andy Matthews", "Andy Phillips"}


def test_lookup_not_found(vm):
    vm.add_member("Maggie Cochrane")
    res = vm.is_known_member(
        "Nobody Atall",
        roster_names=["Geoff Chapman"],
    )
    assert res.found is False
    assert res.candidates == ()


def test_lookup_empty_query(vm):
    res = vm.is_known_member("   ", roster_names=["Geoff Chapman"])
    assert res.found is False
    assert res.canonical_name is None


def test_lookup_with_no_roster_only_uses_whitelist(vm):
    vm.add_member("Maggie Cochrane")
    res = vm.is_known_member("maggie", roster_names=None)
    assert res.found is True
    assert res.source == "validated_members"
