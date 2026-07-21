"""Tests for the intervention policy.

The load-bearing one is `test_causal_policy_beats_the_risk_threshold`. If a policy that models
incremental effect can't beat one that just ranks by risk, the whole layer is ceremony and
should go.

Two failures are pinned deliberately: the per-class uplift table is too coarse to be useful
(`test_per_class_table_is_too_coarse_to_act_on`), and a risk threshold set slightly wrong turns
value negative (`test_risk_threshold_is_sensitive_to_where_you_set_it`).
"""
from __future__ import annotations

import numpy as np
import pytest

from rdi.decision import (
    ACTION_COST,
    CORRECT_ACTION_EFFECT,
    WRONG_ACTION_EFFECT,
    TLearner,
    evaluate_policies,
    policy,
    policy_from_effects,
    policy_value,
    qini_curve,
    risk_threshold_policy,
    simulate_experiment,
    true_effect,
    uplift,
)
from rdi.events import ACTIONS, REMEDIATIONS, generate
from rdi.model import build_dataset, temporal_split, train


@pytest.fixture(scope="module")
def world():
    ev = sorted(generate(n_ticks=6000, seed=7, incidents_per_service=10),
                key=lambda e: e["ts"])
    X, y, ts = build_dataset(ev)
    Xtr, Xte, ytr, yte = temporal_split(X, y, ts)
    clf = train(Xtr, ytr)
    cut = np.quantile(ts, 0.7)
    warm = [e for e in ev if e["ts"] >= 60.0]
    tr = [e for e, k in zip(warm, ts < cut, strict=True) if k]
    te = [e for e, k in zip(warm, ts >= cut, strict=True) if k]
    pred = list(clf.predict(Xte))
    risk = 1.0 - clf.predict_proba(Xte)[:, list(clf.classes_).index("normal")]
    exp = simulate_experiment(tr, seed=1)
    tbl = uplift(exp)
    eff = TLearner().fit(Xtr, exp["action"], exp["outcome"]).effects(Xte)
    return dict(tr=tr, te=te, pred=pred, risk=risk, exp=exp, table=tbl, effects=eff,
                Xtr=Xtr, Xte=Xte)


# ---- the effect model ----

def test_acting_correctly_helps_and_acting_wrongly_hurts():
    """A wrong remediation isn't merely less useful, it's worse than doing nothing."""
    assert true_effect("memory_leak", "restart") == CORRECT_ACTION_EFFECT
    assert true_effect("memory_leak", "failover") == WRONG_ACTION_EFFECT
    assert WRONG_ACTION_EFFECT < 0 < CORRECT_ACTION_EFFECT
    assert true_effect("memory_leak", "none") == 0.0


def test_acting_on_a_healthy_service_is_pure_loss():
    assert true_effect(None, "restart") < 0


def test_every_action_has_a_cost_except_doing_nothing():
    assert ACTION_COST["none"] == 0.0
    assert all(ACTION_COST[a] > 0 for a in ACTIONS if a != "none")


def test_experiment_randomises_the_action(world):
    """Randomisation is what makes treated and untreated groups comparable."""
    counts = {a: world["exp"]["action"].count(a) for a in ACTIONS}
    assert all(c > 0 for c in counts.values())
    spread = max(counts.values()) / min(counts.values())
    assert spread < 1.5, f"assignment is lopsided: {counts}"


def test_experiment_is_reproducible():
    ev = generate(n_ticks=300, seed=3)
    assert simulate_experiment(ev, seed=5)["action"] == simulate_experiment(ev, seed=5)["action"]


# ---- uplift recovers the runbook ----

def test_uplift_recovers_the_right_action_per_incident(world):
    """The correct remediation was never a field on the event. It has to be estimated."""
    for cls, row in world["table"].items():
        if cls == "normal":
            continue
        best = max((a for a in ACTIONS if a != "none"), key=lambda a: row[a])
        assert best == REMEDIATIONS[cls], f"{cls}: estimated {best}, truth {REMEDIATIONS[cls]}"


def test_wrong_actions_estimate_as_harmful(world):
    for cls, row in world["table"].items():
        if cls == "normal":
            continue
        for a in ACTIONS:
            if a not in ("none", REMEDIATIONS[cls]):
                assert row[a] < 0.05, f"{cls}/{a} looks helpful and should not"


# ---- the reason the layer exists ----

def test_causal_policy_beats_the_risk_threshold(world):
    """Modelling incremental effect has to beat ranking by risk, or it isn't worth shipping."""
    ranked = evaluate_policies(world["te"], world["pred"], world["table"], world["risk"],
                               world["effects"])
    by_name = {p.name: p for p in ranked}
    causal = by_name["causal, T-learner (cost-aware)"]
    best_baseline = max(p.value for n, p in by_name.items() if not n.startswith("causal"))
    assert causal.value > best_baseline
    assert ranked[0].name == "causal, T-learner (cost-aware)"


def test_causal_policy_acts_less_than_the_threshold_it_beats(world):
    """More value from fewer interventions is the whole claim."""
    by_name = {p.name: p for p in evaluate_policies(
        world["te"], world["pred"], world["table"], world["risk"], world["effects"])}
    causal = by_name["causal, T-learner (cost-aware)"]
    rt = by_name["risk threshold, top 10% (runbook)"]
    assert causal.value > rt.value
    assert causal.treated < rt.treated


def test_doing_nothing_scores_zero(world):
    assert policy_value(world["te"], ["none"] * len(world["te"])) == 0.0


def test_treating_everything_wastes_value(world):
    """Acting on every alert is positive but leaks cost on events that never needed it."""
    everything = [REMEDIATIONS.get(p, "none") for p in world["pred"]]
    by_name = {p.name: p for p in evaluate_policies(
        world["te"], world["pred"], world["table"], world["risk"], world["effects"])}
    causal_value = by_name["causal, T-learner (cost-aware)"].value
    assert 0 < policy_value(world["te"], everything) < causal_value


# ---- pinned failures ----

def test_per_class_table_is_too_coarse_to_act_on(world):
    """Averaging over a whole class hides the effect on the ticks that matter.

    Most events in a class aren't breaching, so a class average understates the gain, and a
    cost-aware rule then refuses to act. Measured, the table put memory_leak restarts at ~0.13
    against a cost of 0.30 and never restarted anything.
    """
    coarse = policy(world["pred"], world["table"])
    fine = policy_from_effects(world["effects"])
    assert policy_value(world["te"], coarse) < policy_value(world["te"], fine)
    assert "restart" not in set(coarse), "the coarse table now acts on leaks, re-check the claim"


def test_risk_threshold_is_sensitive_to_where_you_set_it(world):
    """Same rule, different cutoff, and the sign of the value flips.

    At the top 10% it's a strong baseline. At the top 25% it treats so many non-breaching
    events that the costs outrun the benefit. The causal policy picks its own operating point
    instead of having one handed to it.
    """
    tight = policy_value(world["te"], risk_threshold_policy(world["pred"], world["risk"], 0.9))
    loose = policy_value(world["te"], risk_threshold_policy(world["pred"], world["risk"], 0.75))
    assert tight > 0 > loose


# ---- mechanics ----

def test_tlearner_needs_untreated_events():
    X = np.random.default_rng(0).random((60, 8))
    with pytest.raises(ValueError, match="no untreated"):
        TLearner().fit(X, ["restart"] * 60, np.zeros(60))


def test_effects_cover_every_action_that_was_tried(world):
    assert set(world["effects"]) == {a for a in ACTIONS if a != "none"}
    assert all(len(v) == len(world["Xte"]) for v in world["effects"].values())


def test_policy_only_emits_known_actions(world):
    assert set(policy_from_effects(world["effects"])) <= set(ACTIONS)


def test_policy_does_nothing_when_costs_swamp_the_gains(world):
    tiny = {a: np.full(len(world["Xte"]), 0.01) for a in ACTIONS if a != "none"}
    assert set(policy_from_effects(tiny)) == {"none"}


def test_ignoring_cost_makes_the_policy_act_more(world):
    charged = policy_from_effects(world["effects"], charge_cost=True)
    free = policy_from_effects(world["effects"], charge_cost=False)
    assert sum(a != "none" for a in free) > sum(a != "none" for a in charged)


def test_qini_curve_rises_and_ends_at_the_policy_value(world):
    actions = policy_from_effects(world["effects"])
    fracs, gains = qini_curve(world["te"], world["risk"], actions, steps=10)
    assert fracs[0] == 0.0 and gains[0] == 0.0
    assert gains[-1] == pytest.approx(policy_value(world["te"], actions), abs=1e-6)
