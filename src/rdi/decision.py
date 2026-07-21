"""Deciding which remediation to run, and whether to run one at all.

Knowing what broke isn't the same as knowing what to do. Every remediation costs something:
a restart drops in-flight requests, a failover moves traffic to a colder replica, scaling out
bills more, a rollback loses whatever the release shipped. Acting is only worth it when the
action changes the outcome by more than it costs.

That's an incremental-effect question, not a risk question, and the difference shows up in two
places this data was built around.

Absorbed traffic spikes. About 40% of spikes never breach the SLO, so scaling out for them
buys nothing and still bills. A policy that ranks by "how bad does this look" treats them the
same as a breaking spike, because they look identical on rps and CPU until the latency lands.

Wrong remediations. Restarting a service whose upstream database is down costs the restart and
fixes nothing. The effect isn't merely smaller, it's negative. A risk threshold cannot express
that, since it only ranks how alarming an event is and never asks which of five actions helps.

So `uplift` estimates, per event and per action, the change in outcome from acting versus not,
and `policy` picks the action with the best expected value net of cost. `qini` and
`policy_value` score it against the baselines worth beating: treat nobody, treat everybody,
and treat whatever looks riskiest.

The counterfactual problem is real here as everywhere. You never observe both outcomes for the
same event. This module resolves it the way an experiment does, by randomising the action at
logging time (`simulate_experiment`) so that treated and untreated groups are comparable, then
estimating the effect as a difference between them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rdi.events import ACTIONS, REMEDIATIONS

# What each action costs to run, in the same units as the outcome below. Rough, and deliberately
# not zero: an intervention that were free would make the whole decision layer pointless.
ACTION_COST = {
    "none": 0.0,
    "restart": 0.30,      # drops in-flight requests
    "failover": 0.35,     # traffic to a colder replica
    "scale_out": 0.25,    # extra capacity, billed
    "rollback": 0.40,     # loses the release
}

# How much of a breach the correct remediation removes. Below 1.0 because no action is a
# complete fix inside one tick.
CORRECT_ACTION_EFFECT = 0.75
# What a wrong remediation does. Negative: it costs the disruption and leaves the fault.
WRONG_ACTION_EFFECT = -0.10


@dataclass
class Policy:
    """A decision rule, plus how it scored."""
    name: str
    actions: list[str]
    value: float
    treated: int


def true_effect(incident: str | None, action: str) -> float:
    """Ground-truth incremental effect, used to score a policy rather than to build one.

    Available only because the stream is generated. A real deployment measures this with an
    experiment instead, which is what `simulate_experiment` imitates.
    """
    if action == "none":
        return 0.0
    if incident is None:
        return WRONG_ACTION_EFFECT          # acting on a healthy service is pure loss
    return CORRECT_ACTION_EFFECT if REMEDIATIONS.get(incident) == action else WRONG_ACTION_EFFECT


def simulate_experiment(events: list[dict], seed: int = 0) -> dict:
    """Randomise an action per event and record what happened.

    Randomising is what makes the groups comparable. If actions were assigned by how alarming
    an event looked, treated and untreated events would differ in ways that have nothing to do
    with the action, and the measured difference would be confounded.
    """
    rng = np.random.default_rng(seed)
    assigned = [ACTIONS[int(i)] for i in rng.integers(0, len(ACTIONS), len(events))]
    outcomes, breach = [], []
    for e, a in zip(events, assigned, strict=True):
        base = float(e["label"])                       # 1.0 when this tick breaches the SLO
        gain = true_effect(e["incident_type"], a) * base
        outcomes.append(base - gain - ACTION_COST[a] * 0.0)   # cost is charged by the policy
        breach.append(base)
    return {"action": assigned, "outcome": np.array(outcomes), "breach": np.array(breach),
            "events": events}


def uplift(exp: dict) -> dict[str, dict[str, float]]:
    """Average incremental effect of each action, per predicted incident class.

    A difference in means inside each class. Because the action was randomised, the two groups
    are comparable and the difference estimates the effect rather than a correlation.
    """
    events, actions, outcome = exp["events"], exp["action"], exp["outcome"]
    classes = sorted({e["incident_type"] or "normal" for e in events})
    table: dict[str, dict[str, float]] = {}
    for cls in classes:
        idx = [i for i, e in enumerate(events) if (e["incident_type"] or "normal") == cls]
        untreated = [outcome[i] for i in idx if actions[i] == "none"]
        base = float(np.mean(untreated)) if untreated else 0.0
        row = {}
        for a in ACTIONS:
            if a == "none":
                row[a] = 0.0
                continue
            treated = [outcome[i] for i in idx if actions[i] == a]
            # Outcome is "badness", so a drop is a gain.
            row[a] = float(base - np.mean(treated)) if treated else 0.0
        table[cls] = row
    return table


class TLearner:
    """One outcome model per action, so the effect can depend on the event.

    The per-class table above is too coarse to decide anything. It averages over every event in
    a class, and most of them aren't breaching, so an action that helps a lot on the ticks that
    matter shows up as a small average. Charge that average against a real cost and the policy
    declines to act at all: measured, the class table put memory_leak restarts at 0.134 against
    a cost of 0.30 and never restarted anything.

    Conditioning on features fixes it. Fit E[outcome | X] separately for each action, and the
    effect for a given event is the untreated prediction minus the treated one. The action was
    randomised, so the per-action subsets stay comparable.
    """

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.models: dict[str, object] = {}

    def fit(self, X: np.ndarray, actions: list[str], outcome: np.ndarray) -> TLearner:
        from lightgbm import LGBMRegressor
        acts = np.asarray(actions)
        for a in ACTIONS:
            m = acts == a
            if m.sum() < 20:          # too few to fit anything trustworthy
                continue
            reg = LGBMRegressor(n_estimators=60, num_leaves=7, learning_rate=0.1,
                                min_child_samples=20, random_state=self.seed, verbose=-1,
                                n_jobs=1)
            reg.fit(X[m], outcome[m])
            self.models[a] = reg
        if "none" not in self.models:
            raise ValueError("no untreated events to compare against")
        return self

    def effects(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Per-event incremental effect of each action. Outcome is badness, so a drop is a gain."""
        base = self.models["none"].predict(X)
        return {a: base - m.predict(X) for a, m in self.models.items() if a != "none"}


def policy(predictions: list[str], table: dict[str, dict[str, float]],
           charge_cost: bool = True) -> list[str]:
    """Pick, per event, the action whose estimated gain best clears its cost.

    Class-level version, kept because it's the obvious thing to try and it's measurably not
    good enough. `policy_from_effects` is the one to use.
    """
    out = []
    for cls in predictions:
        row = table.get(cls) or table.get("normal") or {}
        best, best_v = "none", 0.0
        for a in ACTIONS:
            if a == "none":
                continue
            v = row.get(a, 0.0) - (ACTION_COST[a] if charge_cost else 0.0)
            if v > best_v:
                best, best_v = a, v
        out.append(best)
    return out


def policy_from_effects(effects: dict[str, np.ndarray], charge_cost: bool = True) -> list[str]:
    """Per event, take the action whose estimated effect best clears its cost, else do nothing."""
    n = len(next(iter(effects.values())))
    out = []
    for i in range(n):
        best, best_v = "none", 0.0
        for a, vals in effects.items():
            v = float(vals[i]) - (ACTION_COST[a] if charge_cost else 0.0)
            if v > best_v:
                best, best_v = a, v
        out.append(best)
    return out


def risk_threshold_policy(predictions: list[str], scores: np.ndarray,
                          quantile: float = 0.9) -> list[str]:
    """The baseline worth beating: act on whatever looks riskiest.

    It ranks events well and still loses value, because ranking says nothing about which of
    five actions helps, and nothing about whether acting is worth its cost.
    """
    cut = float(np.quantile(scores, quantile))
    return [REMEDIATIONS.get(p, "restart") if s >= cut else "none"
            for p, s in zip(predictions, scores, strict=True)]


def policy_value(events: list[dict], actions: list[str], charge_cost: bool = True) -> float:
    """Total value a policy delivers: breaches avoided, minus what the actions cost."""
    total = 0.0
    for e, a in zip(events, actions, strict=True):
        total += true_effect(e["incident_type"], a) * float(e["label"])
        if charge_cost:
            total -= ACTION_COST[a]
    return float(total)


def qini_curve(events: list[dict], ranking: np.ndarray, actions: list[str],
               steps: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative gain as the policy is allowed to treat more of the stream, best first.

    A policy that ranks well rises steeply then flattens. A policy no better than random is a
    straight line, which is the comparison the curve exists to make.
    """
    order = np.argsort(-ranking)
    fracs = np.linspace(0, 1, steps + 1)
    gains = []
    for f in fracs:
        k = int(f * len(order))
        chosen = set(order[:k].tolist())
        gains.append(sum(
            true_effect(events[i]["incident_type"], actions[i]) * float(events[i]["label"])
            - ACTION_COST[actions[i]]
            for i in chosen))
    return fracs, np.array(gains)


def evaluate_policies(events: list[dict], predictions: list[str],
                      table: dict[str, dict[str, float]], scores: np.ndarray,
                      effects: dict[str, np.ndarray] | None = None) -> list[Policy]:
    """Score every policy against the baselines a causal one has to beat to be worth shipping.

    The risk-threshold baseline is deliberately strong: it gets the correct remediation from
    the runbook for whatever class was predicted, which is what a real on-call rule would do.
    Beating a weak baseline would prove nothing.
    """
    candidates = {
        "treat nobody": ["none"] * len(events),
        "treat every alert (runbook)": [REMEDIATIONS.get(p, "none") for p in predictions],
        "risk threshold, top 10% (runbook)": risk_threshold_policy(predictions, scores, 0.9),
        "risk threshold, top 25% (runbook)": risk_threshold_policy(predictions, scores, 0.75),
        "causal, per-class table": policy(predictions, table),
    }
    if effects is not None:
        candidates["causal, T-learner (cost-aware)"] = policy_from_effects(effects)
    return sorted(
        (Policy(n, a, policy_value(events, a), sum(x != "none" for x in a))
         for n, a in candidates.items()),
        key=lambda p: -p.value)
