"""
Research verdict (Phase 9) -- a MECHANICAL evidence summary.

HARD RULE: this system never declares a strategy profitable or
production-ready. The only verdicts it can emit:

    PASS_CANDIDATE  evidence gates passed; MAY proceed to paper trading
    REJECT          at least one evidence gate failed
    INCONCLUSIVE    insufficient evidence to judge (gaps marked None)

Every gate is explicit, thresholded, and reproducible.
"""
from typing import Literal

from pydantic import BaseModel

DISCLAIMER = (
    "This is a statistical research decision, NOT a claim of "
    "profitability or production-readiness. Positive backtests do not "
    "guarantee future performance."
)


class VerdictThresholds(BaseModel):
    model_config = {"frozen": True}

    min_windows: int = 3
    min_oos_positive_rate: float = 0.6
    min_sensitivity_fraction: float = 0.6
    require_mc_p05_positive: bool = True
    require_regime_consistency: bool = True


class Gate(BaseModel):
    model_config = {"frozen": True}

    name: str
    passed: bool | None               # None == cannot evaluate
    detail: str


class ResearchVerdict(BaseModel):
    verdict: Literal["PASS_CANDIDATE", "REJECT", "INCONCLUSIVE"]
    gates: list[Gate]
    disclaimer: str = DISCLAIMER


def decide(walkforward_report, sensitivity_report,
           mc_report=None, regime_report=None,
           thresholds: VerdictThresholds | None = None) -> ResearchVerdict:
    th = thresholds or VerdictThresholds()
    gates: list[Gate] = []

    # --- walk-forward evidence ---
    n_evaluated = sum(
        1 for w in walkforward_report.windows if w.oos_return is not None)
    if n_evaluated < th.min_windows:
        gates.append(Gate(
            name="walk_forward_sufficient_windows", passed=None,
            detail=f"{n_evaluated} evaluated windows < {th.min_windows}"))
    else:
        mean_ok = (walkforward_report.mean_oos_return or 0.0) > 0
        rate_ok = ((walkforward_report.positive_rate or 0.0)
                   >= th.min_oos_positive_rate)
        gates.append(Gate(
            name="walk_forward_mean_oos_return_positive",
            passed=mean_ok,
            detail=f"mean={walkforward_report.mean_oos_return}"))
        gates.append(Gate(
            name="walk_forward_positive_rate",
            passed=rate_ok,
            detail=(f"rate={walkforward_report.positive_rate} vs "
                    f"threshold={th.min_oos_positive_rate}")))

    # --- neighborhood sensitivity ---
    if sensitivity_report.fraction_positive is None:
        gates.append(Gate(name="neighborhood_sensitivity", passed=None,
                          detail="no evaluable neighbors"))
    else:
        ok = sensitivity_report.fraction_positive \
            >= th.min_sensitivity_fraction
        gates.append(Gate(
            name="neighborhood_sensitivity", passed=ok,
            detail=(f"fraction={sensitivity_report.fraction_positive} "
                    f"vs threshold={th.min_sensitivity_fraction}")))

    # --- monte carlo ---
    if mc_report is None:
        gates.append(Gate(name="monte_carlo_p05", passed=None,
                          detail="no trade data"))
    elif mc_report.p50_total_pnl <= 0 and mc_report.mean_total_pnl <= 0:
        gates.append(Gate(name="monte_carlo_p05", passed=False,
                          detail="median/mean cumulative pnl not positive"))
    elif not th.require_mc_p05_positive:
        gates.append(Gate(name="monte_carlo_p05", passed=True,
                          detail="requirement disabled"))
    else:
        ok = mc_report.p05_total_pnl > 0
        gates.append(Gate(
            name="monte_carlo_p05", passed=ok,
            detail=f"p05={mc_report.p05_total_pnl}"))

    # --- regime consistency ---
    if regime_report is None:
        gates.append(Gate(name="regime_consistency", passed=None,
                          detail="not provided"))
    else:
        if regime_report.consistent_sign is None:
            gates.append(Gate(name="regime_consistency", passed=None,
                              detail="a half was flat; sign undefined"))
        elif not th.require_regime_consistency:
            gates.append(Gate(name="regime_consistency", passed=True,
                              detail="requirement disabled"))
        else:
            gates.append(Gate(
                name="regime_consistency", passed=regime_report.consistent_sign,
                detail=(f"halves: {regime_report.first_half_return} / "
                        f"{regime_report.second_half_return}")))

    any_failed = any(g.passed is False for g in gates)
    any_unknown = any(g.passed is None for g in gates)

    if any_failed:
        verdict = "REJECT"
    elif any_unknown:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "PASS_CANDIDATE"

    return ResearchVerdict(verdict=verdict, gates=gates)


# Forbidden-language guard for the whole package.
FORBIDDEN_CLAIMS = ("is profitable", "production ready",
                    "production-ready", "will make money")
