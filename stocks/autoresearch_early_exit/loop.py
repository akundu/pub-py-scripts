"""
loop.py — Hill-climbing sweep over early-exit threshold combinations.

Two phases (mirrors autoresearch_dte/loop_v2.py pattern):
  Phase 1: random seeds — score N random configs, keep best
  Phase 2: hill-climb — mutate one knob at a time, keep if score improves

Score = annualized_sharpe × (1 - dd_penalty)
  where dd_penalty = max_drawdown_pct / 30%

Config space:
  profit_target_pct : int  (30–95, step 5)
  stop_loss_pct     : int  (9999=disabled, 50, 100, 150, 200, 300)
  apply_dtes        : list[int]  subsets of [0,1,2,3]
  apply_tiers       : list[str]  subsets of [aggressive, moderate, conservative]
  apply_sides       : list[str]  subsets of [put, call]

Usage:
    python3 loop.py
    python3 loop.py --trials 400 --seeds 50
    python3 loop.py --dtes 1,2,3 --sides put   # restrict search space
    python3 loop.py --help

Outputs:
    results_{prefix}.tsv   — every trial
    best_config_{prefix}.json — best config found
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from pathlib import Path

from experiment import (
    PROFIT_TARGETS, STOP_LOSSES, APPLY_DTES, APPLY_TIERS, APPLY_SIDES,
    ALL_DTES, ALL_TIERS, ALL_SIDES,
    default_config, run_experiment, score_metrics
)

ROOT = Path(__file__).parent


# ── Config serialization ──────────────────────────────────────────────────────

def _ser(cfg: dict) -> dict:
    out = dict(cfg)
    out["apply_dtes"]  = sorted(cfg["apply_dtes"])
    out["apply_tiers"] = sorted(cfg["apply_tiers"])
    out["apply_sides"] = sorted(cfg["apply_sides"])
    return out


def chash(cfg: dict) -> str:
    return hashlib.md5(
        json.dumps(_ser(cfg), sort_keys=True).encode()
    ).hexdigest()[:7]


# ── Random config / mutation ──────────────────────────────────────────────────

def random_config(rng: random.Random,
                  allowed_dtes: list[int],
                  allowed_sides: list[str],
                  fixed_pt: int | None = None,
                  fixed_sl: int | None = None) -> dict:
    dtes = [d for d in rng.sample(allowed_dtes, rng.randint(1, len(allowed_dtes)))]
    tiers = rng.choice(APPLY_TIERS)
    sides = [s for s in allowed_sides if rng.random() < 0.7] or [rng.choice(allowed_sides)]
    return {
        "profit_target_pct": fixed_pt if fixed_pt is not None else rng.choice(PROFIT_TARGETS),
        "stop_loss_pct":     fixed_sl if fixed_sl is not None else rng.choice(STOP_LOSSES),
        "apply_dtes":        sorted(dtes),
        "apply_tiers":       sorted(tiers),
        "apply_sides":       sorted(sides),
    }


def mutate(cfg: dict, rng: random.Random,
           allowed_dtes: list[int],
           allowed_sides: list[str],
           fixed_pt: int | None = None,
           fixed_sl: int | None = None) -> tuple[dict, str]:
    new = copy.deepcopy(cfg)
    # Only include ops for the axes that are free
    ops = []
    if fixed_pt is None:
        ops.append("shift_profit_target")
    if fixed_sl is None:
        ops.append("shift_stop_loss")
    ops += ["toggle_dte", "swap_tiers", "toggle_side"]
    op = rng.choice(ops)

    if op == "shift_profit_target":
        idx = PROFIT_TARGETS.index(new["profit_target_pct"])
        idx = max(0, min(len(PROFIT_TARGETS) - 1,
                         idx + rng.choice([-1, -1, 1, 1, 2, -2])))
        new["profit_target_pct"] = PROFIT_TARGETS[idx]

    elif op == "shift_stop_loss":
        idx = STOP_LOSSES.index(new["stop_loss_pct"])
        idx = max(0, min(len(STOP_LOSSES) - 1, idx + rng.choice([-1, 1])))
        new["stop_loss_pct"] = STOP_LOSSES[idx]

    elif op == "toggle_dte":
        if len(allowed_dtes) > 1:
            d = rng.choice(allowed_dtes)
            dtes = list(new["apply_dtes"])
            if d in dtes and len(dtes) > 1:
                dtes.remove(d)
            elif d not in dtes:
                dtes.append(d)
            new["apply_dtes"] = sorted(dtes)

    elif op == "swap_tiers":
        new["apply_tiers"] = sorted(rng.choice(APPLY_TIERS))

    elif op == "toggle_side":
        if len(allowed_sides) > 1:
            s = rng.choice(allowed_sides)
            sides = list(new["apply_sides"])
            if s in sides and len(sides) > 1:
                sides.remove(s)
            elif s not in sides:
                sides.append(s)
            new["apply_sides"] = sorted(sides)

    # Enforce fixed axes (overwrite whatever mutation did)
    if fixed_pt is not None:
        new["profit_target_pct"] = fixed_pt
    if fixed_sl is not None:
        new["stop_loss_pct"] = fixed_sl

    return new, op


# ── Logging ───────────────────────────────────────────────────────────────────

def log_trial(f, trial: int, h: str, status: str, op: str, m: dict, cfg: dict):
    f.write(
        f"{trial}\t{h}\t{status}\t{op}\t"
        f"{m.get('score', 0):.4f}\t"
        f"{m.get('annualized_sharpe', 0):.3f}\t"
        f"{m.get('hold_sharpe', 0):.3f}\t"
        f"{m.get('annualized_roi_pct', 0):.2f}\t"
        f"{m.get('max_drawdown_pct', 0):.2f}\t"
        f"{m.get('win_day_pct', 0):.2f}\t"
        f"{m.get('improvement_pct', 0):+.1f}\t"
        f"{cfg['profit_target_pct']}\t"
        f"{cfg['stop_loss_pct']}\t"
        f"{','.join(str(d) for d in sorted(cfg['apply_dtes']))}\t"
        f"{','.join(sorted(cfg['apply_tiers']))}\t"
        f"{','.join(sorted(cfg['apply_sides']))}\n"
    )


def save_best(best_path: Path, cfg: dict, m: dict):
    payload = {
        "config": _ser(cfg),
        "metrics": {k: v for k, v in m.items()},
    }
    best_path.write_text(json.dumps(payload, indent=2))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Hill-climbing sweep over early-exit threshold combinations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 loop.py
      Default: 300 trials, 40 random seeds, all DTEs and sides

  python3 loop.py --trials 600 --seeds 80
      Larger search budget

  python3 loop.py --dtes 1,2,3 --sides put
      Restrict to multi-day put-only rules

  python3 loop.py --out-prefix puts_only --dtes 1,2,3 --sides put
      Save separate results for puts-only search
        """,
    )
    p.add_argument("--trials",      type=int, default=300,
                   help="Total trials (random + hill-climb). Default: 300")
    p.add_argument("--seeds",       type=int, default=40,
                   help="Random seed configs in Phase 1. Default: 40")
    p.add_argument("--seed",        type=int, default=42,
                   help="RNG seed for reproducibility. Default: 42")
    p.add_argument("--stagnation",  type=int, default=150,
                   help="Stop hill-climb after N no-improvement steps. Default: 150")
    p.add_argument("--out-prefix",  default="v1",
                   help="Output file prefix. Default: v1")
    p.add_argument("--dtes",        default="0,1,2,3",
                   help="DTEs to include in search space. Default: 0,1,2,3")
    p.add_argument("--sides",       default="put,call",
                   help="Sides to include in search space. Default: put,call")
    p.add_argument("--fix-profit-target", type=int, default=None,
                   help="Lock profit_target_pct to this value (disable as free variable). "
                        "Use 9999 to effectively disable the profit target.")
    p.add_argument("--fix-stop-loss", type=int, default=None,
                   help="Lock stop_loss_pct to this value (disable as free variable). "
                        "Use 9999 to effectively disable the stop loss.")
    args = p.parse_args()

    allowed_dtes  = sorted(int(x) for x in args.dtes.split(","))
    allowed_sides = sorted(s.strip() for s in args.sides.split(","))
    fixed_pt = args.fix_profit_target
    fixed_sl = args.fix_stop_loss

    rng = random.Random(args.seed)
    results_path = ROOT / f"results_{args.out_prefix}.tsv"
    best_path    = ROOT / f"best_config_{args.out_prefix}.json"

    if not results_path.exists():
        results_path.write_text(
            "trial\thash\tstatus\top\tscore\tann_sharpe\thold_sharpe\t"
            "ann_roi\tmax_dd\twin_d\timprovement_pct\t"
            "profit_target\tstop_loss\tapply_dtes\tapply_tiers\tapply_sides\n"
        )

    best_cfg    = None
    best_score  = -999.0
    best_metrics = None
    trial = 0

    fix_desc = []
    if fixed_pt is not None: fix_desc.append(f"profit_target={fixed_pt}")
    if fixed_sl is not None: fix_desc.append(f"stop_loss={fixed_sl}")
    if fix_desc: print(f"Fixed axes: {', '.join(fix_desc)}")

    print(f"=== Phase 1: random seeds (n={args.seeds}) ===")
    with open(results_path, "a") as f:
        for i in range(args.seeds):
            cfg = random_config(rng, allowed_dtes, allowed_sides, fixed_pt, fixed_sl)
            m   = run_experiment(cfg)
            trial += 1
            h  = chash(cfg)
            st = "discard"
            if m.get("score", -999) > best_score:
                best_cfg    = cfg
                best_score  = m["score"]
                best_metrics = m
                st = "keep"
                save_best(best_path, cfg, m)
                print(
                    f"  {trial:04d} ★  score={m['score']:.4f}  "
                    f"sh={m.get('annualized_sharpe', 0):.2f}  "
                    f"(hold={m.get('hold_sharpe', 0):.2f})  "
                    f"improve={m.get('improvement_pct', 0):+.1f}%  "
                    f"pt={cfg['profit_target_pct']}  sl={cfg['stop_loss_pct']}  "
                    f"dtes={cfg['apply_dtes']}  tiers={cfg['apply_tiers']}  "
                    f"sides={cfg['apply_sides']}"
                )
            log_trial(f, trial, h, st, "rand", m, cfg)

    if best_cfg is None:
        best_cfg    = default_config()
        best_metrics = run_experiment(best_cfg)
        best_score  = best_metrics.get("score", -999)
        save_best(best_path, best_cfg, best_metrics)

    print(f"\n=== Phase 2: hill-climb to {args.trials} trials (stagnation={args.stagnation}) ===")
    stag = 0
    with open(results_path, "a") as f:
        while trial < args.trials and stag < args.stagnation:
            cfg, op = mutate(best_cfg, rng, allowed_dtes, allowed_sides, fixed_pt, fixed_sl)
            m       = run_experiment(cfg)
            trial  += 1
            h       = chash(cfg)
            if m.get("score", -999) > best_score:
                best_cfg    = cfg
                best_score  = m["score"]
                best_metrics = m
                save_best(best_path, cfg, m)
                log_trial(f, trial, h, "keep", op, m, cfg)
                print(
                    f"  {trial:04d} ★  op={op:22s}  score={m['score']:.4f}  "
                    f"sh={m.get('annualized_sharpe', 0):.2f}  "
                    f"improve={m.get('improvement_pct', 0):+.1f}%  "
                    f"pt={cfg['profit_target_pct']}  sl={cfg['stop_loss_pct']}  "
                    f"dtes={cfg['apply_dtes']}"
                )
                stag = 0
            else:
                log_trial(f, trial, h, "discard", op, m, cfg)
                stag += 1

    print(f"\n=== Done: {trial} trials, best score={best_score:.4f} ===")
    if best_metrics:
        print(f"  profit_target  = {best_cfg['profit_target_pct']}%")
        print(f"  stop_loss      = {best_cfg['stop_loss_pct']}  (9999=off)")
        print(f"  apply_dtes     = {best_cfg['apply_dtes']}")
        print(f"  apply_tiers    = {best_cfg['apply_tiers']}")
        print(f"  apply_sides    = {best_cfg['apply_sides']}")
        print(f"  ann_sharpe     = {best_metrics.get('annualized_sharpe', 0):.3f}")
        print(f"  hold_sharpe    = {best_metrics.get('hold_sharpe', 0):.3f}")
        print(f"  improvement    = {best_metrics.get('improvement_pct', 0):+.1f}%")
        print(f"  ann_roi        = {best_metrics.get('annualized_roi_pct', 0):.2f}%")
        print(f"  max_dd         = {best_metrics.get('max_drawdown_pct', 0):.2f}%")
        print(f"  win_days       = {best_metrics.get('win_day_pct', 0):.1f}%")
    print(f"\nResults TSV: {results_path}")
    print(f"Best config: {best_path}")


if __name__ == "__main__":
    main()
