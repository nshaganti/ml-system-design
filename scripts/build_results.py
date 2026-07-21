"""
build_results.py -- regenerate the scoreboard tables in docs/results.md.
========================================================================
The single source of truth for every number in the scoreboard is the
results.json each phase writes when its run.py executes. This script reads those
files and rewrites ONLY the table blocks marked with:

    <!-- AUTOGEN:groupX --> ... <!-- /AUTOGEN:groupX -->

The prose, verdicts, and caveats around the tables stay hand-written. This is the
DRY fix for the thing that bit us before: numbers hand-copied into docs that
silently drifted from what the code produced.

If a phase hasn't been run yet (no results.json), its block is left UNTOUCHED so
we never blow away good numbers with blanks.

Usage:
    python scripts/build_results.py            # rewrite docs/results.md in place
    python scripts/build_results.py --check     # exit 1 if it WOULD change (CI)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS_MD = ROOT / "docs" / "results.md"


# --------------------------------------------------------------- helpers

def load(phase: int) -> dict | None:
    path = ROOT / f"phase{phase}" / "results.json"
    return json.loads(path.read_text()) if path.exists() else None


def pct_change(old: float, new: float) -> str:
    if old == 0:
        return "n/a"
    return f"{(new / old - 1) * 100:+.0f}%"


def bold_max(a: float, b: float, fmt: str = "{:.4f}") -> tuple[str, str]:
    """Format two numbers, bolding whichever is larger (the 'winner')."""
    sa, sb = fmt.format(a), fmt.format(b)
    if a > b:
        return f"**{sa}**", sb
    if b > a:
        return sa, f"**{sb}**"
    return sa, sb


def replace_block(text: str, name: str, table: str) -> str:
    """Swap the content between the AUTOGEN markers for `name`."""
    pattern = re.compile(
        rf"(<!-- AUTOGEN:{name} -->\n).*?(\n<!-- /AUTOGEN:{name} -->)",
        re.DOTALL,
    )
    if not pattern.search(text):
        print(f"  [warn] no marker block found for '{name}' -- skipping")
        return text
    return pattern.sub(lambda m: m.group(1) + table + m.group(2), text)


# --------------------------------------------------------------- table builders

def group_a(p0: dict, p1: dict) -> str:
    rows = [
        ("Recall@20", "recall_at_k"),
        ("Warm-user recall", "warm_user_recall"),
        ("Cold-start recall", "cold_start_recall"),
        ("Catalog coverage", "catalog_coverage"),
    ]
    out = ["| Metric | Phase 0 (heuristic) | Phase 1 (two-tower) | Change |",
           "|---|---|---|---|"]
    for label, key in rows:
        a, b = bold_max(p0[key], p1[key])
        out.append(f"| {label} | {a} | {b} | {pct_change(p0[key], p1[key])} |")
    return "\n".join(out)


def group_b(p2: dict) -> str:
    out = ["| Metric | Popularity order | LR ranker | Change |", "|---|---|---|---|"]
    for label, pk, lk in [("Recall@20", "pop_recall", "lr_recall"),
                          ("NDCG@20", "pop_ndcg", "lr_ndcg")]:
        a, b = bold_max(p2[pk], p2[lk])
        out.append(f"| {label} | {a} | {b} | {pct_change(p2[pk], p2[lk])} |")
    return "\n".join(out)


def group_c(p5: dict | None, p6: dict | None) -> str:
    out = ["| Experiment | Arm A | Arm B | Lift | Significance |",
           "|---|---|---|---|---|"]
    if p5:
        sig = "**significant**" if p5["significant"] else "**not significant**"
        out.append(
            f"| **Phase 5:** popularity vs LR ranker | "
            f"**{p5['control_rate']:.4f} (popularity)** | {p5['treatment_rate']:.4f} (LR) | "
            f"{p5['relative_lift'] * 100:+.1f}% | p={p5['p_value']:.4f} -- {sig} |"
        )
    if p6:
        sig = "**significant**" if p6["significant"] else "**not significant**"
        out.append(
            f"| **Phase 6:** frozen vs fresh features | "
            f"{p6['control_rate']:.4f} (frozen batch) | {p6['treatment_rate']:.4f} (fresh stream) | "
            f"{p6['relative_lift'] * 100:+.1f}% | p={p6['p_value']:.3f} -- {sig} |"
        )
    return "\n".join(out)


def group_d(p7: dict) -> str:
    pop, cov = p7["popularity"], p7["covisitation"]
    out = ["| Metric | Popularity | Co-visitation | Lift |", "|---|---|---|---|"]
    for label, key in [("Recall@20", "recall"), ("MRR@20", "mrr"), ("NDCG@20", "ndcg")]:
        a, b = bold_max(pop[key], cov[key])
        out.append(f"| {label} | {a} | {b} | {pct_change(pop[key], cov[key])} |")
    return "\n".join(out)


def group_e(p8: dict) -> str:
    vt = p8["v_true"]

    def err(v: float) -> str:
        return f"{abs(v - vt) / vt * 100:.1f}%" if vt else "n/a"

    out = ["| Estimator | Value | Error vs truth |", "|---|---|---|",
           f"| **Ground truth** (\u03c0 \u00d7 random-log rewards) | {vt:.3f} | -- |",
           f"| Naive / Direct Method (biased log) | {p8['v_naive']:.3f} | **{err(p8['v_naive'])}** |",
           f"| IPS | {p8['ips']:.3f} | {err(p8['ips'])} |",
           f"| **SNIPS** | {p8['snips']:.3f} | **{err(p8['snips'])}** |",
           f"| Doubly Robust | {p8['doubly_robust']:.3f} | {err(p8['doubly_robust'])} |"]
    return "\n".join(out)


def group_f(p9: dict) -> str:
    naive = p9["v_naive_policy"]

    def rel(v: float) -> str:
        return f"{(v / naive - 1) * 100:+.0f}%" if naive else "n/a"

    return "\n".join([
        "| Policy (how it was learned) | True value | vs naive |",
        "|---|---|---|",
        f"| uniform random (no learning) | {p9['v_uniform']:.3f} | {rel(p9['v_uniform'])} |",
        f"| pi_naive = softmax(biased-log rates) | {naive:.3f} | -- |",
        f"| **pi_learned = softmax(random-log rates)** | "
        f"**{p9['v_learned_policy']:.3f}** | **{rel(p9['v_learned_policy'])}** |",
        f"| pi_greedy = argmax(random-log rates) | {p9['v_greedy_policy']:.3f} | {rel(p9['v_greedy_policy'])} |",
    ])


def group_g(p10: dict) -> str:
    pop, cov, gru = p10["popularity"], p10["covisitation"], p10["gru4rec"]
    out = ["| Metric | Popularity | Co-visitation | GRU4Rec |", "|---|---|---|---|"]
    for label, key in [("Recall@20", "recall"), ("MRR@20", "mrr"), ("NDCG@20", "ndcg")]:
        vals = {"pop": pop[key], "cov": cov[key], "gru": gru[key]}
        best = max(vals, key=vals.get)
        cells = {k: f"**{v:.4f}**" if k == best else f"{v:.4f}" for k, v in vals.items()}
        out.append(f"| {label} | {cells['pop']} | {cells['cov']} | {cells['gru']} |")
    return "\n".join(out)


def group_h(p11: dict) -> str:
    rows = [
        ("Naive CTR", p11["naive"]),
        ("IPW (true propensity)", p11["ipw_true"]),
        ("IPW (estimated propensity)", p11["ipw_estimated"]),
    ]
    best = max(rows, key=lambda r: r[1]["spearman"])[0]
    out = ["| Estimator | Spearman vs truth | Top-10 recovery |", "|---|---|---|"]
    for label, m in rows:
        sp = f"**{m['spearman']:.3f}**" if label == best else f"{m['spearman']:.3f}"
        out.append(f"| {label} | {sp} | {m['topk_recovery'] * 100:.0f}% |")
    return "\n".join(out)


def group_i(p12: dict) -> str:
    rows = [
        ("Two-tower alone (P1)", p12["two_tower"]),
        ("Popularity -> LR (P2)", p12["pop_lr"]),
        ("Two-tower -> LR (P12)", p12["two_stage"]),
    ]
    out = ["| Ranker | Recall@20 | NDCG@20 | Coverage |", "|---|---|---|---|"]
    # bold the best cell per column
    best_by = {m: max(rows, key=lambda r: r[1][m])[0] for m in ("recall", "ndcg", "coverage")}
    for label, m in rows:
        cells = []
        for metric in ("recall", "ndcg", "coverage"):
            v = f"{m[metric]:.4f}"
            cells.append(f"**{v}**" if label == best_by[metric] else v)
        out.append(f"| {label} | {cells[0]} | {cells[1]} | {cells[2]} |")
    return "\n".join(out)


def group_j(p13: dict) -> str:
    rows = [
        ("Two-tower alone", p13["two_tower"]),
        ("TT -> LR (pop feats)", p13["two_stage_pop"]),
        ("TT -> LR + tt_score", p13["two_stage_tt"]),
    ]
    out = ["| Ranker | Recall@20 | NDCG@20 | Coverage |", "|---|---|---|---|"]
    best_by = {m: max(rows, key=lambda r: r[1][m])[0] for m in ("recall", "ndcg", "coverage")}
    for label, m in rows:
        cells = []
        for metric in ("recall", "ndcg", "coverage"):
            v = f"{m[metric]:.4f}"
            cells.append(f"**{v}**" if label == best_by[metric] else v)
        out.append(f"| {label} | {cells[0]} | {cells[1]} | {cells[2]} |")
    return "\n".join(out)


def group_k(p14: dict) -> str:
    rows = [
        ("Greedy (exploit only)", p14["greedy"]),
        ("Epsilon-greedy", p14["epsilon_greedy"]),
        ("Thompson sampling", p14["thompson"]),
    ]
    # lower regret is better; higher is better for the rest
    best_regret = min(rows, key=lambda r: r[1]["final_regret"])[0]
    best = {m: max(rows, key=lambda r: r[1][m])[0]
            for m in ("best_arm_pct", "arm_support", "found_best_pct")}
    out = ["| Strategy | Mean regret | Best-arm % | Log support | Found best % |",
           "|---|---|---|---|---|"]
    for label, m in rows:
        reg = f"{m['final_regret']:.0f}"
        reg = f"**{reg}**" if label == best_regret else reg
        cells = []
        for metric in ("best_arm_pct", "arm_support", "found_best_pct"):
            v = f"{m[metric]*100:.0f}%"
            cells.append(f"**{v}**" if label == best[metric] else v)
        out.append(f"| {label} | {reg} | {cells[0]} | {cells[1]} | {cells[2]} |")
    return "\n".join(out)


def group_l(p15: dict) -> str:
    rows = [
        ("Context-free Thompson", p15["thompson"]),
        ("LinUCB (alpha=0, greedy)", p15["linucb_greedy"]),
        ("LinUCB (alpha=1)", p15["linucb"]),
    ]
    best_regret = min(rows, key=lambda r: r[1]["final_regret"])[0]
    best_pick = max(rows, key=lambda r: r[1]["best_arm_pct"])[0]
    out = ["| Policy | Mean regret | Per-user-best % |", "|---|---|---|"]
    for label, m in rows:
        reg = f"{m['final_regret']:.0f}"
        reg = f"**{reg}**" if label == best_regret else reg
        pick = f"{m['best_arm_pct']*100:.0f}%"
        pick = f"**{pick}**" if label == best_pick else pick
        out.append(f"| {label} | {reg} | {pick} |")
    return "\n".join(out)


def group_m(p16: dict) -> str:
    rows = [
        ("Uniform floor", p16["uniform"], 0.0),
        ("No exploration (trap)", p16["no_explore"]["final_value"], p16["no_explore"]["gap_closed"]),
        ("Explore, no IPS", p16["explore_no_ips"]["final_value"], p16["explore_no_ips"]["gap_closed"]),
        ("Closed loop (explore+IPS)", p16["closed_loop"]["final_value"], p16["closed_loop"]["gap_closed"]),
        ("Skyline (oracle)", p16["skyline"], 1.0),
    ]
    out = ["| Policy | True deployed value | % of skyline gap closed |", "|---|---|---|"]
    for label, val, gap in rows:
        out.append(f"| {label} | {val:.4f} | {gap*100:.0f}% |")
    return "\n".join(out)


# --------------------------------------------------------------- main

def build(text: str) -> str:
    p = {n: load(n) for n in range(17)}
    if p[0] and p[1]:
        text = replace_block(text, "groupA", group_a(p[0], p[1]))
    if p[2]:
        text = replace_block(text, "groupB", group_b(p[2]))
    if p[5] or p[6]:
        text = replace_block(text, "groupC", group_c(p[5], p[6]))
    if p[7]:
        text = replace_block(text, "groupD", group_d(p[7]))
    if p[8]:
        text = replace_block(text, "groupE", group_e(p[8]))
    if p[9]:
        text = replace_block(text, "groupF", group_f(p[9]))
    if p[10]:
        text = replace_block(text, "groupG", group_g(p[10]))
    if p[11]:
        text = replace_block(text, "groupH", group_h(p[11]))
    if p[12]:
        text = replace_block(text, "groupI", group_i(p[12]))
    if p[13]:
        text = replace_block(text, "groupJ", group_j(p[13]))
    if p[14]:
        text = replace_block(text, "groupK", group_k(p[14]))
    if p[15]:
        text = replace_block(text, "groupL", group_l(p[15]))
    if p[16]:
        text = replace_block(text, "groupM", group_m(p[16]))
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if results.md is out of date (do not write)")
    args = ap.parse_args()

    original = RESULTS_MD.read_text()
    updated = build(original)

    if args.check:
        if original != updated:
            print("docs/results.md is OUT OF DATE -- run: python scripts/build_results.py")
            return 1
        print("docs/results.md is up to date.")
        return 0

    if original == updated:
        print("docs/results.md already up to date (no results.json changed the tables).")
    else:
        RESULTS_MD.write_text(updated)
        print(f"Rewrote scoreboard tables in {RESULTS_MD.relative_to(ROOT)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
