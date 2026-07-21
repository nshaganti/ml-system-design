"""
build_report.py -- render a flat, self-contained HTML scoreboard from results.json.
==================================================================================
Same single source of truth as the markdown scoreboard (each phase's results.json),
but rendered as a shareable dashboard with charts. No server needed -- it's one
static file you can open in a browser or attach to a doc.

Usage:
    python scripts/build_report.py               # writes docs/report.html
    python scripts/build_report.py --open         # ... and opens it in a browser

Design: Tailwind (CDN) for layout, Chart.js (CDN) for charts. Every <canvas> lives
in a fixed-height wrapper div because Chart.js's responsive mode ignores the canvas
height attribute.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT = ROOT / "docs" / "report.html"


def load(phase: int) -> dict | None:
    path = ROOT / f"phase{phase}" / "results.json"
    return json.loads(path.read_text()) if path.exists() else None


def _card(title: str, body: str, note: str = "") -> str:
    note_html = f'<p class="text-sm text-slate-500 mt-3">{note}</p>' if note else ""
    return f"""
    <section class="bg-white rounded-2xl shadow-sm ring-1 ring-slate-200 p-6">
      <h2 class="text-lg font-semibold text-slate-800 mb-4">{title}</h2>
      {body}
      {note_html}
    </section>"""


def _chart(canvas_id: str, height: int = 260) -> str:
    # Fixed-height wrapper: Chart.js responsive mode ignores canvas height attr.
    return f'<div style="height:{height}px"><canvas id="{canvas_id}"></canvas></div>'


def build() -> str:
    p = {n: load(n) for n in range(15)}
    charts_js: list[str] = []
    cards: list[str] = []

    # --- Group A: Phase 0 vs Phase 1 ------------------------------------
    if p[0] and p[1]:
        cards.append(_card(
            "Group A - Retrieval: heuristic vs two-tower (Phase 0 -> 1)",
            _chart("chartA"),
            "Learned retrieval beats the heuristic on recall & coverage. Cold-start is "
            "a wash (both fall back to the same heuristic).",
        ))
        charts_js.append(f"""
        new Chart(document.getElementById('chartA'), {{
          type: 'bar',
          data: {{
            labels: ['Recall@20', 'Warm recall', 'Cold-start', 'Coverage'],
            datasets: [
              {{ label: 'Phase 0 (heuristic)', backgroundColor: '#94a3b8',
                 data: [{p[0]['recall_at_k']:.4f}, {p[0]['warm_user_recall']:.4f}, {p[0]['cold_start_recall']:.4f}, {p[0]['catalog_coverage']:.4f}] }},
              {{ label: 'Phase 1 (two-tower)', backgroundColor: '#2563eb',
                 data: [{p[1]['recall_at_k']:.4f}, {p[1]['warm_user_recall']:.4f}, {p[1]['cold_start_recall']:.4f}, {p[1]['catalog_coverage']:.4f}] }}
            ]
          }},
          options: {{ maintainAspectRatio: false, plugins: {{ legend: {{ position: 'bottom' }} }} }}
        }});""")

    # --- Group B: reranking (Phase 2) -----------------------------------
    if p[2]:
        cards.append(_card(
            "Group B - Reranking a fixed pool (Phase 2)",
            _chart("chartB"),
            "The single weak cross-feature LR ranker LOSES to popularity order - "
            "a feature with no signal is dead weight. An honest negative result.",
        ))
        charts_js.append(f"""
        new Chart(document.getElementById('chartB'), {{
          type: 'bar',
          data: {{
            labels: ['Recall@20', 'NDCG@20'],
            datasets: [
              {{ label: 'Popularity order', backgroundColor: '#94a3b8',
                 data: [{p[2]['pop_recall']:.4f}, {p[2]['pop_ndcg']:.4f}] }},
              {{ label: 'LR ranker', backgroundColor: '#dc2626',
                 data: [{p[2]['lr_recall']:.4f}, {p[2]['lr_ndcg']:.4f}] }}
            ]
          }},
          options: {{ maintainAspectRatio: false, plugins: {{ legend: {{ position: 'bottom' }} }} }}
        }});""")

    # --- Group E: OPE (Phase 8) -----------------------------------------
    if p[8]:
        cards.append(_card(
            "Group E - Off-policy evaluation (Phase 8)",
            _chart("chartE"),
            "The naive offline metric overstates true value by ~2x. SNIPS on the "
            "random log recovers the truth.",
        ))
        charts_js.append(f"""
        new Chart(document.getElementById('chartE'), {{
          type: 'bar',
          data: {{
            labels: ['Ground truth', 'Naive / DM', 'IPS', 'SNIPS', 'Doubly Robust'],
            datasets: [{{ label: 'Estimated policy value',
              backgroundColor: ['#16a34a', '#dc2626', '#f59e0b', '#2563eb', '#7c3aed'],
              data: [{p[8]['v_true']:.4f}, {p[8]['v_naive']:.4f}, {p[8]['ips']:.4f}, {p[8]['snips']:.4f}, {p[8]['doubly_robust']:.4f}] }}]
          }},
          options: {{ maintainAspectRatio: false, plugins: {{ legend: {{ display: false }} }} }}
        }});""")

    # --- Group F: off-policy learning (Phase 9) -------------------------
    if p[9]:
        sweep = p[9]["sweep"]
        cards.append(_card(
            "Group F - Off-policy learning (Phase 9)",
            _chart("chartF") + '<div class="mt-6"></div>' + _chart("chartFsweep"),
            "Left: learning from unbiased data nearly doubles true policy value. "
            "Right: bias doesn't average out - it takes ~100k unbiased rows to beat "
            "a policy fit on 1.44M biased rows (dashed = naive baseline).",
        ))
        charts_js.append(f"""
        new Chart(document.getElementById('chartF'), {{
          type: 'bar',
          data: {{
            labels: ['uniform', 'naive (biased)', 'learned (random)', 'greedy skyline'],
            datasets: [{{ label: 'True value',
              backgroundColor: ['#94a3b8', '#dc2626', '#2563eb', '#16a34a'],
              data: [{p[9]['v_uniform']:.4f}, {p[9]['v_naive_policy']:.4f}, {p[9]['v_learned_policy']:.4f}, {p[9]['v_greedy_policy']:.4f}] }}]
          }},
          options: {{ maintainAspectRatio: false, plugins: {{ legend: {{ display: false }} }} }}
        }});""")
        labels = ", ".join(f"'{s['n']:,}'" for s in sweep)
        vals = ", ".join(f"{s['true_value']:.4f}" for s in sweep)
        naive = p[9]["v_naive_policy"]
        charts_js.append(f"""
        new Chart(document.getElementById('chartFsweep'), {{
          type: 'line',
          data: {{
            labels: [{labels}],
            datasets: [
              {{ label: 'learned-policy true value', borderColor: '#2563eb',
                 backgroundColor: '#2563eb', data: [{vals}], tension: 0.2 }},
              {{ label: 'naive baseline', borderColor: '#dc2626', borderDash: [6,4],
                 pointRadius: 0, data: [{", ".join(f"{naive:.4f}" for _ in sweep)}] }}
            ]
          }},
          options: {{ maintainAspectRatio: false,
            plugins: {{ legend: {{ position: 'bottom' }} }},
            scales: {{ x: {{ title: {{ display: true, text: 'random-log rows used' }} }} }} }}
        }});""")

    # --- Group D/G: session next-item (Phase 7 + 10) -------------------
    if p[10]:
        cards.append(_card(
            "Group G - Session next-item: popularity vs co-visitation vs GRU4Rec (Phase 10)",
            _chart("chartG"),
            "Honest surprise: the GRU4Rec sequence model beats popularity but LOSES to "
            "cheap co-visitation on this small catalog. Fancier isn't automatically better.",
        ))
        pop, cov, gru = p[10]["popularity"], p[10]["covisitation"], p[10]["gru4rec"]
        charts_js.append(f"""
        new Chart(document.getElementById('chartG'), {{
          type: 'bar',
          data: {{
            labels: ['Recall@20', 'MRR@20', 'NDCG@20'],
            datasets: [
              {{ label: 'Popularity', backgroundColor: '#94a3b8',
                 data: [{pop['recall']:.4f}, {pop['mrr']:.4f}, {pop['ndcg']:.4f}] }},
              {{ label: 'Co-visitation', backgroundColor: '#16a34a',
                 data: [{cov['recall']:.4f}, {cov['mrr']:.4f}, {cov['ndcg']:.4f}] }},
              {{ label: 'GRU4Rec', backgroundColor: '#2563eb',
                 data: [{gru['recall']:.4f}, {gru['mrr']:.4f}, {gru['ndcg']:.4f}] }}
            ]
          }},
          options: {{ maintainAspectRatio: false, plugins: {{ legend: {{ position: 'bottom' }} }} }}
        }});""")

    # --- Group H: position-bias debiasing (Phase 11) -------------------
    if p[11]:
        cards.append(_card(
            "Group H - Position-bias debiasing (Phase 11, simulation)",
            _chart("chartH"),
            "Naive CTR ranks positions as much as items. IPW divides out the slot "
            "effect and recovers the true ranking - even with the examination curve "
            "estimated from a randomization bucket. (Controlled sim: Pure logs no position.)",
        ))
        n, t, e = p[11]["naive"], p[11]["ipw_true"], p[11]["ipw_estimated"]
        charts_js.append(f"""
        new Chart(document.getElementById('chartH'), {{
          type: 'bar',
          data: {{
            labels: ['Spearman vs truth', 'Top-10 recovery'],
            datasets: [
              {{ label: 'Naive CTR', backgroundColor: '#dc2626',
                 data: [{n['spearman']:.3f}, {n['topk_recovery']:.3f}] }},
              {{ label: 'IPW (true propensity)', backgroundColor: '#16a34a',
                 data: [{t['spearman']:.3f}, {t['topk_recovery']:.3f}] }},
              {{ label: 'IPW (estimated propensity)', backgroundColor: '#2563eb',
                 data: [{e['spearman']:.3f}, {e['topk_recovery']:.3f}] }}
            ]
          }},
          options: {{ maintainAspectRatio: false, plugins: {{ legend: {{ position: 'bottom' }} }},
            scales: {{ y: {{ min: 0, max: 1 }} }} }}
        }});""")

    # --- Group I: two-stage integration (Phase 12) ---------------------
    if p[12]:
        cards.append(_card(
            "Group I - Two-stage integration: retrieve + rank (Phase 12)",
            _chart("chartI"),
            "Honest architecture result: two-tower -> LR rerank LOSES to two-tower "
            "alone (popularity-flavored ranker undoes personalization) but beats "
            "popularity -> LR. A pipeline is only as good as the signal stage 2 adds.",
        ))
        tt, pl_, ts = p[12]["two_tower"], p[12]["pop_lr"], p[12]["two_stage"]
        charts_js.append(f"""
        new Chart(document.getElementById('chartI'), {{
          type: 'bar',
          data: {{
            labels: ['Recall@20', 'NDCG@20', 'Coverage'],
            datasets: [
              {{ label: 'Two-tower alone', backgroundColor: '#16a34a',
                 data: [{tt['recall']:.4f}, {tt['ndcg']:.4f}, {tt['coverage']:.4f}] }},
              {{ label: 'Popularity -> LR', backgroundColor: '#94a3b8',
                 data: [{pl_['recall']:.4f}, {pl_['ndcg']:.4f}, {pl_['coverage']:.4f}] }},
              {{ label: 'Two-tower -> LR', backgroundColor: '#2563eb',
                 data: [{ts['recall']:.4f}, {ts['ndcg']:.4f}, {ts['coverage']:.4f}] }}
            ]
          }},
          options: {{ maintainAspectRatio: false, plugins: {{ legend: {{ position: 'bottom' }} }} }}
        }});""")

    # --- Group J: stage-2 earns its place (Phase 13) -------------------
    if p[13]:
        cards.append(_card(
            "Group J - Stage 2 earns its place: two-tower score as a feature (Phase 13)",
            _chart("chartJ"),
            "The fix for Group I: feeding the retrieval score into the LR flips the "
            "-19% regression into a +3.3% win over two-tower alone. Two stages beat "
            "one only once stage 2 can see what stage 1 knows.",
        ))
        tt, sp, st = p[13]["two_tower"], p[13]["two_stage_pop"], p[13]["two_stage_tt"]
        charts_js.append(f"""
        new Chart(document.getElementById('chartJ'), {{
          type: 'bar',
          data: {{
            labels: ['Recall@20', 'NDCG@20', 'Coverage'],
            datasets: [
              {{ label: 'Two-tower alone', backgroundColor: '#94a3b8',
                 data: [{tt['recall']:.4f}, {tt['ndcg']:.4f}, {tt['coverage']:.4f}] }},
              {{ label: 'TT -> LR (pop feats)', backgroundColor: '#dc2626',
                 data: [{sp['recall']:.4f}, {sp['ndcg']:.4f}, {sp['coverage']:.4f}] }},
              {{ label: 'TT -> LR + tt_score', backgroundColor: '#16a34a',
                 data: [{st['recall']:.4f}, {st['ndcg']:.4f}, {st['coverage']:.4f}] }}
            ]
          }},
          options: {{ maintainAspectRatio: false, plugins: {{ legend: {{ position: 'bottom' }} }} }}
        }});""")

    # --- Group K: explore-and-learn loop (Phase 14) --------------------
    if p[14]:
        cards.append(_card(
            "Group K - Explore-and-learn: where unbiased data comes from (Phase 14)",
            _chart("chartK"),
            "Mean cumulative regret over 50k rounds (60 seeded worlds). Thompson's "
            "curve FLATTENS (it converges and stops paying) while greedy/epsilon stay "
            "linear. Exploration is the online source of the unbiased data Part II needs.",
        ))
        cx = p[14]["_curve_x"]
        cv = p[14]["_regret_curves"]
        charts_js.append(f"""
        new Chart(document.getElementById('chartK'), {{
          type: 'line',
          data: {{
            labels: {[int(x) for x in cx]},
            datasets: [
              {{ label: 'Greedy', borderColor: '#dc2626', backgroundColor: '#dc2626',
                 data: {cv['greedy']}, pointRadius: 0, borderWidth: 2 }},
              {{ label: 'Epsilon-greedy', borderColor: '#f59e0b', backgroundColor: '#f59e0b',
                 data: {cv['epsilon_greedy']}, pointRadius: 0, borderWidth: 2 }},
              {{ label: 'Thompson', borderColor: '#16a34a', backgroundColor: '#16a34a',
                 data: {cv['thompson']}, pointRadius: 0, borderWidth: 2 }}
            ]
          }},
          options: {{ maintainAspectRatio: false, plugins: {{ legend: {{ position: 'bottom' }} }},
            scales: {{ x: {{ title: {{ display: true, text: 'round' }} }},
                       y: {{ title: {{ display: true, text: 'cumulative regret' }} }} }} }}
        }});""")

    # --- serving + gate summary tiles (Phase 3, 4) ----------------------
    tiles = []
    if p[3]:
        tiles.append(("Serving p50 latency", f"{p[3]['p50_ms']:.1f} ms",
                      f"p99 {p[3]['p99_ms']:.1f} ms | {p[3]['pct_within_budget']:.0f}% within {p[3]['budget_ms']:.0f}ms budget"))
    if p[4]:
        tiles.append(("Pipeline gate", p[4]["gate_status"],
                      f"drift gate (row-count + calibration ECE {p[4]['calibration_ece']:.3f})"))
    if p[5]:
        sig = "significant" if p[5]["significant"] else "not significant"
        tiles.append(("A/B test (Phase 5)", f"{p[5]['relative_lift'] * 100:+.1f}%",
                      f"popularity vs LR, p={p[5]['p_value']:.3f} ({sig})"))
    if p[6]:
        sig = "significant" if p[6]["significant"] else "not significant"
        tiles.append(("Freshness (Phase 6)", f"{p[6]['relative_lift'] * 100:+.1f}%",
                      f"frozen vs fresh, p={p[6]['p_value']:.3f} ({sig})"))
    if p[7]:
        cov, pop = p[7]["covisitation"], p[7]["popularity"]
        lift = (cov["recall"] / pop["recall"] - 1) * 100 if pop["recall"] else 0
        tiles.append(("Co-visitation (Phase 7)", f"{lift:+.0f}%",
                      "session next-item recall vs popularity"))

    tiles_html = "".join(
        f"""<div class="bg-white rounded-2xl shadow-sm ring-1 ring-slate-200 p-5">
              <p class="text-xs uppercase tracking-wide text-slate-500">{t}</p>
              <p class="text-2xl font-bold text-slate-800 mt-1">{v}</p>
              <p class="text-sm text-slate-500 mt-1">{note}</p>
            </div>""" for t, v, note in tiles
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ML System Design - Scoreboard</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
</head>
<body class="bg-slate-100 text-slate-900">
  <div class="max-w-5xl mx-auto px-4 py-10 space-y-6">
    <header>
      <h1 class="text-3xl font-bold text-slate-900">ML System Design - Scoreboard</h1>
      <p class="text-slate-600 mt-2">Real results on KuaiRand-Pure (seed 42, 80/20 temporal split).
      Auto-generated from each phase's <code>results.json</code> - the same source of
      truth as <code>docs/results.md</code>. Regenerate with
      <code>python scripts/build_report.py</code>.</p>
    </header>

    <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
      {tiles_html}
    </div>

    {"".join(cards)}

    <footer class="text-center text-sm text-slate-500 pt-4">
      Part I (Phases 0-7, +10, 12-13): build a recommender with production discipline.
      Part II (Phases 8-9, 11, 14): prove the metric was biased, fix the learning,
      debias positions, and explore to mint unbiased data at the source.
    </footer>
  </div>
  <script>
    {"".join(charts_js)}
  </script>
</body>
</html>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--open", action="store_true", help="open the report after writing")
    args = ap.parse_args()

    OUT.write_text(build())
    print(f"Wrote {OUT.relative_to(ROOT)}")

    if args.open:
        import webbrowser
        webbrowser.open(OUT.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
