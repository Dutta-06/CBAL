"""
Ablation Results Aggregator
============================
After running all 4 ablation scripts, run this to produce a clean
comparison table suitable for the paper.

Usage:
    python ablation_results_table.py

Reads from:
    results/ablations/rule_only_summary.json
    results/ablations/no_gapclear_summary.json
    results/ablations/no_calibration_summary.json
    results/ablations/context_window_summary.json
"""

import json, os

ABLATION_DIR = "C:\\Users\\Odwitiyo\\Desktop\\CBAL\\results\\ablations"

# ── Full CBAL numbers (from paper Table 2 / Table 4) ─────────────────────────
CBAL_FULL = {
    'label':         'Full CBAL (LLM + Gap-Clear + Calibration, w=20)',
    'merges':        359,
    'correct':       268,
    'incorrect':     19,
    'fix_accuracy':  0.934,
}


def load(filename):
    path = os.path.join(ABLATION_DIR, filename)
    if not os.path.exists(path):
        print(f"  [!] Missing: {path} — run the corresponding ablation script first.")
        return None
    with open(path) as f:
        return json.load(f)


def fmt_acc(v):
    if v is None or isinstance(v, str): return "  N/A  "
    return f"{v:.1%}"


def print_table(rows):
    col_w = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    sep   = "-+-".join("-" * w for w in col_w)
    header = rows[0]
    print("  " + " | ".join(h.ljust(col_w[i]) for i, h in enumerate(header)))
    print("  " + sep)
    for row in rows[1:]:
        print("  " + " | ".join(str(v).ljust(col_w[i]) for i, v in enumerate(row)))


def main():
    print("\n" + "=" * 72)
    print("CBAL ABLATION STUDY — COMPONENT CONTRIBUTION ANALYSIS")
    print("=" * 72)

    rows = [
        ["Configuration", "Merges", "Correct", "Incorrect", "Fix Acc.", "vs. Full CBAL"],
    ]

    # ── Full CBAL ─────────────────────────────────────────────────────────────
    rows.append([
        "Full CBAL (LLM + GapClear + Calib, w=20)",
        str(CBAL_FULL['merges']),
        str(CBAL_FULL['correct']),
        str(CBAL_FULL['incorrect']),
        fmt_acc(CBAL_FULL['fix_accuracy']),
        "—  (baseline)",
    ])

    # ── Ablation 1: Rule-Only ─────────────────────────────────────────────────
    d = load("rule_only_summary.json")
    if d:
        acc  = d['fix_accuracy']
        diff = acc - CBAL_FULL['fix_accuracy']
        rows.append([
            f"Rule-Only (no LLM, gap<{d['rule_gap_threshold']}s / sim>{d['rule_sim_threshold']})",
            str(d['total_merges']),
            str(d['correct']),
            str(d['incorrect']),
            fmt_acc(acc),
            f"{diff:+.1%}",
        ])

    # ── Ablation 2: No Gap-Clear ──────────────────────────────────────────────
    d = load("no_gapclear_summary.json")
    if d:
        acc  = d['fix_accuracy']
        diff = acc - CBAL_FULL['fix_accuracy']
        rows.append([
            "LLM + No Gap-Clear + Calibration",
            str(d['total_merges']),
            str(d['correct']),
            str(d['incorrect']),
            fmt_acc(acc),
            f"{diff:+.1%}  (Gap-Clear blocks {d.get('merges_gap_clear_would_have_blocked','?')} bad merges)",
        ])

    # ── Ablation 3: No Calibration ────────────────────────────────────────────
    d = load("no_calibration_summary.json")
    if d:
        acc  = d['fix_accuracy']
        diff = acc - CBAL_FULL['fix_accuracy']
        rows.append([
            "LLM + Gap-Clear + No Calibration (raw conf.)",
            str(d['total_merges']),
            str(d['correct']),
            str(d['incorrect']),
            fmt_acc(acc),
            f"{diff:+.1%}  (calib. blocks {d.get('extra_merges_calibration_blocked','?')} extra)",
        ])

    # ── Ablation 4: Context Window Sweep ──────────────────────────────────────
    d = load("context_window_summary.json")
    if d:
        for r in d['results']:
            acc  = r['fix_accuracy']
            diff = acc - CBAL_FULL['fix_accuracy']
            marker = " ← default" if r['window'] == 20 else ""
            rows.append([
                f"LLM + GapClear + Calib, w={r['window']}{marker}",
                str(r['total_merges']),
                str(r['correct']),
                str(r['incorrect']),
                fmt_acc(acc),
                f"{diff:+.1%}",
            ])

    print_table(rows)

    print("\n" + "=" * 72)
    print("INTERPRETATION GUIDE")
    print("=" * 72)
    print("""
  Ablation 1 (Rule-Only):
    A large accuracy DROP here proves the LLM adds real value beyond
    simple threshold rules. A small drop means the LLM is mostly
    redundant — an important finding either way.

  Ablation 2 (No Gap-Clear):
    A DROP in accuracy or RISE in merges confirms the Gap-Clear
    constraint is catching harmful merges the LLM misses.
    The 'Gap-Clear blocks N' count shows its direct safety impact.

  Ablation 3 (No Calibration):
    More merges with lower accuracy → overconfidence is real and
    calibration is doing useful work. Similar numbers → calibration
    is conservative but not impactful.

  Ablation 4 (Context Window):
    Stable accuracy across w=5/10/20 → LLM is robust to context size.
    Accuracy drops at w=5 → long context is important for reasoning.
""")


if __name__ == "__main__":
    main()