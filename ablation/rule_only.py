"""
ABLATION 1: Rule-Only Baseline (No LLM)  [FIXED verify_fix]
============================================================
Uses crop() instead of get_labels() for ground truth lookup.
"""

import os, sys, json, gc
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

import yaml
from pyannote.core import Annotation, Segment

from cbal.core.segmentation import load_rttm, write_rttm
from cbal.acoustic.embedding_extractor import EmbeddingExtractor
from cbal.linguistic.transcription import TranscriptLoader
from cbal.repair.conflict_detector import ConflictDetector
from ablation_helpers import (load_reference_rttm, verify_fix,
                               load_transcript_words, is_gap_clear,
                               apply_merges)

RULE_GAP_THRESHOLD = 0.5
RULE_SIM_THRESHOLD = 0.90

BATCH_MEETINGS = [
    "ES2004a", "ES2004b", "ES2004c", "ES2004d",
    "IS1009a",  "IS1009b",  "IS1009c",  "IS1009d",
    "TS3003a",  "TS3003b",  "TS3003c",  "TS3003d",
]


def rule_decision(error):
    err_type = error['type']
    if err_type == 'false_split':
        gap = error.get('gap', 999)
        if gap < RULE_GAP_THRESHOLD:
            return {'action': 'MERGE', 'confidence': 1.0, 'reasoning': f'Rule: gap {gap:.2f}s < {RULE_GAP_THRESHOLD}s'}
        return {'action': 'KEEP', 'confidence': 1.0, 'reasoning': f'Rule: gap {gap:.2f}s >= {RULE_GAP_THRESHOLD}s'}
    elif err_type == 'acoustic_confusion':
        sim = error.get('similarity', 0.0)
        if sim > RULE_SIM_THRESHOLD:
            return {'action': 'MERGE', 'confidence': 1.0, 'reasoning': f'Rule: sim {sim:.3f} > {RULE_SIM_THRESHOLD}'}
        return {'action': 'KEEP', 'confidence': 1.0, 'reasoning': f'Rule: sim {sim:.3f} <= {RULE_SIM_THRESHOLD}'}
    else:
        return {'action': 'KEEP', 'confidence': 1.0, 'reasoning': 'Rule: short turns always KEEP'}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    with open("configs/base_config.yaml") as f:
        cfg = yaml.safe_load(f)

    print("=" * 60)
    print("ABLATION 1: Rule-Only Baseline (No LLM)")
    print(f"  false_split  → MERGE if gap  < {RULE_GAP_THRESHOLD}s")
    print(f"  acous. conf  → MERGE if sim  > {RULE_SIM_THRESHOLD}")
    print(f"  short_turn   → always KEEP")
    print("=" * 60)

    extractor = EmbeddingExtractor(cfg['models']['wavlm_repo'])
    batch_data = {}

    for mid in BATCH_MEETINGS:
        rttm_path  = f"results/baseline_{mid}.rttm"
        trans_path = f"results/transcripts/{mid}.json"
        audio_path = os.path.join(cfg['paths']['base_dir'],
                                  cfg['paths']['audio_subdir'],
                                  f"{mid}.Mix-Headset.wav")
        ref_path   = f"data/ami/rttm/{mid}.rttm"

        if not os.path.exists(rttm_path):
            print(f"  Skipping {mid} — no baseline RTTM")
            continue

        segments    = load_rttm(rttm_path)
        transcriber = TranscriptLoader(trans_path)
        detector    = ConflictDetector(extractor, transcriber, cfg['thresholds'])
        errors      = detector.scan(segments, audio_path)

        batch_data[mid] = {
            'errors': errors, 'segments': segments,
            'trans_path': trans_path, 'ref_path': ref_path,
        }
        print(f"  {mid}: {len(errors)} candidates detected")

    del extractor
    gc.collect()

    total_correct = total_incorrect = total_unknown = total_merges = 0
    meeting_results = []

    for mid, data in batch_data.items():
        errors    = data['errors']
        segments  = data['segments']
        raw_words = load_transcript_words(data['trans_path'])
        reference = load_reference_rttm(data['ref_path'], mid)

        decisions = []
        correct = incorrect = unknown = 0

        for err in tqdm(errors, desc=mid):
            dec = rule_decision(err)

            if dec['action'] == 'MERGE':
                idx1, idx2 = err['indices']
                s1, s2 = segments[idx1], segments[idx2]

                if not is_gap_clear(s1.end, s2.start, raw_words):
                    dec['action'] = 'KEEP'
                else:
                    verdict = verify_fix(reference, s1, s2)
                    if   verdict == 'CORRECT':   correct   += 1
                    elif verdict == 'INCORRECT': incorrect += 1
                    else:                        unknown   += 1

            decisions.append(dec)

        fixed_segs, n_merges = apply_merges(segments, decisions, errors)
        out_path = f"results/ablation_rule_only_{mid}.rttm"
        write_rttm(fixed_segs, out_path, mid)

        acc = correct / (correct + incorrect) if (correct + incorrect) > 0 else float('nan')
        print(f"  {mid}: {n_merges} merges | {correct}C {incorrect}I {unknown}U | acc={acc:.1%}")

        meeting_results.append({
            'meeting': mid, 'merges': n_merges,
            'correct': correct, 'incorrect': incorrect, 'unknown': unknown,
        })
        total_merges  += n_merges
        total_correct += correct
        total_incorrect += incorrect
        total_unknown += unknown

    overall_acc = total_correct / (total_correct + total_incorrect) \
        if (total_correct + total_incorrect) > 0 else float('nan')

    print("\n" + "=" * 60)
    print("ABLATION 1 RESULTS  (Rule-Only, No LLM)")
    print(f"  Total merges applied : {total_merges}")
    print(f"  Correct              : {total_correct}")
    print(f"  Incorrect            : {total_incorrect}")
    print(f"  Uncertain            : {total_unknown}")
    print(f"  Fix Accuracy         : {overall_acc:.1%}")
    print(f"\n  CBAL (with LLM)      : 93.4%  ← compare here")
    print("=" * 60)

    summary = {
        'ablation': 'rule_only',
        'rule_gap_threshold': RULE_GAP_THRESHOLD,
        'rule_sim_threshold': RULE_SIM_THRESHOLD,
        'total_merges': total_merges,
        'correct': total_correct,
        'incorrect': total_incorrect,
        'unknown': total_unknown,
        'fix_accuracy': overall_acc,
        'per_meeting': meeting_results,
    }
    os.makedirs("results/ablations", exist_ok=True)
    with open("results/ablations/rule_only_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("  Summary saved → results/ablations/rule_only_summary.json")


if __name__ == "__main__":
    main()