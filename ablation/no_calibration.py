"""
ABLATION 3: No Confidence Calibration
=======================================
Runs the full CBAL pipeline but skips the calibration step — raw LLM
confidence scores are used directly to gate merge decisions.

This shows the effect of the piecewise calibration function:
  c* = 0.0   if c < 0.6   (reject uncertain)
  c* = 0.9   if c >= 0.95 (cap extreme)
  c* = 0.9*c otherwise    (scale by 0.9)

Without calibration, the LLM's overconfident scores (often 0.95–1.0)
may cause more merges to pass the threshold, potentially reducing precision.

Compare against CBAL full:
  - Full CBAL (calibrated):   93.4% accuracy, 359 merges
  - This script (raw scores): expect more merges, lower precision

── BUG FIX (infrastructure only, no experimental change) ────────────────────
FIX: `from ablation_helper import ...` — the module name was spelled without
     a trailing 's', inconsistent with no_gapclear.py which uses
     `ablation_helpers`. This caused an ImportError before any experiment
     ran. Corrected to `ablation_helpers` to match the actual module name.

All thresholds, calibration logic, LLM settings, and experimental conditions
are strictly unchanged from the original.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import json
import gc
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import yaml
from pyannote.core import Annotation, Segment

from cbal.core.segmentation import load_rttm, write_rttm
from cbal.acoustic.embedding_extractor import EmbeddingExtractor
from cbal.linguistic.transcription import TranscriptLoader
from cbal.linguistic.context_builder import ContextBuilder
from cbal.llm.gemma_agent import GemmaAgent
from cbal.llm.prompt_builder import PromptBuilder
from cbal.repair.conflict_detector import ConflictDetector

# FIX: was `ablation_helper` (missing trailing 's') — caused ImportError on startup
from ablation_helpers import (load_reference_rttm, verify_fix,
                               load_transcript_words, is_gap_clear,
                               apply_merges)

BATCH_MEETINGS = [
    "ES2004a", "ES2004b", "ES2004c", "ES2004d",
    "IS1009a",  "IS1009b",  "IS1009c",  "IS1009d",
    "TS3003a",  "TS3003b",  "TS3003c",  "TS3003d",
]

# Use the same per-type confidence thresholds as full CBAL — UNCHANGED
CONFIDENCE_THRESHOLDS = {
    'false_split':        0.95,
    'acoustic_confusion': 0.96,
    'short_turn_check':   0.95,
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def passes_raw_confidence_gate(decision, error_type):
    """
    Gate using RAW confidence (no calibration).
    Mirrors full CBAL logic but without the calibration transform.
    """
    raw_conf  = decision.get('confidence', 0.0)
    threshold = CONFIDENCE_THRESHOLDS.get(error_type, 0.95)
    return raw_conf >= threshold


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    with open("configs/base_config.yaml") as f:
        cfg = yaml.safe_load(f)

    print("=" * 60)
    print("ABLATION 3: Full CBAL — Confidence Calibration DISABLED")
    print("  LLM reasoning:  ON")
    print("  Gap-Clear:      ON")
    print("  Calibration:    OFF  ← key change (raw scores used)")
    print(f"  Thresholds:     {CONFIDENCE_THRESHOLDS}")
    print("=" * 60)

    # Pass 1
    extractor  = EmbeddingExtractor(cfg['models']['wavlm_repo'])
    batch_data = {}

    for mid in BATCH_MEETINGS:
        rttm_path  = f"results/baseline_{mid}.rttm"
        trans_path = f"results/transcripts/{mid}.json"
        audio_path = os.path.join(cfg['paths']['base_dir'],
                                  cfg['paths']['audio_subdir'],
                                  f"{mid}.Mix-Headset.wav")
        ref_path   = f"data/ami/rttm/{mid}.rttm"
        if not os.path.exists(rttm_path):
            print(f"  Skipping {mid}")
            continue

        segments    = load_rttm(rttm_path)
        transcriber = TranscriptLoader(trans_path)
        detector    = ConflictDetector(extractor, transcriber, cfg['thresholds'])
        errors      = detector.scan(segments, audio_path)
        batch_data[mid] = {
            'errors':     errors,
            'segments':   segments,
            'trans_path': trans_path,
            'ref_path':   ref_path,
        }

    del extractor
    gc.collect()
    torch.cuda.empty_cache()

    # Pass 2: LLM with raw (uncalibrated) confidence gate
    agent    = GemmaAgent(cfg['models'])
    prompter = PromptBuilder()

    total_correct = total_incorrect = total_unknown = total_merges = 0
    calibration_would_have_blocked = 0
    meeting_results = []

    for mid, data in batch_data.items():
        errors    = data['errors']
        segments  = data['segments']
        raw_words = load_transcript_words(data['trans_path'])
        reference = load_reference_rttm(data['ref_path'], mid)

        transcriber = TranscriptLoader(data['trans_path'])
        ctx_builder = ContextBuilder(transcriber)

        decisions   = []
        correct = incorrect = unknown = cal_blocked = 0

        for err in tqdm(errors, desc=mid):
            ctx_str     = ctx_builder.build_context(segments, err['indices'])
            prompt_text = prompter.build(err)
            full_prompt = f"CONTEXT:\n{ctx_str}\n\n{prompt_text}"
            decision    = agent.predict(full_prompt)
            action      = decision.get('action', 'KEEP')
            raw_conf    = decision.get('confidence', 0.0)

            # Simulate calibration to count how many it would have changed.
            # Calibration function — UNCHANGED from calibration.py:
            #   >= 0.95 → 0.90
            #   <  0.60 → 0.0
            #   else    → raw * 0.9
            if raw_conf >= 0.95:
                calibrated = 0.90
            elif raw_conf < 0.6:
                calibrated = 0.0
            else:
                calibrated = raw_conf * 0.9

            threshold  = CONFIDENCE_THRESHOLDS.get(err['type'], 0.95)
            raw_passes = raw_conf   >= threshold
            cal_passes = calibrated >= threshold
            if raw_passes and not cal_passes:
                cal_blocked += 1

            # Apply raw confidence gate (no calibration) — UNCHANGED logic
            if action == 'MERGE' and passes_raw_confidence_gate(decision, err['type']):
                idx1, idx2 = err['indices']
                s1, s2 = segments[idx1], segments[idx2]
                if not is_gap_clear(s1.end, s2.start, raw_words):
                    decision['action'] = 'KEEP'
                else:
                    verdict = verify_fix(reference, s1, s2)
                    if   verdict == 'CORRECT':   correct   += 1
                    elif verdict == 'INCORRECT': incorrect += 1
                    else:                        unknown   += 1
            else:
                decision['action'] = 'KEEP'

            decisions.append(decision)

        fixed_segs, n_merges = apply_merges(segments, decisions, errors)
        out_path = f"results/ablation_no_calibration_{mid}.rttm"
        write_rttm(fixed_segs, out_path, mid)

        acc = correct / (correct + incorrect) if (correct + incorrect) > 0 else float('nan')
        print(f"  {mid}: {n_merges} merges | {correct}C {incorrect}I {unknown}U | "
              f"acc={acc:.1%} | cal_would_block={cal_blocked}")

        meeting_results.append({
            'meeting':                        mid,
            'merges':                         n_merges,
            'correct':                        correct,
            'incorrect':                      incorrect,
            'unknown':                        unknown,
            'calibration_would_have_blocked': cal_blocked,
        })
        total_merges                   += n_merges
        total_correct                  += correct
        total_incorrect                += incorrect
        total_unknown                  += unknown
        calibration_would_have_blocked += cal_blocked

    overall_acc = total_correct / (total_correct + total_incorrect) \
        if (total_correct + total_incorrect) > 0 else float('nan')

    print("\n" + "=" * 60)
    print("ABLATION 3 RESULTS  (No Calibration)")
    print(f"  Total merges applied            : {total_merges}")
    print(f"  Correct                         : {total_correct}")
    print(f"  Incorrect                       : {total_incorrect}")
    print(f"  Uncertain                        : {total_unknown}")
    print(f"  Fix Accuracy                    : {overall_acc:.1%}")
    print(f"  Extra merges calibration blocked: {calibration_would_have_blocked}")
    print(f"  LLM parse failure rate          : {agent.parse_failure_rate():.1%}")
    print(f"\n  CBAL (with calibration)         : 359 merges, 93.4%  ← compare")
    print("=" * 60)

    summary = {
        'ablation':                         'no_calibration',
        'calibration_enabled':              False,
        'confidence_thresholds':            CONFIDENCE_THRESHOLDS,
        'total_merges':                     total_merges,
        'correct':                          total_correct,
        'incorrect':                        total_incorrect,
        'unknown':                          total_unknown,
        'fix_accuracy':                     overall_acc,
        'extra_merges_calibration_blocked': calibration_would_have_blocked,
        'llm_parse_failure_rate':           agent.parse_failure_rate(),
        'per_meeting':                      meeting_results,
    }
    os.makedirs("results/ablations", exist_ok=True)
    with open("results/ablations/no_calibration_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("  Summary saved → results/ablations/no_calibration_summary.json")


if __name__ == "__main__":
    main()