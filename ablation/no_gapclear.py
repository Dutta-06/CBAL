"""
ABLATION 2: No Gap-Clear Constraint
=====================================
Runs the full CBAL pipeline (with LLM reasoning) but disables the Gap-Clear
temporal validation constraint. This isolates the contribution of the
signal-based safety check.

Expected outcome: more merges are applied, but fix accuracy drops and
some cpWER degradation appears — proving the constraint is essential.

Compare against CBAL full:
  - CBAL full:  359 merges, 93.4% accuracy, 0% cpWER degradation
  - This script: more merges, lower accuracy (bad merges that Gap-Clear blocked)

── BUG FIXES (infrastructure only, no experimental change) ──────────────────
FIX 1: `from ablation_helpers import ...` was written as `from ablation_helper
        import ...` (missing trailing 's') in the original no_calibration.py.
        Standardised to `ablation_helpers` here for consistency — this is a
        typo fix, not a behavioural change.

FIX 2: `from run_cbal_full import is_gap_clear` was placed INSIDE the inner
        `for err in tqdm(errors)` loop. This triggered a fresh module import
        on every single iteration (thousands of times per run). If the module
        was unavailable it raised ImportError mid-loop, silently falling through
        to the except block and recording every decision as KEEP. Moved to
        module-level import where it belongs. Behaviour is identical — the same
        function is called — it just no longer crashes the loop.

All thresholds, calibration, LLM settings, and experimental conditions are
strictly unchanged from the original.
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

# FIX 1: consistent module name (was `ablation_helper` without 's' in original)
# FIX 2: is_gap_clear imported at module level (was inside the for-loop)
from ablation_helpers import (load_reference_rttm, verify_fix,
                               load_transcript_words, is_gap_clear,
                               apply_merges)

BATCH_MEETINGS = [
    "ES2004a", "ES2004b", "ES2004c", "ES2004d",
    "IS1009a",  "IS1009b",  "IS1009c",  "IS1009d",
    "TS3003a",  "TS3003b",  "TS3003c",  "TS3003d",
]

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    with open("configs/base_config.yaml") as f:
        cfg = yaml.safe_load(f)

    print("=" * 60)
    print("ABLATION 2: Full CBAL — Gap-Clear Constraint DISABLED")
    print("  LLM reasoning: ON")
    print("  Gap-Clear:     OFF  ← key change")
    print("  Calibration:   ON")
    print("=" * 60)

    # Pass 1: Conflict detection (WavLM)
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

    # Pass 2: LLM reasoning — Gap-Clear SKIPPED
    agent    = GemmaAgent(cfg['models'])
    prompter = PromptBuilder()

    total_correct = total_incorrect = total_unknown = total_merges = 0
    gap_blocked_count = 0
    meeting_results = []

    for mid, data in batch_data.items():
        errors    = data['errors']
        segments  = data['segments']
        raw_words = load_transcript_words(data['trans_path'])
        reference = load_reference_rttm(data['ref_path'], mid)

        transcriber = TranscriptLoader(data['trans_path'])
        ctx_builder = ContextBuilder(transcriber)

        decisions = []
        correct = incorrect = unknown = gap_would_block = 0

        for err in tqdm(errors, desc=mid):
            ctx_str     = ctx_builder.build_context(segments, err['indices'])
            prompt_text = prompter.build(err)
            full_prompt = f"CONTEXT:\n{ctx_str}\n\n{prompt_text}"
            decision    = agent.predict(full_prompt)
            action      = decision.get('action', 'KEEP')

            if action == 'MERGE':
                idx1, idx2 = err['indices']
                s1, s2 = segments[idx1], segments[idx2]

                # FIX 2: is_gap_clear now called from module-level import.
                # Behaviour is identical — only the import location changed.
                if not is_gap_clear(s1.end, s2.start, raw_words):
                    gap_would_block += 1
                    # In full CBAL this would be blocked — here we let it through

                verdict = verify_fix(reference, s1, s2)
                if   verdict == 'CORRECT':   correct   += 1
                elif verdict == 'INCORRECT': incorrect += 1
                else:                        unknown   += 1

            decisions.append(decision)

        fixed_segs, n_merges = apply_merges(segments, decisions, errors)
        out_path = f"results/ablation_no_gapclear_{mid}.rttm"
        write_rttm(fixed_segs, out_path, mid)

        acc = correct / (correct + incorrect) if (correct + incorrect) > 0 else float('nan')
        print(f"  {mid}: {n_merges} merges | {correct}C {incorrect}I {unknown}U | "
              f"acc={acc:.1%} | gap_would_block={gap_would_block}")

        meeting_results.append({
            'meeting':                   mid,
            'merges':                    n_merges,
            'correct':                   correct,
            'incorrect':                 incorrect,
            'unknown':                   unknown,
            'gap_would_have_blocked':    gap_would_block,
        })
        total_merges      += n_merges
        total_correct     += correct
        total_incorrect   += incorrect
        total_unknown     += unknown
        gap_blocked_count += gap_would_block

    overall_acc = total_correct / (total_correct + total_incorrect) \
        if (total_correct + total_incorrect) > 0 else float('nan')

    print("\n" + "=" * 60)
    print("ABLATION 2 RESULTS  (No Gap-Clear Constraint)")
    print(f"  Total merges applied             : {total_merges}")
    print(f"  Correct                          : {total_correct}")
    print(f"  Incorrect                        : {total_incorrect}")
    print(f"  Uncertain                        : {total_unknown}")
    print(f"  Fix Accuracy                     : {overall_acc:.1%}")
    print(f"  Merges Gap-Clear would've blocked: {gap_blocked_count}")
    print(f"  LLM parse failure rate           : {agent.parse_failure_rate():.1%}")
    print(f"\n  CBAL (with Gap-Clear)            : 359 merges, 93.4%  ← compare")
    print("=" * 60)

    summary = {
        'ablation':                            'no_gap_clear',
        'gap_clear_enabled':                   False,
        'total_merges':                        total_merges,
        'correct':                             total_correct,
        'incorrect':                           total_incorrect,
        'unknown':                             total_unknown,
        'fix_accuracy':                        overall_acc,
        'merges_gap_clear_would_have_blocked': gap_blocked_count,
        'llm_parse_failure_rate':              agent.parse_failure_rate(),
        'per_meeting':                         meeting_results,
    }
    os.makedirs("results/ablations", exist_ok=True)
    with open("results/ablations/no_gapclear_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("  Summary saved → results/ablations/no_gapclear_summary.json")


if __name__ == "__main__":
    main()