"""
ABLATION 4b: Context Window Size — w=10
========================================
Runs the full CBAL pipeline with context window fixed at w=5.
~2–5 seconds of surrounding transcript context per decision.

Part of a three-way sweep: w ∈ {5, 10, 20} (w=20 is the CBAL default).
Run all three scripts independently and compare results.

All thresholds, calibration, and pipeline settings are identical to
full CBAL — only the context window passed to ContextBuilder changes.
"""

import os, sys, json, gc
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
from cbal.llm.calibration import Calibrator

WINDOW_SIZE = 10   # ← only thing that differs across the three scripts

BATCH_MEETINGS = [
    "ES2004a", "ES2004b", "ES2004c", "ES2004d",
    "IS1009a",  "IS1009b",  "IS1009c",  "IS1009d",
    "TS3003a",  "TS3003b",  "TS3003c",  "TS3003d",
]

# Per-type confidence thresholds — UNCHANGED from full CBAL
CONFIDENCE_THRESHOLDS = {
    'false_split':        0.95,
    'acoustic_confusion': 0.96,
    'short_turn_check':   0.95,
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_reference_rttm(path, mid):
    if not os.path.exists(path): return None
    ref = Annotation(uri=mid)
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8: continue
            start, dur = float(parts[3]), float(parts[4])
            ref[Segment(start, start + dur)] = parts[7]
    return ref


def verify_fix(reference, seg1, seg2):
    if reference is None: return "UNKNOWN"
    mid1 = seg1.start + (seg1.end - seg1.start) / 2
    mid2 = seg2.start + (seg2.end - seg2.start) / 2
    labels1 = reference.get_labels(Segment(mid1, mid1 + 0.1))
    labels2 = reference.get_labels(Segment(mid2, mid2 + 0.1))
    if not labels1 or not labels2: return "UNKNOWN"
    spk1, spk2 = list(labels1)[0], list(labels2)[0]
    if spk1 != spk2: return "INCORRECT"
    gap_start, gap_end = seg1.end, seg2.start
    gap_speakers = set()
    for seg, _, label in reference.itertracks():
        if seg.start < gap_end and seg.end > gap_start:
            gap_speakers.add(label)
    if gap_speakers and spk1 not in gap_speakers: return "INCORRECT"
    return "CORRECT"


def load_transcript_words(json_path):
    try:
        with open(json_path) as f: data = json.load(f)
        words = []
        segs = data if isinstance(data, list) else data.get('segments', [])
        for s in segs:
            for w in s.get('words', []):
                words.append({'start': float(w['start']), 'end': float(w['end'])})
        return words
    except:
        return []


def is_gap_clear(start, end, words):
    for w in words:
        mid = (w['start'] + w['end']) / 2
        if start < mid < end: return False
    return True


def apply_merges(segments, decisions, errors):
    indices_to_merge = []
    for i, dec in enumerate(decisions):
        if dec.get('action') == 'MERGE':
            indices_to_merge.append(errors[i]['indices'])
    indices_to_merge.sort(key=lambda x: x[1], reverse=True)
    fixed = segments.copy()
    count = 0
    for idx1, idx2 in indices_to_merge:
        if idx1 >= len(fixed) or idx2 >= len(fixed): continue
        fixed[idx1].end = fixed[idx2].end
        fixed.pop(idx2)
        count += 1
    return fixed, count


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    with open("configs/base_config.yaml") as f:
        cfg = yaml.safe_load(f)

    print("=" * 60)
    print(f"ABLATION 4b: Context Window Size  w={WINDOW_SIZE}")
    print("  LLM reasoning:  ON")
    print("  Gap-Clear:      ON")
    print("  Calibration:    ON")
    print(f"  Window size:    {WINDOW_SIZE}  (~2-5s of context)")
    print("=" * 60)

    # Pass 1: conflict detection
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
        raw_words   = load_transcript_words(trans_path)
        reference   = load_reference_rttm(ref_path, mid)

        batch_data[mid] = {
            'errors': errors, 'segments': segments,
            'trans_path': trans_path, 'ref_path': ref_path,
            'raw_words': raw_words, 'reference': reference,
        }

    del extractor
    gc.collect()
    torch.cuda.empty_cache()

    # Pass 2: LLM with fixed window size
    agent      = GemmaAgent(cfg['models'])
    prompter   = PromptBuilder()
    calibrator = Calibrator()

    total_correct = total_incorrect = total_unknown = total_merges = 0
    meeting_results = []

    for mid, data in batch_data.items():
        errors    = data['errors']
        segments  = data['segments']
        raw_words = data['raw_words']
        reference = data['reference']

        transcriber = TranscriptLoader(data['trans_path'])
        ctx_builder = ContextBuilder(transcriber)

        decisions = []
        correct = incorrect = unknown = 0

        for err in tqdm(errors, desc=f"  {mid} w={WINDOW_SIZE}"):
            ctx_str     = ctx_builder.build_context(segments, err['indices'],
                                                    window=WINDOW_SIZE)
            prompt_text = prompter.build(err)
            full_prompt = f"CONTEXT:\n{ctx_str}\n\n{prompt_text}"
            decision    = agent.predict(full_prompt)
            action      = decision.get('action', 'KEEP')

            if action == 'MERGE':
                raw_conf  = decision.get('confidence', 0.0)
                cal_conf  = calibrator.calibrate(raw_conf)
                threshold = CONFIDENCE_THRESHOLDS.get(err['type'], 0.95)

                if cal_conf < threshold:
                    decision['action'] = 'KEEP'
                else:
                    idx1, idx2 = err['indices']
                    s1, s2 = segments[idx1], segments[idx2]
                    if not is_gap_clear(s1.end, s2.start, raw_words):
                        decision['action'] = 'KEEP'
                    else:
                        verdict = verify_fix(reference, s1, s2)
                        if   verdict == 'CORRECT':   correct   += 1
                        elif verdict == 'INCORRECT': incorrect += 1
                        else:                        unknown   += 1

            decisions.append(decision)

        fixed_segs, n_merges = apply_merges(segments, decisions, errors)
        out_path = f"results/ablation_window{WINDOW_SIZE}_{mid}.rttm"
        write_rttm(fixed_segs, out_path, mid)

        acc = correct / (correct + incorrect) if (correct + incorrect) > 0 else float('nan')
        print(f"    {mid}: {n_merges} merges | {correct}C {incorrect}I {unknown}U | acc={acc:.1%}")

        meeting_results.append({
            'meeting': mid, 'window': WINDOW_SIZE, 'merges': n_merges,
            'correct': correct, 'incorrect': incorrect, 'unknown': unknown,
        })
        total_merges    += n_merges
        total_correct   += correct
        total_incorrect += incorrect
        total_unknown   += unknown

    overall_acc = total_correct / (total_correct + total_incorrect) \
        if (total_correct + total_incorrect) > 0 else float('nan')

    print("\n" + "=" * 60)
    print(f"ABLATION 4b RESULTS  (w={WINDOW_SIZE})")
    print(f"  Total merges : {total_merges}")
    print(f"  Correct      : {total_correct}")
    print(f"  Incorrect    : {total_incorrect}")
    print(f"  Uncertain    : {total_unknown}")
    print(f"  Fix Accuracy : {overall_acc:.1%}")
    print(f"  LLM failures : {agent.parse_failure_rate():.1%}")
    print("=" * 60)

    os.makedirs("results/ablations", exist_ok=True)
    out_json = f"results/ablations/context_window{WINDOW_SIZE}_summary.json"
    with open(out_json, "w") as f:
        json.dump({
            'ablation': 'context_window',
            'window': WINDOW_SIZE,
            'total_merges': total_merges,
            'correct': total_correct,
            'incorrect': total_incorrect,
            'unknown': total_unknown,
            'fix_accuracy': overall_acc,
            'llm_parse_failure_rate': agent.parse_failure_rate(),
            'per_meeting': meeting_results,
        }, f, indent=2)
    print(f"  Summary saved → {out_json}")


if __name__ == "__main__":
    main()