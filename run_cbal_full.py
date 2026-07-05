import os
import torch
import json
import sys
import traceback
import gc
from tqdm import tqdm
import logging
import yaml
import argparse
from datetime import datetime

# --- 1. PATH SETUP ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# FORCE DISABLE TRITON (Fixes Windows crashes)
os.environ["TORCH_COMPILE_DISABLE"] = "1"
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# --- IMPORTS ---
from cbal.core.segmentation import load_rttm, write_rttm
from cbal.acoustic.embedding_extractor import EmbeddingExtractor
from cbal.linguistic.transcription import TranscriptLoader
from cbal.linguistic.context_builder import ContextBuilder
from cbal.llm.gemma_agent import GemmaAgent
from cbal.llm.prompt_builder import PromptBuilder
from cbal.repair.conflict_detector import ConflictDetector
from cbal.utils.logging_utils import setup_logger
from pyannote.core import Annotation, Segment

BATCH_MEETINGS = [
    "ES2004a", "ES2004b", "ES2004c", "ES2004d",
    "IS1009a", "IS1009b", "IS1009c", "IS1009d",
    "TS3003a", "TS3003b", "TS3003c", "TS3003d"
]

# --- HELPER FUNCTIONS ---

def load_reference_rttm(path, mid):
    if not os.path.exists(path): return None
    ref = Annotation(uri=mid)
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8: continue
            start, dur = float(parts[3]), float(parts[4])
            ref[Segment(start, start+dur)] = parts[7]
    return ref

def verify_fix(reference, seg1, seg2):
    """
    Verify if merging seg1 and seg2 is correct according to ground truth.
    
    Checks:
    1. Are both segments from the same speaker in ground truth?
    2. Is there speech in the gap from a different speaker?
    
    Args:
        reference: Pyannote Annotation object with ground truth
        seg1: First baseline segment (has .start, .end, .speaker)
        seg2: Second baseline segment
        
    Returns:
        "CORRECT", "INCORRECT", or "UNKNOWN"
    """
    if reference is None:
        return "UNKNOWN"
    
    # Get midpoints of the two segments
    mid1 = seg1.start + (seg1.end - seg1.start) / 2
    mid2 = seg2.start + (seg2.end - seg2.start) / 2
    
    # Find who's speaking at each midpoint in ground truth
    labels1 = reference.get_labels(Segment(mid1, mid1 + 0.1))
    labels2 = reference.get_labels(Segment(mid2, mid2 + 0.1))
    
    if not labels1 or not labels2:
        return "UNKNOWN"  # No speech found in ground truth
    
    speaker1 = list(labels1)[0]
    speaker2 = list(labels2)[0]
    
    # Check if same speaker
    if speaker1 != speaker2:
        return "INCORRECT"  # Different speakers in truth
    
    # Check gap for other speakers
    gap_start = seg1.end
    gap_end = seg2.start
    
    # Get all speakers in the gap
    gap_speakers = set()
    for segment, track, label in reference.itertracks():
        if segment.start < gap_end and segment.end > gap_start:
            gap_speakers.add(label)
    
    # If gap has speech from someone other than speaker1
    if gap_speakers and speaker1 not in gap_speakers:
        return "INCORRECT"
    
    return "CORRECT"


# USAGE UPDATE in run_cbal_full.py around line 218:
# OLD: verdict = verify_fix(reference, gap_start, gap_end, s1.speaker)
# NEW: verdict = verify_fix(reference, s1, s2)

def load_transcript_words(json_path):
    try:
        with open(json_path, 'r') as f: data = json.load(f)
        words = []
        segs = data if isinstance(data, list) else data.get('segments', [])
        for s in segs:
            if 'words' in s:
                for w in s['words']:
                    words.append({
                        'start': float(w['start']), 
                        'end': float(w['end'])
                    })
        return words
    except: return []

def is_gap_clear(start, end, words):
    for w in words:
        mid = (w['start'] + w['end']) / 2
        if start < mid < end: return False
    return True

# --- INLINE MERGE ENGINE ---
def apply_merges_robust(segments, decisions, errors):
    """
    Directly modifies the segment list based on decisions.
    Sorts DESCENDING to prevent index shifting bugs.
    """
    indices_to_merge = []
    
    if len(decisions) != len(errors):
        print("Warning: Decision count mismatch. Skipping merges.")
        return segments, 0

    for i, dec in enumerate(decisions):
        if dec.get('action') == 'MERGE':
            idx_pair = errors[i]['indices']
            indices_to_merge.append(idx_pair)

    # Sort by the SECOND index in DESCENDING order
    indices_to_merge.sort(key=lambda x: x[1], reverse=True)

    fixed_segments = segments.copy()
    count = 0

    for idx1, idx2 in indices_to_merge:
        # Safety bounds
        if idx1 >= len(fixed_segments) or idx2 >= len(fixed_segments):
            continue

        s1 = fixed_segments[idx1]
        s2 = fixed_segments[idx2]

        # Modify S1 to extend to S2's end
        new_end = getattr(s2, 'end', s2['end'] if isinstance(s2, dict) else 0)
        
        if isinstance(s1, dict):
            s1['end'] = new_end
        else:
            s1.end = new_end 

        # Remove S2
        fixed_segments.pop(idx2)
        count += 1

    return fixed_segments, count

# --- MAIN PROCESS ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base_config.yaml")
    parser.add_argument("--meeting_id", default=None)
    args = parser.parse_args()
    
    logger = setup_logger()
    with open(args.config) as f: cfg = yaml.safe_load(f)
    
    targets = [args.meeting_id] if args.meeting_id else BATCH_MEETINGS
    
    # 1. SCANNING (WavLM)
    logger.info("PASS 1: Scanning...")
    extractor = EmbeddingExtractor(cfg['models']['wavlm_repo'])
    
    batch_data = {}
    for mid in targets:
        # Paths
        rttm_path = f"results/baseline_{mid}.rttm"
        trans_path = f"results/transcripts/{mid}.json"
        audio_path = os.path.join(cfg['paths']['base_dir'], cfg['paths']['audio_subdir'], f"{mid}.Mix-Headset.wav")
        ref_path = f"data/ami/rttm/{mid}.rttm" # Ground Truth

        if not os.path.exists(rttm_path): continue

        # Load
        segments = load_rttm(rttm_path)
        transcriber = TranscriptLoader(trans_path)
        detector = ConflictDetector(extractor, transcriber, cfg['thresholds'])
        
        # Scan
        errors = detector.scan(segments, audio_path)
        
        batch_data[mid] = {
            'errors': errors,
            'segments': segments,
            'trans_path': trans_path,
            'ref_path': ref_path
        }

    del extractor
    gc.collect()
    torch.cuda.empty_cache()

    # 2. REASONING (Gemma)
    logger.info("PASS 2: Reasoning...")
    agent = GemmaAgent(cfg['models'])
    prompter = PromptBuilder()
    
    for mid, data in batch_data.items():
        errors = data['errors']
        segments = data['segments']
        logger.info(f"Processing {mid}: {len(errors)} candidates")

        transcriber = TranscriptLoader(data['trans_path'])
        ctx_builder = ContextBuilder(transcriber)
        raw_words = load_transcript_words(data['trans_path'])
        reference = load_reference_rttm(data['ref_path'], mid)

        final_decisions = []
        log_entries = []

        # Decision Loop
        for i, err in enumerate(tqdm(errors)):
            log_entry = {'type': err['type'], 'text': err['text']}
            
            # Context & Prompt
            ctx_str = ctx_builder.build_context(segments, err['indices'])
            prompt_text = prompter.build(err)
            full_prompt = f"CONTEXT:\n{ctx_str}\n\n{prompt_text}"

            # Predict
            decision = agent.predict(full_prompt)
            action = decision.get('action', 'KEEP')
            
            log_entry['action'] = action
            
            if action == 'MERGE':
                # Verification Logic
                idx1, idx2 = err['indices']
                s1, s2 = segments[idx1], segments[idx2]
                gap_start, gap_end = s1.end, s2.start

                # 1. Gap Speech Check
                if not is_gap_clear(gap_start, gap_end, raw_words):
                    decision['action'] = 'KEEP' # Override
                    log_entry['status'] = "BLOCKED (Gap Speech)"
                else:
                    # 2. Ground Truth Check
                    verdict = verify_fix(reference, s1, s2)
                    log_entry['status'] = f"APPLIED ({verdict})"
            else:
                log_entry['status'] = "KEPT"

            final_decisions.append(decision)
            log_entries.append(log_entry)

        # --- SAVE DECISION LOG ---
        log_filename = f"results/logs/decisions_{mid}.log"
        os.makedirs(os.path.dirname(log_filename), exist_ok=True)
        
        # Use utf-8 to be safe, but content has no emojis now
        with open(log_filename, 'w', encoding='utf-8') as f:
            for le in log_entries:
                f.write(f"{le}\n")

        # --- APPLY & SAVE FIXES ---
        fixed_segments, count = apply_merges_robust(segments, final_decisions, errors)
        
        out_path = f"results/fixed_{mid}.rttm"
        write_rttm(fixed_segments, out_path, mid)
        logger.info(f"Saved {mid} | Fixes: {count}")

if __name__ == "__main__":
    main()