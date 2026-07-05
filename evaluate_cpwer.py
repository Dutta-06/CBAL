import os
import json
import pandas as pd
import numpy as np
import itertools
from tqdm import tqdm

# --- CONFIGURATION ---
BASE_RTTM_DIR = "results"
FIXED_RTTM_DIR = "results"
REF_RTTM_DIR = "data/ami/rttm"  # Path to Ground Truth
TRANS_DIR = "results/transcripts"
OUTPUT_CSV = "cpwer_comparison_results.csv"

def calculate_levenshtein(ref_words, hyp_words):
    """Standard Levenshtein distance for WER."""
    r = ref_words.split()
    h = hyp_words.split()
    if not r: return len(h)
    
    # Memory-optimized row-based calculation
    current_row = range(len(r) + 1)
    for i in range(1, len(h) + 1):
        previous_row, current_row = current_row, [i] + [0] * len(r)
        for j in range(1, len(r) + 1):
            add, delete, change = previous_row[j] + 1, current_row[j-1] + 1, previous_row[j-1]
            if h[i-1] != r[j-1]: change += 1
            current_row[j] = min(add, delete, change)
    return current_row[len(r)]

def load_rttm_segments(path):
    segments = []
    if not os.path.exists(path): return []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8:
                segments.append({
                    'start': float(parts[3]), 
                    'end': float(parts[3]) + float(parts[4]), 
                    'speaker': parts[7]
                })
    return segments

def load_transcript_words(path):
    if not os.path.exists(path): return []
    with open(path, 'r') as f:
        data = json.load(f)
    words = []
    # Handle both list-of-segments and direct-word formats
    segs = data if isinstance(data, list) else data.get('segments', [])
    for segment in segs:
        if 'words' in segment:
            for w in segment['words']:
                words.append({
                    'text': w.get('word', '').strip().lower(), # Lowercase for WER
                    'start': float(w.get('start', 0)),
                    'end': float(w.get('end', 0))
                })
    return words

def get_speaker_text_map(segments, words):
    """Buckets words by speaker."""
    speaker_map = {}
    
    # Initialize empty text for all speakers found in segments
    all_speakers = set(s['speaker'] for s in segments)
    for s in all_speakers: speaker_map[s] = []

    for word in words:
        w_mid = (word['start'] + word['end']) / 2
        
        # Simple attribution: Who owns the midpoint?
        assigned = None
        for seg in segments:
            if seg['start'] <= w_mid <= seg['end']:
                assigned = seg['speaker']
                break
        
        if assigned:
            speaker_map[assigned].append(word['text'])
            
    # Join into strings
    return {k: " ".join(v) for k, v in speaker_map.items()}

def compute_cpwer(ref_map, hyp_map):
    """
    Calculates Concatenated Permutation WER (cpWER).
    Finds best 1:1 mapping between Reference and Hypothesis speakers.
    """
    ref_speakers = list(ref_map.keys())
    hyp_speakers = list(hyp_map.keys())
    
    # Pad with "dummy" speakers if counts don't match
    while len(ref_speakers) < len(hyp_speakers):
        ref_speakers.append(f"DUMMY_REF_{len(ref_speakers)}")
        ref_map[ref_speakers[-1]] = ""
    while len(hyp_speakers) < len(ref_speakers):
        hyp_speakers.append(f"DUMMY_HYP_{len(hyp_speakers)}")
        hyp_map[hyp_speakers[-1]] = ""
        
    # Generate all permutations of Hypothesis speakers
    min_total_dist = float('inf')
    total_ref_len = sum(len(s.split()) for s in ref_map.values())
    if total_ref_len == 0: return 1.0

    # For AMI (4 speakers), permutations are small (4! = 24). Brute force is safe.
    for perm in itertools.permutations(hyp_speakers):
        current_dist = 0
        
        # Pair Ref[i] with Perm[i]
        for i, r_spk in enumerate(ref_speakers):
            h_spk = perm[i]
            ref_text = ref_map.get(r_spk, "")
            hyp_text = hyp_map.get(h_spk, "")
            current_dist += calculate_levenshtein(ref_text, hyp_text)
            
        if current_dist < min_total_dist:
            min_total_dist = current_dist
            
    return min_total_dist / total_ref_len

def main():
    results = []
    # Identify common meetings
    files = [f for f in os.listdir(FIXED_RTTM_DIR) if f.startswith('fixed_') and f.endswith('.rttm')]
    mids = [f.replace('fixed_', '').replace('.rttm', '') for f in files]

    print(f"📊 Running cpWER comparison for {len(mids)} meetings...")
    print("(This calculates the Best-Match Permutation for each meeting)")

    for mid in tqdm(mids):
        # Paths
        ref_p = os.path.join(REF_RTTM_DIR, f"{mid}.rttm") # GROUND TRUTH
        base_p = os.path.join(BASE_RTTM_DIR, f"baseline_{mid}.rttm")
        fixed_p = os.path.join(FIXED_RTTM_DIR, f"fixed_{mid}.rttm")
        trans_p = os.path.join(TRANS_DIR, f"{mid}.json")
        
        if not all(os.path.exists(p) for p in [ref_p, base_p, fixed_p, trans_p]):
            continue
            
        words = load_transcript_words(trans_p)
        if not words: continue
        
        # 1. Build Reference Map (Ground Truth Speakers -> Words)
        ref_segs = load_rttm_segments(ref_p)
        ref_map = get_speaker_text_map(ref_segs, words)
        
        # 2. Build Baseline Map
        base_segs = load_rttm_segments(base_p)
        base_map = get_speaker_text_map(base_segs, words)
        
        # 3. Build Fixed (CBAL) Map
        fixed_segs = load_rttm_segments(fixed_p)
        fixed_map = get_speaker_text_map(fixed_segs, words)
        
        # 4. Compute cpWER
        base_cpwer = compute_cpwer(ref_map, base_map)
        fixed_cpwer = compute_cpwer(ref_map, fixed_map)
        
        results.append({
            "Meeting_ID": mid,
            "Words": len(words),
            "Baseline_cpWER": round(base_cpwer, 4),
            "CBAL_cpWER": round(fixed_cpwer, 4),
            "Improvement": round(base_cpwer - fixed_cpwer, 4)
        })

    if not results:
        print("❌ No valid files found. Check directories.")
        return

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n✅ Success! CSV saved to {OUTPUT_CSV}")
    print(df[['Meeting_ID', 'Baseline_cpWER', 'CBAL_cpWER', 'Improvement']].to_string(index=False))
    
    avg_imp = df['Improvement'].mean()
    print(f"\n📈 Average cpWER Improvement: {avg_imp:.2%}")

if __name__ == "__main__":
    main()