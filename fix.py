import os
import yaml
import ast
from cbal.acoustic.embedding_extractor import EmbeddingExtractor
from cbal.linguistic.transcription import TranscriptLoader
from cbal.repair.conflict_detector import ConflictDetector
from pyannote.core import Annotation, Segment

# --- CONFIGURATION ---
MEETING_ID = "TS3003d"
CONFIG_PATH = "configs/base_config.yaml"

# --- SMART THRESHOLDS ---
# Gaps smaller than this are treated as "Jitter" and filled with speech.
# Gaps larger than this are treated as "Pauses" and kept as silence (bridged).
MAX_FILL_DURATION = 0.50  # 500ms is a standard linguistic pause threshold

# --- LOCAL DATA STRUCTURES ---
class SimpleSegment:
    def __init__(self, start, end, speaker):
        self.start = float(start)
        self.end = float(end)
        self.speaker = str(speaker)
        self.duration = self.end - self.start

    def to_rttm_line(self):
        dur = self.end - self.start
        return f"SPEAKER {MEETING_ID} 1 {self.start:.3f} {dur:.3f} <NA> <NA> {self.speaker} <NA>"

def load_rttm_as_list(path):
    segments = []
    if not os.path.exists(path): return []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8: continue
            try:
                start = float(parts[3])
                dur = float(parts[4])
                speaker = parts[7]
                segments.append(SimpleSegment(start, start+dur, speaker))
            except ValueError: continue
    segments.sort(key=lambda x: x.start)
    return segments

def main():
    print(f"🚀 Starting Hybrid Repair for {MEETING_ID}...")
    
    # 1. SETUP
    with open(CONFIG_PATH) as f: config = yaml.safe_load(f)
    audio_path = os.path.join(config['paths']['base_dir'], config['paths']['audio_subdir'], f"{MEETING_ID}.Mix-Headset.wav")
    baseline_path = f"results/baseline_{MEETING_ID}.rttm"
    log_path = f"results/logs/decisions_{MEETING_ID}.log"
    output_path = f"results/fixed_{MEETING_ID}.rttm" 

    # 2. LOAD
    print("🔵 Loading Baseline...")
    segments = load_rttm_as_list(baseline_path)
    if not segments:
        print("❌ Error: Could not load baseline RTTM.")
        return

    # 3. SCAN (To recover indices)
    print("🔵 Scanning indices...")
    transcriber = TranscriptLoader(f"results/transcripts/{MEETING_ID}.json")
    extractor = EmbeddingExtractor(config['models']['wavlm_repo'])
    detector = ConflictDetector(extractor, transcriber, config['thresholds'])
    errors = detector.scan(segments, audio_path) 
    
    # 4. READ LOG
    print("📖 Reading Decision Log...")
    decisions = []
    if os.path.exists(log_path):
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try: decisions.append(ast.literal_eval(line.strip()))
                    except: continue

    # 5. APPLY HYBRID FIXES
    print(f"🛠️ Applying Hybrid Fixes (Max Fill: {MAX_FILL_DURATION}s)...")
    indices_to_process = []
    
    for i, dec in enumerate(decisions):
        if i >= len(errors): break
        status = dec.get('status', '')
        if dec.get('action') == 'MERGE' and 'APPLIED' in status:
            indices_to_process.append(errors[i]['indices'])

    # Sort Descending to handle pops correctly
    indices_to_process.sort(key=lambda x: x[1], reverse=True)
    
    filled_count = 0
    bridged_count = 0
    
    for idx1, idx2 in indices_to_process:
        s1 = segments[idx1]
        s2 = segments[idx2]
        
        # Calculate Gap
        gap = s2.start - s1.end
        
        if gap <= MAX_FILL_DURATION:
            # STRATEGY A: HARD MERGE (Fill Silence)
            # Best for: Stutter, jitter, very short breaks.
            # Effect: Creates one continuous segment.
            s1.end = s2.end
            segments.pop(idx2)
            filled_count += 1
        else:
            # STRATEGY B: BRIDGE (Rename Only)
            # Best for: Long pauses where speaker takes a breath.
            # Effect: Keeps the silence gap (saving DER) but renames
            # the second segment to match the first (fixing identity).
            s2.speaker = s1.speaker
            # Do NOT pop s2.
            bridged_count += 1

    print(f"   ✅ Filled Gaps (Merge):   {filled_count}")
    print(f"   ✅ Bridged Gaps (Rename): {bridged_count}")
    print(f"   Total Repairs: {filled_count + bridged_count}")

    # 6. SAVE
    print(f"💾 Saving to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for s in segments:
            f.write(s.to_rttm_line() + "\n")

    # 7. EVALUATE
    print("\n🔍 Verifying DER Impact...")
    os.system("python evaluate_der.py")

if __name__ == "__main__":
    main()