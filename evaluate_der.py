import os
import csv
import pandas as pd
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate

# --- CONFIGURATION ---
OUTPUT_CSV = "evaluation_results.csv"

# Define the meetings you want to process
# Note: 'ES10024' looks like a typo for standard AMI file 'ES2004'. 
# I have used 'ES2004' below. Change it back if you strictly need 'ES10024'.
MEETING_PREFIXES = ["ES2004", "IS1009", "TS3003"] 
SUFFIXES = ["a", "b", "c", "d"]

def load_rttm_to_annotation(path, uri="meeting"):
    """Converts RTTM file to Pyannote Annotation object."""
    annotation = Annotation(uri=uri)
    try:
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 8: continue
                start = float(parts[3])
                duration = float(parts[4])
                speaker = parts[7]
                annotation[Segment(start, start + duration)] = speaker
    except FileNotFoundError:
        return None
    return annotation

def evaluate_single_meeting(meeting_id, ref_path, base_path, fixed_path):
    """Calculates metrics for a single meeting ID."""
    
    # 1. Check files exist
    if not os.path.exists(ref_path):
        print(f"⚠️ Missing Reference: {ref_path}")
        return None
    if not os.path.exists(base_path):
        print(f"⚠️ Missing Baseline: {base_path}")
        return None
    if not os.path.exists(fixed_path):
        print(f"⚠️ Missing Fixed: {fixed_path}")
        return None

    # 2. Load Annotations
    ref = load_rttm_to_annotation(ref_path, uri=meeting_id)
    base = load_rttm_to_annotation(base_path, uri=meeting_id)
    fixed = load_rttm_to_annotation(fixed_path, uri=meeting_id)

    # 3. Initialize Metric
    metric = DiarizationErrorRate(collar=0.0, skip_overlap=False)

    # 4. Calculate Baseline Stats
    base_report = metric(ref, base, detailed=True)
    base_der = base_report['diarization error rate']
    base_conf = base_report['confusion'] / base_report['total']
    base_miss = base_report['missed detection'] / base_report['total']
    base_fa = base_report['false alarm'] / base_report['total']

    # Reset metric
    metric.reset()

    # 5. Calculate Fixed Stats
    fixed_report = metric(ref, fixed, detailed=True)
    fixed_der = fixed_report['diarization error rate']
    fixed_conf = fixed_report['confusion'] / fixed_report['total']
    fixed_miss = fixed_report['missed detection'] / fixed_report['total']
    fixed_fa = fixed_report['false alarm'] / fixed_report['total']

    # 6. Calculate Delta
    abs_reduction = base_der - fixed_der
    rel_improvement = (abs_reduction / base_der) if base_der > 0 else 0.0

    print(f"✅ {meeting_id}: Base DER {base_der:.2%} -> Fixed DER {fixed_der:.2%} (Imp: {abs_reduction:.2%})")

    return {
        "Meeting_ID": meeting_id,
        "Base_DER": base_der,
        "Fixed_DER": fixed_der,
        "Abs_Reduction": abs_reduction,
        "Rel_Improvement": rel_improvement,
        "Base_Conf": base_conf,
        "Fixed_Conf": fixed_conf,
        "Base_Miss": base_miss,
        "Fixed_Miss": fixed_miss,
        "Base_FA": base_fa,
        "Fixed_FA": fixed_fa
    }

def main():
    all_results = []
    
    print(f"🚀 Starting Bulk Evaluation...")
    print(f"   Targets: {MEETING_PREFIXES}")
    print("-" * 60)

    # Generate full list of IDs (e.g., IS1009a, IS1009b...)
    meeting_ids = []
    for prefix in MEETING_PREFIXES:
        for suffix in SUFFIXES:
            meeting_ids.append(f"{prefix}{suffix}")

    # Process each meeting
    for mid in meeting_ids:
        # Define Paths
        ref_path = f"data/ami/rttm/{mid}.rttm"
        base_path = f"results/baseline_{mid}.rttm"
        fixed_path = f"results/fixed_{mid}.rttm"

        # Run Eval
        stats = evaluate_single_meeting(mid, ref_path, base_path, fixed_path)
        if stats:
            all_results.append(stats)

    # Save to CSV
    if all_results:
        df = pd.DataFrame(all_results)
        
        # Reorder columns for readability
        cols = ["Meeting_ID", "Base_DER", "Fixed_DER", "Abs_Reduction", "Rel_Improvement", 
                "Base_Conf", "Fixed_Conf", "Base_Miss", "Fixed_Miss"]
        df = df[cols]
        
        df.to_csv(OUTPUT_CSV, index=False)
        print("-" * 60)
        print(f"💾 Results saved to: {os.path.abspath(OUTPUT_CSV)}")
        
        # Print Averages
        print("\n📊 SUMMARY AVERAGES:")
        print(df[["Base_DER", "Fixed_DER", "Abs_Reduction"]].mean().apply(lambda x: f"{x:.2%}"))
    else:
        print("\n⚠️ No results to save. Check your file paths.")

if __name__ == "__main__":
    main()