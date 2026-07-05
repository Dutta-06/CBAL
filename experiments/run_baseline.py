import argparse
import os
from cbal.acoustic.baseline_runner import BaselineRunner
from cbal.core.segmentation import write_rttm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--hf_token", required=True)
    args = parser.parse_args()

    runner = BaselineRunner(args.hf_token)
    runner.load_pipeline()

    os.makedirs(args.output_dir, exist_ok=True)

    for file in os.listdir(args.audio_dir):
        if file.endswith(".wav"):
            path = os.path.join(args.audio_dir, file)
            meeting_id = file.split('.')[0]
            
            # Run
            segments_dict = runner.run(path)
            
            # Convert dicts to Segment objects if needed, or just write
            # For this simple script, we construct lines directly
            out_file = os.path.join(args.output_dir, f"{meeting_id}.rttm")
            with open(out_file, 'w') as f:
                for s in segments_dict:
                    dur = s['end'] - s['start']
                    f.write(f"SPEAKER {meeting_id} 1 {s['start']:.3f} {dur:.3f} <NA> <NA> {s['speaker']} <NA> <NA>\n")
            print(f"Saved {out_file}")

if __name__ == "__main__":
    main()