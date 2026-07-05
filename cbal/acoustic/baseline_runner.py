import os
import torch
from pyannote.audio import Pipeline

class BaselineRunner:
    def __init__(self, hf_token=None):
        self.token = hf_token or os.environ.get("HF_TOKEN")
        if not self.token:
            print("⚠️ Warning: No HF_TOKEN found. Pyannote pipeline might fail.")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_pipeline(self, model="pyannote/speaker-diarization-3.1"):
        print(f"Loading baseline pipeline: {model}...")
        try:
            self.pipeline = Pipeline.from_pretrained(
                model, 
                use_auth_token=self.token
            ).to(self.device)
            return True
        except Exception as e:
            print(f"❌ Failed to load Pyannote: {e}")
            return False

    def run(self, audio_path):
        """
        Runs diarization and returns RTTM-compatible segments.
        """
        print(f"Running baseline on {audio_path}...")
        diarization = self.pipeline(audio_path)
        
        # Convert to list of dicts for internal use
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })
        return segments