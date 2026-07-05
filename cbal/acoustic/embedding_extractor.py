import torch
import librosa
import numpy as np
from transformers import WavLMModel

class EmbeddingExtractor:
    def __init__(self, model_name="microsoft/wavlm-base-plus", device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"🔊 Loading WavLM on {self.device}...")
        self.model = WavLMModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def extract(self, audio_path, start, end):
        duration = end - start
        if duration < 0.05: return None

        try:
            # Load with Librosa (safe, robust, resamples to 16k)
            y, sr = librosa.load(audio_path, sr=16000, offset=start, duration=duration)
            
            # WavLM needs at least ~25ms
            if len(y) < 400: return None

            # Prepare tensor
            input_values = torch.from_numpy(y).float().unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(input_values)
                # Mean pooling
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            
            return embedding
        except Exception as e:
            # print(f"Audio Error: {e}")
            return None