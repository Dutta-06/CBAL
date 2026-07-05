import json

class TranscriptLoader:
    def __init__(self, json_path):
        self.words = self._load_words(json_path)

    def _load_words(self, json_path):
        """
        Parses the Whisper/AMI JSON structure.
        Iterates through segments to extract word-level timestamps.
        """
        words = []
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Since the JSON starts with '[', it is a list of segment dictionaries
            segments = data if isinstance(data, list) else data.get('segments', [])
            
            for seg in segments:
                # Drill into the 'words' array inside each segment
                seg_words = seg.get('words', [])
                for w in seg_words:
                    words.append({
                        # Whisper often includes a leading space in "word" (e.g., " The")
                        'text': w.get('word', w.get('text', '')).strip(),
                        'start': float(w['start']),
                        'end': float(w['end'])
                    })
            
            # Sort chronologically to ensure range lookups are consistent
            return sorted(words, key=lambda x: x['start'])
        except Exception as e:
            print(f"❌ TranscriptLoader Error: {e}")
            return []

    def get_text(self, start, end):
        """
        Retrieves all words within the specified RTTM segment boundaries.
        Includes a 0.1s tolerance to handle alignment drift.
        """
        # Buffer helps catch words that start slightly before/after the RTTM boundary
        matches = [
            w['text'] for w in self.words 
            if (start - 0.1) <= w['start'] < (end + 0.1)
        ]
        return " ".join(matches)