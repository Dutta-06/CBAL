from cbal.acoustic.acoustic_features import cosine_similarity

class ConflictDetector:
    def __init__(self, extractor, transcript_loader, thresholds):
        self.extractor = extractor
        self.transcriber = transcript_loader
        self.thresh = thresholds

    def scan(self, segments, audio_path):
        errors = []
        
        for i in range(len(segments) - 1):
            curr = segments[i]
            next_s = segments[i+1]
            
            # --- 1. PRE-CALCULATE METRICS ---
            gap = next_s.start - curr.end
            curr_dur = curr.end - curr.start
            next_dur = next_s.end - next_s.start
            
            # Safe text lookup
            if not getattr(curr, 'text', None):
                curr.text = self.transcriber.get_text(curr.start, curr.end)
            if not getattr(next_s, 'text', None):
                next_s.text = self.transcriber.get_text(next_s.start, next_s.end)
            
            text_a = curr.text.strip()
            text_b = next_s.text.strip()

            # Base record (prevents KeyErrors later)
            base_record = {
                'indices': [i, i+1],
                'gap': gap,
                'duration': curr_dur,
                'text': f"{text_a} [GAP] {text_b}",
                'similarity': 0.0 
            }

            # --- CHECK 1: False Split (Same Speaker) ---
            if curr.speaker == next_s.speaker and gap < self.thresh.get('false_split_gap', 3.0):
                # Only flag if previous sentence looks unfinished
                if not any(text_a.endswith(p) for p in ['.', '?', '!']):
                    err = base_record.copy()
                    err['type'] = 'false_split'
                    errors.append(err)

            # --- CHECK 2: Acoustic Confusion (Diff Speaker) ---
            elif curr.speaker != next_s.speaker:
                sim = 0.0 # <--- FIX: Initialize explicitly for all paths
                
                # Optimization: Only extract if segments are long enough
                if (curr_dur > self.thresh.get('min_segment_len_for_embedding', 0.2) and 
                    next_dur > self.thresh.get('min_segment_len_for_embedding', 0.2)):
                    try:
                        emb1 = self.extractor.extract(audio_path, curr.start, curr.end)
                        emb2 = self.extractor.extract(audio_path, next_s.start, next_s.end)
                        if emb1 is not None and emb2 is not None:
                            sim = cosine_similarity(emb1, emb2)
                    except:
                        sim = 0.0

                if sim > self.thresh.get('acoustic_similarity', 0.85):
                    err = base_record.copy()
                    err['type'] = 'acoustic_confusion'
                    err['similarity'] = sim
                    errors.append(err)

        return errors