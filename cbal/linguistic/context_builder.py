class ContextBuilder:
    def __init__(self, transcriber):
        self.transcriber = transcriber

    def build_context(self, segments, target_indices, window=20):
        """
        Build context with clear TEXT A and TEXT B marking.
        """
        context_lines = []
        start_idx = max(0, target_indices[0] - window)
        end_idx = min(len(segments), target_indices[-1] + window + 1)
        
        for i in range(start_idx, end_idx):
            seg = segments[i]
            speaker_label = seg.speaker 
            text = self.transcriber.get_text(seg.start, seg.end)
            
            if i == target_indices[0]:
                # First target - TEXT A
                context_lines.append(f"[TEXT A] {speaker_label}: {text}")
            elif i == target_indices[-1]:
                # Second target - TEXT B
                context_lines.append(f"[TEXT B] {speaker_label}: {text}")
            else:
                # Context
                context_lines.append(f"        {speaker_label}: {text}")
            
        return "\n".join(context_lines)