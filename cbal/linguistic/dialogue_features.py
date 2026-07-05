class DialogueAnalyzer:
    @staticmethod
    def get_turn_metrics(segment):
        """
        Returns objective metrics about the turn.
        No word lists. No decisions. Just data.
        """
        text = segment.text or ""
        words = text.split()
        
        return {
            "duration": segment.end - segment.start,
            "word_count": len(words),
            "char_count": len(text),
            "words_per_sec": len(words) / (segment.end - segment.start + 1e-6)
        }