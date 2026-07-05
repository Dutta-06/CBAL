class Validator:
    def __init__(self, thresholds):
        self.thresh = thresholds

    def validate_merge(self, segment1, segment2, audio_similarity):
        """
        Safety check: Even if LLM says MERGE, forbid it if audio is totally different.
        """
        # If LLM says merge but audio is < 50% similar, block it.
        # This prevents "hallucinated" merges.
        if audio_similarity < 0.50:
            return False, "Audio mismatch (Safety Block)"
        
        return True, "Valid"