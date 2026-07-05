class Calibrator:
    def __init__(self, model_type="2b"):
        self.model_type = model_type

    def calibrate(self, raw_confidence):
        """
        Adjusts raw LLM confidence scores to be more realistic.
        """
        # Gemma 2B tends to be overconfident. 
        # We apply a penalty to smooth the distribution.
        
        if raw_confidence >= 0.95:
            # Cap extreme confidence
            return 0.90
        
        if raw_confidence < 0.6:
            # If it's shaky, kill it.
            return 0.0
            
        # Linear scaling for the middle range
        return raw_confidence * 0.9