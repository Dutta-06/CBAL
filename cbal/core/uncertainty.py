class UncertaintyManager:
    def __init__(self, config):
        self.thresh = config['thresholds'] # Use the unified thresholds dict

    def evaluate_decision(self, decision, error_metadata):
        action = decision.get('action', 'KEEP')
        score = decision.get('confidence', 0.0)

        if action == 'KEEP': return False
        if score < self.thresh['min_confidence']: return False

        # Physics Checks
        err_type = error_metadata['type']

        if err_type == 'acoustic_confusion':
            # Uses 'similarity' key which exists for this type
            sim = error_metadata.get('similarity', 0.0)
            if sim < self.thresh['min_allowed_sim_for_merge']: 
                return False

        if err_type == 'false_split':
            # Uses 'gap' key which exists for this type
            gap = error_metadata.get('gap', 1.0)
            if gap > self.thresh['max_allowed_gap_for_merge']: 
                return False

        return True