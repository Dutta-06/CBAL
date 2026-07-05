import copy

class MergeEngine:
    def apply_fixes(self, segments, decisions, errors):
        # 1. Deep copy to protect the original baseline data
        new_segments = [copy.deepcopy(s) for s in segments]
        applied_count = 0
        
        # 2. Identify indices to merge
        to_merge = []
        for i, decision in enumerate(decisions):
            if decision.get('action') == 'MERGE':
                to_merge.append(errors[i]['indices'])
        
        # 3. Sort by second index descending to safely pop from the list
        to_merge.sort(key=lambda x: x[1], reverse=True)

        for idx1, idx2 in to_merge:
            # Anchor metadata
            original_id = new_segments[idx1].speaker
            meeting_id = new_segments[idx1].meeting_id
            
            # THE DURATION FIX: 
            # The new start is the MIN of both (usually idx1.start)
            # The new end is the MAX of both (usually idx2.end)
            new_start = min(new_segments[idx1].start, new_segments[idx2].start)
            new_end = max(new_segments[idx1].end, new_segments[idx2].end)
            
            # Update the primary segment
            new_segments[idx1].start = new_start
            new_segments[idx1].end = new_end
            new_segments[idx1].duration = new_end - new_start
            new_segments[idx1].speaker = original_id 
            
            # Remove the redundant segment
            new_segments.pop(idx2)
            applied_count += 1
                
        return new_segments, applied_count