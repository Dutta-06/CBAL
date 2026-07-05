def calculate_changes(original_segments, new_segments):
    """
    Simple diff metric to see how many turns were merged/changed.
    """
    changes = {
        'total_turns_before': len(original_segments),
        'total_turns_after': len(new_segments),
        'merges_made': len(original_segments) - len(new_segments)
    }
    return changes