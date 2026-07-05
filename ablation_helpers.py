"""
ablation_helpers.py
====================
Shared utilities for all CBAL ablation scripts.
Drop this file in the same directory as the ablation scripts.

Key fix: verify_fix() uses crop() instead of get_labels(),
which correctly finds overlapping reference segments.
"""

import json
import os
from pyannote.core import Annotation, Segment


def load_reference_rttm(path, mid):
    if not os.path.exists(path):
        return None
    ref = Annotation(uri=mid)
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            start, dur = float(parts[3]), float(parts[4])
            ref[Segment(start, start + dur)] = parts[7]
    return ref


def get_speaker_at(reference, time_point):
    """
    Returns the dominant speaker label at a given time point.
    Uses crop() which correctly finds overlapping segments.
    Returns None if no speech found at that point.
    """
    probe = Segment(time_point - 0.05, time_point + 0.05)
    try:
        cropped = reference.crop(probe)
        tracks = list(cropped.itertracks(yield_label=True))
        if not tracks:
            return None
        # Return the label with the longest overlap
        best_label, best_dur = None, 0.0
        for seg, _, label in tracks:
            dur = min(seg.end, probe.end) - max(seg.start, probe.start)
            if dur > best_dur:
                best_dur = dur
                best_label = label
        return best_label
    except Exception:
        return None


def verify_fix(reference, seg1, seg2):
    """
    Check if merging seg1+seg2 is correct against ground truth.
    Returns 'CORRECT', 'INCORRECT', or 'UNKNOWN'.

    Fix vs original run_cbal_full.py:
      - Uses crop() instead of get_labels() — get_labels() returns
        empty set when the probe window doesn't fully overlap a track,
        causing all results to be UNKNOWN.
    """
    if reference is None:
        return "UNKNOWN"

    mid1 = seg1.start + (seg1.end - seg1.start) / 2
    mid2 = seg2.start + (seg2.end - seg2.start) / 2

    spk1 = get_speaker_at(reference, mid1)
    spk2 = get_speaker_at(reference, mid2)

    if spk1 is None or spk2 is None:
        return "UNKNOWN"

    if spk1 != spk2:
        return "INCORRECT"

    # Check if any OTHER speaker occupies the gap between seg1 and seg2
    gap_start, gap_end = seg1.end, seg2.start
    if gap_end > gap_start:
        try:
            gap_cropped = reference.crop(Segment(gap_start, gap_end))
            for _, _, label in gap_cropped.itertracks(yield_label=True):
                if label != spk1:
                    return "INCORRECT"
        except Exception:
            pass

    return "CORRECT"


def load_transcript_words(json_path):
    try:
        with open(json_path) as f:
            data = json.load(f)
        words = []
        segs = data if isinstance(data, list) else data.get('segments', [])
        for s in segs:
            for w in s.get('words', []):
                words.append({
                    'start': float(w['start']),
                    'end':   float(w['end'])
                })
        return words
    except Exception:
        return []


def is_gap_clear(start, end, words):
    """Returns True if no word midpoints fall between start and end."""
    for w in words:
        mid = (w['start'] + w['end']) / 2
        if start < mid < end:
            return False
    return True


def apply_merges(segments, decisions, errors):
    """Apply MERGE decisions to a segment list. Safe for index shifting."""
    indices_to_merge = []
    for i, dec in enumerate(decisions):
        if dec.get('action') == 'MERGE':
            indices_to_merge.append(errors[i]['indices'])
    indices_to_merge.sort(key=lambda x: x[1], reverse=True)
    fixed = segments.copy()
    count = 0
    for idx1, idx2 in indices_to_merge:
        if idx1 >= len(fixed) or idx2 >= len(fixed):
            continue
        fixed[idx1].end = fixed[idx2].end
        fixed.pop(idx2)
        count += 1
    return fixed, count