import os

class Segment:
    def __init__(self, start, end, speaker, meeting_id):
        self.start = float(start)
        self.end = float(end)
        self.duration = self.end - self.start
        self.speaker = speaker  # This will be 'SPEAKER_01', etc.
        self.meeting_id = meeting_id
        self.text = ""

def load_rttm(path):
    segments = []
    if not os.path.exists(path):
        return []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8: continue
            # Index 3: Start, Index 4: Duration, Index 7: SpeakerID
            start = float(parts[3])
            duration = float(parts[4])
            segments.append(Segment(
                start, 
                start + duration, 
                parts[7], 
                parts[1]
            ))
    return segments

def write_rttm(segments, output_path, meeting_id):
    """
    Writes segments to NIST RTTM format.
    Strictly uses the anonymous labels stored in the segment objects.
    """
    with open(output_path, 'w') as f:
        for seg in segments:
            # Format: SPEAKER <file> <chnl> <start> <dur> <ortho> <stype> <name> <conf>
            line = (
                f"SPEAKER {meeting_id} 1 "
                f"{seg.start:.3f} {seg.duration:.3f} "
                f"<NA> <NA> {seg.speaker} <NA> <NA>\n"
            )
            f.write(line)