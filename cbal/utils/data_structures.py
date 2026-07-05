from dataclasses import dataclass
from typing import Optional

@dataclass
class Segment:
    start: float
    end: float
    speaker: str
    text: Optional[str] = None # Default to None

    @property
    def duration(self):
        return self.end - self.start

    def __repr__(self):
        txt_preview = f" '{self.text[:10]}...'" if self.text else ""
        return f"<Seg {self.speaker} {self.start:.2f}-{self.end:.2f}{txt_preview}>"