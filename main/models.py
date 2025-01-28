
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional

class Timestamp(BaseModel):
    start: float = Field(..., ge=0)
    end: float = Field(..., ge=0)
    
    @field_validator('end')
    def end_must_be_after_start(cls, v, values):
        if 'start' in values.data and v <= values.data['start']:
            raise ValueError('end time must be after start time')
        return v

class Segment(BaseModel):
    timestamp: Timestamp
    rating: float = Field(..., ge=0, le=10)
    title: str = Field(..., min_length=1)

class RemovedSegment(BaseModel):
    start: float = Field(..., ge=0)
    end: float = Field(..., ge=0)
    reason: str = Field(..., min_length=1)
    
class TranscriptAnalysis(BaseModel):
    segments: List[Segment]
    # removed_segments: Optional[List[RemovedSegment]]

    @field_validator('segments')
    def segments_must_not_overlap(cls, v):
        if not v:
            return v
        sorted_segments = sorted(v, key=lambda x: x.timestamp.start)
        for i in range(len(sorted_segments) - 1):
            if sorted_segments[i].timestamp.end > sorted_segments[i + 1].timestamp.start:
                raise ValueError('segments must not overlap')
        return v

def parse_text_format(text: str) -> dict:
    """Parse plaintext format into dictionary matching our schema"""
    segments = []
    
    current_section = None
    for line in text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
            
        if "KEPT SEGMENTS:" in line:
            current_section = "kept"
            continue
            
        if current_section == "kept" and ',' in line:
            try:
                parts = [p.strip('"') for p in line.split(',')]
                if len(parts) >= 6:  # Ensure we have all required parts
                    segments.append({
                        "timestamp": {
                            "start": float(parts[0]),
                            "end": float(parts[3])
                        },
                        "rating": float(parts[5]),
                        "title": parts[4]
                    })
            except (IndexError, ValueError) as e:
                print(f"Warning: Skipping invalid kept segment line: {line}")

    return {
        "segments": segments
    }
    
def parse_text_format_2(text: str) -> dict:
    removed_segments = []
    current_section = None
    
    for line in text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        
        if "REMOVED SEGMENTS:" in line:
            current_section = "removed"
            continue
        
        if current_section == "removed" and ',' in line:
            try:
                start, end, reason = line.split(',', 2)
                removed_segments.append({
                    "start": float(start),
                    "end": float(end),
                    "reason": reason.strip('"')
                })
            except (ValueError, IndexError) as e:
                print(f"Warning: Skipping invalid line: {line}")
                continue
    
    return {"removed_segments": removed_segments}