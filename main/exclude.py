def find_segments_to_remove(input_json: dict, total_duration: float) -> list[dict]:
    """
    Generate timestamps of segments to remove based on kept segments.
    
    Args:
        input_json (dict): JSON with segments to keep
        total_duration (float): Total duration of the video/audio in seconds
        
    Returns:
        list[dict]: List of segments to remove in format [{"start": float, "end": float}, ...]
    """
    # Extract and sort kept segments
    kept_segments = []
    for segment in input_json['segments']:
        kept_segments.append({
            'start': segment['timestamp']['start'],
            'end': segment['timestamp']['end']
        })
    
    # Sort segments by start time
    kept_segments.sort(key=lambda x: x['start'])
    
    # Find gaps between kept segments
    segments_to_remove = []
    
    # Check if there's a gap at the start
    if kept_segments[0]['start'] > 0:
        segments_to_remove.append({
            'start': 0,
            'end': kept_segments[0]['start']
        })
    
    # Find gaps between segments
    for i in range(len(kept_segments) - 1):
        if kept_segments[i]['end'] < kept_segments[i + 1]['start']:
            segments_to_remove.append({
                'start': kept_segments[i]['end'],
                'end': kept_segments[i + 1]['start']
            })
    
    # Check if there's a gap at the end
    if kept_segments[-1]['end'] < total_duration:
        segments_to_remove.append({
            'start': kept_segments[-1]['end'],
            'end': total_duration
        })
    
    return segments_to_remove

