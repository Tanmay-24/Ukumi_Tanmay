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

# Example usage:
def main():
    # Example input
    input_json = {
  "segments": [
    {
      "timestamp": {
        "start": 10.32,
        "end": 155.65
      },
      "rating": 9.0,
      "title": "Responsibilities of a Growth Manager"
    },
    {
      "timestamp": {
        "start": 169.205,
        "end": 257.01
      },
      "rating": 8.0,
      "title": "Critical Skills for Growth Roles"
    },
    {
      "timestamp": {
        "start": 271.965,
        "end": 384.015
      },
      "rating": 7.0,
      "title": "Common Pitfalls in Growth Interviews"
    },
    {
      "timestamp": {
        "start": 531.125,
        "end": 729.245
      },
      "rating": 8.0,
      "title": "Researching Company Growth Strategies"
    },
    {
      "timestamp": {
        "start": 907.815,
        "end": 1235.605
      },
      "rating": 9.0,
      "title": "Technical Skills for Growth Roles"
    },
    {
      "timestamp": {
        "start": 1242.625,
        "end": 1384.115
      },
      "rating": 8.0,
      "title": "Importance of Data Analytics in Growth"
    },
    {
      "timestamp": {
        "start": 1390.13,
        "end": 1517.665
      },
      "rating": 7.0,
      "title": "Soft Skills in Growth Functions"
    },
    {
      "timestamp": {
        "start": 1586.59,
        "end": 1741.175
      },
      "rating": 8.0,
      "title": "Assessing Culture and Team Fit"
    },
    {
      "timestamp": {
        "start": 1813.31,
        "end": 1946.61
      },
      "rating": 7.0,
      "title": "Feedback in Growth Interviews"
    }
  ]
}
    
    total_duration = 1963.2588  # Example duration in seconds
    
    removed_segments = find_segments_to_remove(input_json, total_duration)
    print("Segments to remove:", removed_segments)
    
if __name__ == "__main__":
    main()