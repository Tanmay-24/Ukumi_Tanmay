import subprocess
import os

def process_timestamps(timestamp_str):
    # Split space-separated timestamp groups
    timestamp_groups = timestamp_str.strip().split()
    segments = []
    
    for group in timestamp_groups:
        # Split each group into individual timestamps
        q_start, q_end, a_start, a_end = map(float, group.split(','))
        segments.append((q_start, q_end, a_start, a_end))
    
    return segments

def create_segment(input_file, start, end, output_file):
    command = [
        'ffmpeg',
        '-i', input_file,
        '-ss', str(start),
        '-to', str(end),
        # Use proper encoding settings to prevent playback issues
        '-c:v', 'libx264',
        '-c:a', 'aac',
        '-preset', 'medium',  # Balance between encoding speed and quality
        '-crf', '23',        # Constant Rate Factor for quality (lower = better)
        '-avoid_negative_ts', 'make_zero',  # Prevent timestamp issues
        '-async', '1',       # Audio sync
        output_file
    ]
    subprocess.run(command, check=True)

def concat_videos(segment_files, final_output):
    # Create a file listing all segments
    with open('segments.txt', 'w') as f:
        for file in segment_files:
            f.write(f"file '{file}'\n")
    
    # Concatenate all segments with proper encoding
    command = [
        'ffmpeg',
        '-f', 'concat',
        '-safe', '0',
        '-i', 'segments.txt',
        # Maintain quality with proper encoding
        '-c:v', 'libx264',
        '-c:a', 'aac',
        '-preset', 'medium',
        '-crf', '23',
        final_output
    ]
    subprocess.run(command, check=True)
    os.remove('segments.txt')

def main(input_video, timestamp_data, output_video):
    # Process timestamps
    segments = process_timestamps(timestamp_data)
    temp_files = []
    
    print(f"Processing {len(segments)} Q&A pairs...")
    
    for i, (q_start, q_end, a_start, a_end) in enumerate(segments):
        print(f"Processing Q&A pair {i+1}...")
        
        # Create question segment
        q_file = f'temp_q_{i}.mp4'
        print(f"Extracting question segment {q_start} to {q_end}")
        create_segment(input_video, q_start, q_end, q_file)
        temp_files.append(q_file)
        
        # Create answer segment
        a_file = f'temp_a_{i}.mp4'
        print(f"Extracting answer segment {a_start} to {a_end}")
        create_segment(input_video, a_start, a_end, a_file)
        temp_files.append(a_file)
    
    print("Concatenating all segments...")
    # Concatenate all segments
    concat_videos(temp_files, output_video)
    
    print("Cleaning up temporary files...")
    # Cleanup temp files
    for file in temp_files:
        try:
            os.remove(file)
        except Exception as e:
            print(f"Warning: Could not remove temporary file {file}: {e}")
    
    print(f"Complete! Output saved to {output_video}")

if __name__ == "__main__":
    input_video = "817cb0da-5498-4868-bea6-a1de70217a46.mp4"
    output_video = "new_edit_python.mp4"
    
    try:
        with open("/home/tanmay/Downloads/selected_transcript.txt") as f:
            timestamp_data = f.read()
            if timestamp_data.startswith('```'):
                timestamp_data = timestamp_data.strip('```')
    except FileNotFoundError: 
        print("!!!!!!!!!!!! No timestamp data found in processed_transcript.txt, switching to default data !!!!!!!!!!!!!!")           
        timestamp_data=  """10.32,18.105,18.485,155.65001 169.205,172.98,174.34,257.01 271.965,279.745,280.41998,517.14 531.125,539.865,540.77,729.245 907.815,916.31,916.93,1084.61 1586.5901,1599.235,1604.46,1691.8301 1743.0751,1746.275,1757.37,1810.09 1813.31,1814.17,1830.24,1907.815"""
    
    main(input_video, timestamp_data, output_video)