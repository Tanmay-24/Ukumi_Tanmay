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
        '-c', 'copy',           # Stream copy instead of re-encoding
        '-avoid_negative_ts', 'make_zero',
        output_file
    ]
    subprocess.run(command, check=True)

def concat_videos(segment_files, final_output):
    # Create a file listing all segments
    with open('segments.txt', 'w') as f:
        for file in segment_files:
            f.write(f"file '{file}'\n")
    
    # Concatenate with stream copy
    command = [
        'ffmpeg',
        '-f', 'concat',
        '-safe', '0',
        '-i', 'segments.txt',
        '-c', 'copy',           # Stream copy for concatenation
        '-movflags', '+faststart',  # Enable fast start for web playback
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
    input_video = "/home/tanmay/Desktop/Ukumi_Tanmay/data/output.mp4"
    output_video = "output/new_edit_python_2.mp4"
    
    try:
        with open("/home/tanmay/Downloads/selected_segments.txt") as f:
            timestamp_data = f.read()
            if timestamp_data.startswith('```'):
                timestamp_data = timestamp_data.strip('```')
    except FileNotFoundError: 
        print("!!!!!!!!!!!! No timestamp data found in processed_transcript.txt, switching to default data !!!!!!!!!!!!!!")           
        
    
    main(input_video, timestamp_data, output_video)