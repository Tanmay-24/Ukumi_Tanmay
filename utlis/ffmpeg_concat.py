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
        '-c:v', 'libx264',       # Use H.264 encoding for video
        '-c:a', 'aac',           # Use AAC encoding for audio
        '-preset', 'ultrafast',  # Optimize for speed
        '-crf', '23',            # Set quality
        '-avoid_negative_ts', 'make_zero',  # Fix timestamp issues
        '-threads', '0',         # Use all available CPU cores
        output_file
    ]
    subprocess.run(command, check=True)

def concat_videos(segment_files, final_output):
    # Create a temporary file listing all segment files
    with open('segments.txt', 'w') as f:
        for file in segment_files:
            f.write(f"file '{file}'\n")

    # Concatenate segments into the final video
    command = [
        'ffmpeg',
        '-f', 'concat',
        '-safe', '0',
        '-i', 'segments.txt',
        '-c:v', 'libx264',      # Ensure re-encoding for smooth output
        '-c:a', 'aac',
        '-preset', 'ultrafast',    # Balance speed and quality
        '-crf', '23',           # Maintain quality
        '-threads', '0',
        final_output
    ]
    subprocess.run(command, check=True)
   

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
    concat_videos(temp_files, output_video)

    print("Cleaning up temporary files...")
    for file in temp_files:
        try:
            os.remove(file)
        except Exception as e:
            print(f"Warning: Could not remove temporary file {file}: {e}")

    print(f"Complete! Output saved to {output_video}")

if __name__ == "__main__":
    input_video = "/home/tanmay/Desktop/Ukumi_Tanmay/data/riverside_backup-video_the_india opportuni.mp4"
    output_video = "output/4_agents_riverside_shri.mp4"

    try:
        with open("/home/tanmay/Desktop/Ukumi_Tanmay/script_output_riverside_shri_2.txt") as f:
            timestamp_data = f.read().strip()
            if timestamp_data.startswith('```'):
                timestamp_data = timestamp_data.strip('```')
    except FileNotFoundError:
        print("No timestamp data found in processed_transcript.txt, switching to default data.")

    main(input_video, timestamp_data, output_video)
