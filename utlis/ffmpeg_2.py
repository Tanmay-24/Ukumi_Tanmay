import json
import subprocess
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

def trim_video_fast(input_file: str, json_file: str, output_file: str):
    """Fast trimming using stream copy"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    kept_segments = data['kept_segments']
    
    # Create complex filter for segments
    filter_parts = []
    for i, segment in enumerate(kept_segments):
        start = segment['timestamp']['start']
        end = segment['timestamp']['end']
        filter_parts.append(f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS[v{i}]")
        filter_parts.append(f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a{i}]")
    
    # Create concat parts
    v_concat = ''.join(f'[v{i}]' for i in range(len(kept_segments)))
    a_concat = ''.join(f'[a{i}]' for i in range(len(kept_segments)))
    
    filter_parts.append(f"{v_concat}concat=n={len(kept_segments)}:v=1:a=0[vout]")
    filter_parts.append(f"{a_concat}concat=n={len(kept_segments)}:v=0:a=1[aout]")
    
    filter_complex = ';'.join(filter_parts)
    
    command = [
        'ffmpeg',
        '-i', input_file,
        '-filter_complex', filter_complex,
        '-map', '[vout]',
        '-map', '[aout]',
        '-c:v', 'libx264',
        '-c:a', 'aac',
        '-preset', 'medium',
        output_file
    ]
    
    try:
        subprocess.run(command, check=True)
        logging.info(f"Trimmed video saved to: {output_file}")
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg error: {e}")
        raise

def main():
    input_video = "/home/tanmay/Desktop/Ukumi_Tanmay/data/riverside_backup-video_the_india opportuni.mp4"
    json_file = "/home/tanmay/Desktop/Ukumi_Tanmay/script_3_output_shri.json"
    output_video = "/home/tanmay/Desktop/Ukumi_Tanmay/output/best_riverside_shri.mp4"
    
    
    trim_video_fast(
        str(input_video),
        str(json_file),
        str(output_video)
    )

if __name__ == "__main__":
    main()