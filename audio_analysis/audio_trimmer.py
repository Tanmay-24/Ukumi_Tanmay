import json
import subprocess
import sys

def read_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def generate_ffmpeg_command(input_file, output_file, timestamps):
    filter_complex = []
    for i, segment in enumerate(timestamps):
        filter_complex.append(f"[0:a]atrim={segment['start']}:{segment['end']},asetpts=PTS-STARTPTS[a{i}]")
    
    filter_complex.append(f"{' '.join([f'[a{i}]' for i in range(len(timestamps))])}concat=n={len(timestamps)}:v=0:a=1[outa]")
    
    command = [
        "ffmpeg",
        "-i", input_file,
        "-filter_complex", ';'.join(filter_complex),
        "-map", "[outa]",
        output_file
    ]
    return command

def main(input_file, json_file, output_file):
    timestamps = read_json(json_file)
    command = generate_ffmpeg_command(input_file, output_file, timestamps)
    
    try:
        subprocess.run(command, check=True)
        print(f"Successfully created trimmed audio: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running ffmpeg: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python audio_trimmer.py <input_audio_file> <json_file> <output_audio_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    json_file = sys.argv[2]
    output_file = sys.argv[3]
    
    main(input_file, json_file, output_file)
