import os
import logging
import json
import subprocess
from dotenv import load_dotenv
from deepgram import DeepgramClient, PrerecordedOptions

# Load environment variables from .env file
load_dotenv()

def extract_audio_from_video(video_path, audio_path):
    command = [
        'ffmpeg',
        '-i', video_path,
        '-q:a', '0',
        '-map', 'a',
        audio_path
    ]
    subprocess.run(command, check=True)

def generate_transcript_deepgram(audio_path):
    try:
        # STEP 1 Create a Deepgram client using the DEEPGRAM_API_KEY from your environment variables
        deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
        if not deepgram_api_key:
            raise ValueError("DEEPGRAM_API_KEY not found in environment variables")
        deepgram = DeepgramClient(api_key=deepgram_api_key)

        # STEP 2 Read the audio file
        with open(audio_path, 'rb') as audio_file:
            audio_data = audio_file.read()

        # STEP 3 Call the transcribe method on the rest class
        options = PrerecordedOptions(
            model="nova-2",
            smart_format=True,
        )
        response = deepgram.listen.rest.v("1").transcribe_file({"buffer": audio_data}, options)
        save_response_to_json(response.to_dict())
        return response.to_dict()

    except Exception as e:
        print(f"Exception: {e}")
        return None

def save_response_to_json(response, filename='deepgram_response_big_video.json'):
    try:
        with open(filename, 'w') as f:
            json.dump(response, f, indent=4)
        print(f"Response saved to {filename}")
    except Exception as e:
        print(f"Error saving response to JSON: {e}")

if __name__ == "__main__":
    video_path = "/home/tanmay/Desktop/Ukumi_Tanmay/data/output.mp4"
    audio_path = "extracted_audio_big_video.mp3"

    # Extract audio from video
    extract_audio_from_video(video_path, audio_path)

    # Generate transcript from extracted audio
    # response = generate_transcript_deepgram(audio_path)
    # if response:
    #     print("Transcription completed successfully.")
    # else:
    #     print("Transcription failed.")