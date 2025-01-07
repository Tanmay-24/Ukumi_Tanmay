import os
import logging
import json
from dotenv import load_dotenv

from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
)

# Load environment variables from .env file
load_dotenv()

AUDIO_URL = {
    "url": "https://ukumi-audio.s3.amazonaws.com/676a857fc290daf5f2c9f366.mp3"
}

def generate_transcript_deepgram(audio_url):
    try:
        # STEP 1 Create a Deepgram client using the DEEPGRAM_API_KEY from your environment variables
        deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
        if not deepgram_api_key:
            raise ValueError("DEEPGRAM_API_KEY not found in environment variables")
        deepgram: DeepgramClient = DeepgramClient(api_key=deepgram_api_key)

        # STEP 2 Call the transcribe_url method on the rest class
        options: PrerecordedOptions = PrerecordedOptions(
            model="nova-2",
            smart_format=True,
        )
        response = deepgram.listen.rest.v("1").transcribe_url({"url":audio_url}, options)
        # print(response["results"])
        save_response_to_json(response.to_dict())
        return response.to_dict()

    except Exception as e:
        print(f"Exception: {e}")
        return None

def save_response_to_json(response, filename='deepgram_response.json'):
    try:
        with open(filename, 'w') as f:
            json.dump(response, f, indent=4)
        print(f"Response saved to {filename}")
    except Exception as e:
        print(f"Error saving response to JSON: {e}")

if __name__ == "__main__":
    response = generate_transcript_deepgram(AUDIO_URL["url"])
    # print(response)