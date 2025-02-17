import json
from pydub import AudioSegment



def detect_segments(
    audio,
    frame_duration_ms=30,
    padding_duration_ms=300,
    aggressiveness=3,
    post_speech_padding_sec=0.2,
):
    print(f"Detecting segments with frame_duration_ms={frame_duration_ms}, padding_duration_ms={padding_duration_ms}, aggressiveness={aggressiveness}, post_speech_padding_sec={post_speech_padding_sec}")
    """
    Detect speech segments using voice activity detection (VAD) via webrtcvad,
    with an adjustable post-speech padding to determine the exact cut.

    This function converts the given pydub AudioSegment to mono 16-bit PCM at a supported
    sample rate (8000, 16000, 32000, or 48000 Hz), splits it into frames, and uses webrtcvad
    to determine which frames contain speech. Contiguous speech frames are merged into segments.
    The end of each segment is adjusted by adding a post-speech padding (in seconds). Setting
    post_speech_padding_sec to 0 cuts immediately when speech stops; setting it to 1 waits 1 second
    of silence before cutting.

    Parameters:
      audio (AudioSegment): The audio to analyze.
      frame_duration_ms (int): Duration of each frame in milliseconds (must be 10, 20, or 30).
          Defaults to 30. If an alternate value is passed via the keyword 'chunk_ms', that value is used.
      padding_duration_ms (int): If the gap between speech segments is less than this (in ms),
          the segments are merged. Defaults to 300 ms.
      aggressiveness (int): VAD aggressiveness mode (0 = least, 3 = most aggressive). Defaults to 3.
      post_speech_padding_sec (float): Additional time (in seconds) to include after the last detected speech
          frame. Use 0 for an immediate cut, 1 to wait 1 second in silence.
      **kwargs: Accepts an optional 'chunk_ms' keyword to override frame_duration_ms for backward compatibility.

    Returns:
      List[Dict]: A list of dictionaries, each with 'start' and 'end' (in seconds) marking a detected speech segment.
    """
    import webrtcvad

    # webrtcvad only supports frame durations of 10, 20, or 30 ms.
    frame_duration_ms = 30 if frame_duration_ms not in (10, 20, 30) else frame_duration_ms
    print(f"Using frame_duration_ms: {frame_duration_ms}")

    # Ensure audio is mono and at a supported sample rate.
    audio = audio.set_channels(1)
    if audio.frame_rate not in (8000, 16000, 32000, 48000):
        audio = audio.set_frame_rate(16000)
    sample_rate = audio.frame_rate
    sample_width = audio.sample_width  # bytes per sample (typically 2 for 16-bit PCM)
    raw_audio = audio.raw_data
    print(f"Audio properties: channels=1, sample_rate={sample_rate}, sample_width={sample_width}")

    vad = webrtcvad.Vad(aggressiveness)
    # Calculate frame size in samples and then in bytes.
    frame_size = int(sample_rate * frame_duration_ms / 1000)
    frame_bytes = frame_size * sample_width

    # Split raw audio into frames of exact length.
    frames = []
    for i in range(0, len(raw_audio) - frame_bytes + 1, frame_bytes):
        frame = raw_audio[i : i + frame_bytes]
        timestamp = i / (sample_rate * sample_width)
        frames.append((timestamp, frame))

    # Label each frame using VAD.
    speech_flags = []
    for timestamp, frame in frames:
        try:
            is_speech = vad.is_speech(frame, sample_rate)
        except Exception as e:
            print(f"Error processing frame at {timestamp:.2f} sec: {e}")
            is_speech = False
        speech_flags.append((timestamp, is_speech))

    # Aggregate contiguous speech frames into segments,
    # and adjust the segment end by adding post_speech_padding_sec.
    segments = []
    segment_start = None
    last_speech_timestamp = None
    for timestamp, is_speech in speech_flags:
        if is_speech:
            if segment_start is None:
                segment_start = timestamp
            last_speech_timestamp = timestamp
        else:
            if segment_start is not None and last_speech_timestamp is not None:
                # End the segment at the last speech timestamp plus the post-speech padding.
                segments.append(
                    {
                        "start": segment_start,
                        "end": last_speech_timestamp + post_speech_padding_sec,
                    }
                )
                segment_start = None
                last_speech_timestamp = None
    if segment_start is not None:
        total_duration = len(raw_audio) / (sample_rate * sample_width)
        segments.append({"start": segment_start, "end": total_duration})

    # Merge segments that are separated by less than padding_duration_ms.
    merged_segments = []
    if segments:
        current = segments[0]
        for seg in segments[1:]:
            if seg["start"] - current["end"] < padding_duration_ms / 1000.0:
                current["end"] = seg["end"]
            else:
                merged_segments.append(current)
                current = seg
        merged_segments.append(current)

    return merged_segments


def save_json(data, filename):
    print(f"Saving JSON data to {filename}")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"JSON data saved successfully to {filename}")



def process_audio(file_path):
    print(f"Processing audio file: {file_path}")

    audio = AudioSegment.from_file(file=file_path)
    print(f"Audio loaded: duration={len(audio)/1000:.2f}s, channels={audio.channels}, sample_width={audio.sample_width}, frame_rate={audio.frame_rate}")

    raw_segments = detect_segments(audio)
    print(f"Detected {len(raw_segments)} segments")
    return raw_segments

def get_silence_gaps(raw_segments):
    print("Calculating silence gaps")
    silence_gaps = []
    for i in range(len(raw_segments) - 1):
        silence_start = raw_segments[i]['end']
        silence_end = raw_segments[i + 1]['start']
        silence_gaps.append({'start': silence_start, 'end': silence_end})
    print(f"Found {len(silence_gaps)} silence gaps")
    return silence_gaps
