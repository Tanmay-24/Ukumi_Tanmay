import json
import os
import requests
import random
import argparse
from dotenv import load_dotenv
from openai import OpenAI

def load_json_data(file_path):
    """Load and parse JSON data from a file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_segments(data):
    """Extract segment information from the JSON data."""
    return [{'start': float(item['start']), 'end': float(item['end']), 'text': item['text']} 
            for item in data[0]['text']]

def get_llm_keywords(client, transcript, model="gpt-4o", temperature=0.7):
    """Generate relevant b-roll keywords using LLM for the entire transcript."""
    podcast_prompt = """You are analyzing a transcript from a podcast video. Your task is to determine if b-roll footage would enhance the viewer's understanding for each segment of the conversation, and if so, provide a specific keyword to search for relevant footage.

B-roll should ONLY be suggested when it adds significant value:
1. When explaining complex concepts or ideas
2. When describing physical objects or places
3. When referring to specific visual examples
4. When explaining processes or demonstrations

DO NOT suggest b-roll for:
- General conversation
- Personal opinions
- Abstract discussions without clear visual elements
- Segments that are purely conversational

Analyze the entire transcript and provide b-roll suggestions for relevant segments. For each suggestion, provide:
1. "k": A specific, concrete keyword for searching video footage
2. "i": The index of the segment in the transcript (starting from 0)
3. "need_broll": true (always true for suggested b-roll)

Input:
{transcript}

Output format (JSON only):
[{{"k": "keyword", "i": 0, "need_broll": true}}, {{"k": "keyword", "i": 5, "need_broll": true}}]

Ensure that your suggestions are accurate, relevant, and enhance the viewer's understanding of the content.
"""

    prompt = podcast_prompt.format(transcript=str(transcript))
    
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=temperature
    )
    
    broll_data = chat_completion.choices[0].message.content
    print("Processing transcript for b-roll suggestions...")
    
    try:
        # Handle possible JSON formatting issues
        if "```json" in broll_data:
            broll_data = broll_data.split('```json')[1].split('```')[0].strip()
        if "```" in broll_data:
            broll_data = broll_data.split('```')[1].split('```')[0].strip()
            
        broll_info = json.loads(broll_data)
        return broll_info
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        print(f"Raw response: {broll_data}")
        return []

def fetch_pexels_video(keyword, api_key, orientation="landscape", per_page=5, quality=["hd", "sd"]):
    """Fetch video from Pexels API with improved parameters."""
    url = f"https://api.pexels.com/videos/search?query={keyword}&orientation={orientation}&per_page={per_page}&size=medium"
    headers = {"Authorization": api_key}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        if 'total_results' in data and data['total_results'] > 0:
            # Try to get most relevant video by browsing a few options
            for video_idx in range(min(per_page, len(data['videos']))):
                video_info = data['videos'][video_idx]
                
                # Find a suitable video file with preferred quality
                for video_file in video_info['video_files']:
                    if video_file['quality'] in quality:
                        return {
                            'video': video_file['link'], 
                            'thumbnail': video_info['image'],
                            'duration': video_info['duration']
                        }
                
            # If no suitable quality found in any video, use the first video's first file
            return {
                'video': data['videos'][0]['video_files'][0]['link'],
                'thumbnail': data['videos'][0]['image'],
                'duration': data['videos'][0]['duration']
            }
        else:
            return "Invalid keyword"
    except Exception as e:
        print(f"Error fetching video for keyword '{keyword}': {e}")
        return "API error"

def save_json_output(data, output_file):
    """Save output data to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"JSON output has been saved to {output_file}")

def main():
    """Main function with configurable parameters."""
    # Configurable parameters
    config = {
        "input_file": "output/sen_AI.json",         # Path to input JSON file
        "output_file": "broll_data.json",           # Path to output JSON file
        "llm_model": "gpt-4o",                      # Model to use for keyword generation
        "llm_temperature": 0.6,                     # Temperature for LLM responses
        "video_orientation": "landscape",           # Orientation of videos to fetch
        "video_results_per_query": 5,               # Number of videos to consider per query
        "preferred_qualities": ["hd"],              # Preferred video qualities in order of preference
    }
    
    # Command line arguments override config
    parser = argparse.ArgumentParser(description='Generate B-roll suggestions for podcast videos')
    parser.add_argument('--input', help='Input JSON file path')
    parser.add_argument('--output', help='Output JSON file path')
    args = parser.parse_args()
    
    if args.input:
        config["input_file"] = args.input
    if args.output:
        config["output_file"] = args.output
    
    # Load environment variables for API keys
    load_dotenv()
    
    # Initialize OpenAI client
    client = OpenAI()
    
    # Pexels API key (from env or config)
    PEXELS_API_KEY = os.getenv("PEXELS_API_KEY", "lrLQbU4XhYTpmpdrxAXr0worOLw3hiTq1gFITCNXMpQJDRKTDxhCm2X3")
    
    # Process data
    print(f"Loading data from {config['input_file']}...")
    sen_AI_data = load_json_data(config['input_file'])
    extracted_data = extract_segments(sen_AI_data)
    transcript = " ".join([x["text"] for x in extracted_data])
    
    # Generate keywords for segments
    print("Generating keywords for B-roll...")
    broll_info = get_llm_keywords(
        client, 
        transcript, 
        model=config["llm_model"],
        temperature=config["llm_temperature"]
    )
    
    # Create a mapping from segment index to broll_info item
    broll_info_map = {item['i']: item for item in broll_info}
    
    # Create final output structure
    print("Fetching B-roll videos...")
    final_output = []
    
    for i, segment in enumerate(extracted_data):
        output_segment = {
            'start': segment['start'],
            'end': segment['end'],
            'text': segment['text'],
            'use_broll': False,
            'broll_url': None,
            'thumbnail': None,
            'keyword': None,
            'duration': None
        }
        
        # Check if this segment needs b-roll
        if i in broll_info_map:
            element = broll_info_map[i]
            print(f"Fetching video for segment {i}, keyword: {element['k']}")
            
            video_data = fetch_pexels_video(
                element["k"], 
                PEXELS_API_KEY,
                orientation=config["video_orientation"],
                per_page=config["video_results_per_query"],
                quality=config["preferred_qualities"]
            )
            
            if video_data not in ["Invalid keyword", "API error"]:
                output_segment['use_broll'] = True
                output_segment['broll_url'] = video_data['video']
                output_segment['thumbnail'] = video_data['thumbnail']
                output_segment['keyword'] = element["k"]
                output_segment['duration'] = video_data.get('duration')
        
        final_output.append(output_segment)
    
    # Save output
    save_json_output(final_output, config["output_file"])
    
    # Print summary
    total_segments = len(extracted_data)
    broll_segments = sum(1 for segment in final_output if segment['use_broll'])
    print(f"Summary: Added B-roll to {broll_segments}/{total_segments} segments ({broll_segments/total_segments*100:.1f}%)")

if __name__ == "__main__":
    main()
