import json
import os
import requests
import logging
import uuid
import time
import argparse
from urllib.parse import urlparse
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("broll_generator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ai_broll")

# Constants
MAX_RETRIES = 3
BACKOFF_FACTOR = 2
MAX_WORKERS = 4  # For parallel processing
DEFAULT_PEXELS_PAGE_SIZE = 5
DEFAULT_UNSPLASH_PAGE_SIZE = 5


# Commented out FAL-related code
# FAL_API_KEY = "your-fal-key"  # Replace with your FAL key
# FAL_API_SECRET = "your-fal-secret"  # Replace with your FAL secret

class BrollGenerator:
    """
    A class to generate B-roll suggestions for podcast videos, 
    with support for chapter-wise processing and multiple B-roll types.
    """
    
    def __init__(self, openai_key=None, pexels_key=None, unsplash_access_key=None, unsplash_secret_key=None):
        """Initialize the BrollGenerator with API keys and configuration."""
        # Set API keys
        self.openai_client = OpenAI(api_key=openai_key or OPENAI_API_KEY)
        self.pexels_api_key = pexels_key or PEXELS_API_KEY
        self.unsplash_access_key = unsplash_access_key or UNSPLASH_ACCESS_KEY
        self.unsplash_secret_key = unsplash_secret_key or UNSPLASH_SECRET_KEY
        
        # Statistics tracking
        self.stats = {
            "api_calls": 0,
            "tokens_used": 0,
            "processing_time": 0,
            "chapters_processed": 0,
            "broll_suggestions": 0
        }
    
    def load_json_data(self, file_path):
        """Load and parse JSON data from a file."""
        logger.info(f"Loading data from {file_path}...")
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading JSON data: {str(e)}")
            raise
    
    def extract_transcript_data(self, data):
        """Extract transcript data including chapters, sentences, and words."""
        try:
            # Get chapters
            channels = data['data'][1]['channels_present']
            if 'chapter_data' not in channels:
                logger.warning("Chapter data not found in input, creating single chapter")
                # Create a default chapter if none exists
                transcript = self.extract_sentences(data)
                if transcript:
                    return {
                        "chapters": [{
                            "title": "Full Podcast",
                            "start": float(transcript[0]['start']),
                            "end": float(transcript[-1]['end'])
                        }],
                        "sentences": transcript,
                        "words": self.extract_words(data)
                    }
                else:
                    raise ValueError("No transcript data found")
            
            # Extract all data
            chapters = data['data'][2]['channels']['chapter_data']
            return {
                "chapters": chapters,
                "sentences": self.extract_sentences(data),
                "words": self.extract_words(data)
            }
        except Exception as e:
            logger.error(f"Error extracting transcript data: {str(e)}")
            raise
    
    def extract_sentences(self, data):
        """Extract sentence-level transcript from the JSON data."""
        try:
            channels = data['data'][1]['channels_present']
            if 'sentence_level_transcript' not in channels:
                raise ValueError("sentence_level_transcript channel not found")
            
            return data['data'][2]['channels']['sentence_level_transcript']
        except Exception as e:
            logger.error(f"Error extracting sentences: {str(e)}")
            raise
    
    def extract_words(self, data):
        """Extract word-level transcript from the JSON data."""
        try:
            channels = data['data'][1]['channels_present']
            if 'word_level_transcripts' not in channels:
                raise ValueError("word_level_transcript channel not found")
            
            return data['data'][2]['channels']['word_level_transcripts']
        except Exception as e:
            logger.error(f"Error extracting words: {str(e)}")
            raise
    
    def get_chapter_transcript(self, transcript_data, chapter):
        """Extract transcript content for a specific chapter."""
        chapter_start = float(chapter['start'])
        chapter_end = float(chapter['end'])
        
        # Filter sentences for this chapter
        chapter_sentences = [
            sentence for sentence in transcript_data['sentences']
            if float(sentence['start']) >= chapter_start and float(sentence['end']) <= chapter_end
        ]
        
        # Filter words for this chapter (if needed for detailed analysis)
        chapter_words = [
            word for word in transcript_data['words']
            if float(word['start']) >= chapter_start and float(word['end']) <= chapter_end
        ]
        
        return {
            "sentences": chapter_sentences,
            "words": chapter_words,
            "text": " ".join([sentence['text'] for sentence in chapter_sentences])
        }
    
    def call_openai_with_retry(self, messages, model="gpt-4o", temperature=0.7, response_format=None):
        """Call OpenAI API with exponential backoff retry logic."""
        retry_count = 0
        while retry_count < MAX_RETRIES:
            try:
                # Track API calls
                self.stats["api_calls"] += 1
                
                if response_format:
                    response = self.openai_client.chat.completions.create(
                        model=model, 
                        messages=messages, 
                        temperature=temperature,
                        response_format=response_format
                    )
                else:
                    response = self.openai_client.chat.completions.create(
                        model=model, 
                        messages=messages, 
                        temperature=temperature
                    )
                
                return response
            except Exception as e:
                retry_count += 1
                wait_time = BACKOFF_FACTOR ** retry_count
                logger.warning(f"API call failed: {str(e)}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                if retry_count == MAX_RETRIES:
                    logger.error(f"Max retries reached. Last error: {str(e)}")
                    raise
    
    def generate_broll_suggestions(self, chapter, chapter_transcript):
        """Generate B-roll suggestions for a chapter using LLM."""
        logger.info(f"Generating B-roll suggestions for chapter: {chapter['title']}")
        
        system_prompt = """You are an expert video editor analyzing podcast transcripts to identify segments where B-roll footage, images, or text overlays would enhance viewer understanding.

## YOUR TASK
For each meaningful segment in the transcript, determine:
1. If B-roll would enhance viewer understanding or engagement
2. What type of B-roll would work best (video, image, or text)
3. A specific keyword for searching for relevant footage/images

## TYPES OF B-ROLL
1. VIDEO - For concepts that involve movement, processes, or demonstrations
2. IMAGE - For static concepts, diagrams, portraits, or specific examples
3. TEXT - For quotes, statistics, key points, or definitions

## CRITERIA FOR B-ROLL
You SHOULD suggest B-roll for:
- Explanations of complex concepts
- Descriptions of physical objects, places, or people
- References to specific visual examples
- Processes or demonstrations
- Key statistics or quotes
- Important definitions or technical terms

You should NOT suggest B-roll for:
- General conversation
- Personal opinions without concrete subjects
- Abstract discussions without clear visual elements
- Short transitional statements

## CHAPTER CONTEXT
Chapter Title: {chapter_title}
Chapter Duration: {chapter_duration} seconds

## OUTPUT FORMAT
Return a JSON array named "suggestions" with objects containing:
- segment_start: Start time of segment (float)
- segment_end: End time of segment (float)
- text: The transcript text for this segment
- type: B-roll type ("VIDEO", "IMAGE", or "TEXT")
- keyword: A specific search keyword for videos/images (e.g., "mars rover" not just "mars")
- description: Why B-roll would enhance this segment
- alt_keywords: Array of 2-3 alternative keywords if primary doesn't yield good results
- confidence: How confident you are this segment needs B-roll (1-10)

Focus on SPECIFICITY in your keyword suggestions - provide detailed, specific keywords that will yield good search results.
"""

        user_prompt = f"""Analyze this podcast transcript chapter and identify segments that would benefit from B-roll.

Chapter Title: {chapter['title']}
Transcript:
{chapter_transcript['text']}

Remember to:
1. Only suggest B-roll for segments where it genuinely enhances understanding
2. Choose the most appropriate type (VIDEO/IMAGE/TEXT) based on content
3. Provide VERY SPECIFIC keywords for searching (e.g. "laboratory scientist pipette" not just "science")
4. Include start and end timestamps exactly as they appear in the transcript
5. Focus on the 3-5 most important segments if the chapter is long
"""

        messages = [
            {"role": "system", "content": system_prompt.format(
                chapter_title=chapter['title'], 
                chapter_duration=float(chapter['end']) - float(chapter['start'])
            )},
            {"role": "user", "content": user_prompt},
        ]

        try:
            logger.info(f"Running LLM analysis for chapter {chapter['title']}...")
            response = self.call_openai_with_retry(
                messages=messages, 
                model="gpt-4o",
                temperature=0.7, 
                response_format={"type": "json_object"}
            )

            json_response = response.choices[0].message.content
            logger.info(f"Raw response from OpenAI: {json_response}")
            
            try:
                result = json.loads(json_response)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.error(f"Raw response: {json_response}")
                return []
            
            # Make sure we have a list of suggestions
            if isinstance(result, dict) and "suggestions" in result:
                suggestions = result["suggestions"]
            elif "b_roll_suggestions" in result:
                suggestions = result["b_roll_suggestions"]
            elif isinstance(result, list):
                suggestions = result
            else:
                logger.warning(f"Unexpected response format: {result}")
                # Try to extract suggestions from any format
                if isinstance(result, dict):
                    for key, value in result.items():
                        if isinstance(value, list) and len(value) > 0:
                            suggestions = value
                            break
                    else:
                        suggestions = []
                else:
                    suggestions = []
            
            # Post-process suggestions
            for suggestion in suggestions:
                # Ensure required fields
                if "segment_start" not in suggestion or "segment_end" not in suggestion:
                    for sentence in chapter_transcript['sentences']:
                        if sentence['text'] in suggestion.get('text', ''):
                            suggestion['segment_start'] = float(sentence['start'])
                            suggestion['segment_end'] = float(sentence['end'])
                            break
                
                # Convert string timestamps to floats if needed
                if isinstance(suggestion.get('segment_start'), str):
                    suggestion['segment_start'] = float(suggestion['segment_start'])
                if isinstance(suggestion.get('segment_end'), str):
                    suggestion['segment_end'] = float(suggestion['segment_end'])
                
                # Add chapter info
                suggestion['chapter_title'] = chapter['title']
                suggestion['chapter_start'] = float(chapter['start'])
                suggestion['chapter_end'] = float(chapter['end'])
                
                # Add default B-roll type if missing
                if "type" not in suggestion:
                    suggestion["type"] = "VIDEO"
                
                # Default confidence if missing
                if "confidence" not in suggestion:
                    suggestion["confidence"] = 7
            
            # Update stats
            self.stats["broll_suggestions"] += len(suggestions)
            
            return suggestions
        
        except Exception as e:
            logger.error(f"Error in generate_broll_suggestions: {str(e)}")
            logger.exception("Full traceback:")
            return []
    
    def fetch_media_for_suggestion(self, suggestion):
        """Fetch appropriate media based on the B-roll type."""
        media_type = suggestion.get("type", "VIDEO").upper()
        keyword = suggestion.get("keyword", "")
        
        try:
            if media_type == "VIDEO":
                media_data = self.fetch_pexels_video(keyword)
            elif media_type == "IMAGE":
                media_data = self.fetch_image(keyword)
            elif media_type == "TEXT":
                media_data = self.generate_text_overlay(suggestion)
            else:
                logger.warning(f"Unknown B-roll type: {media_type}")
                media_data = None
                
            # If primary keyword fails, try alternatives
            if not media_data or media_data == "Invalid keyword":
                alt_keywords = suggestion.get("alt_keywords", [])
                for alt_keyword in alt_keywords:
                    logger.info(f"Trying alternative keyword: {alt_keyword}")
                    if media_type == "VIDEO":
                        media_data = self.fetch_pexels_video(alt_keyword)
                    elif media_type == "IMAGE":
                        media_data = self.fetch_image(alt_keyword)
                    
                    if media_data and media_data != "Invalid keyword":
                        break
            
            return media_data
        
        except Exception as e:
            logger.error(f"Error fetching media for '{keyword}': {str(e)}")
            return None
    
    def fetch_pexels_video(self, keyword, orientation="landscape", per_page=DEFAULT_PEXELS_PAGE_SIZE, quality=["hd", "sd"]):
        """Fetch video from Pexels API with improved parameters."""
        if not keyword or not self.pexels_api_key:
            return "Invalid keyword or missing API key"
            
        url = f"https://api.pexels.com/videos/search?query={keyword}&orientation={orientation}&per_page={per_page}&size=medium"
        headers = {"Authorization": self.pexels_api_key}
        
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
                                'type': 'VIDEO',
                                'url': video_file['link'], 
                                'thumbnail': video_info['image'],
                                'duration': video_info['duration'],
                                'width': video_file['width'],
                                'height': video_file['height'],
                                'source': 'pexels',
                                'source_id': video_info['id'],
                                'source_url': video_info.get('url')
                            }
                    
                # If no suitable quality found in any video, use the first video's first file
                return {
                    'type': 'VIDEO',
                    'url': data['videos'][0]['video_files'][0]['link'],
                    'thumbnail': data['videos'][0]['image'],
                    'duration': data['videos'][0]['duration'],
                    'width': data['videos'][0]['video_files'][0].get('width', 1920),
                    'height': data['videos'][0]['video_files'][0].get('height', 1080),
                    'source': 'pexels',
                    'source_id': data['videos'][0]['id'],
                    'source_url': data['videos'][0].get('url')
                }
            else:
                return "Invalid keyword"
        except Exception as e:
            logger.error(f"Error fetching video for keyword '{keyword}': {e}")
            return "API error"
    
    def fetch_image(self, keyword, per_page=DEFAULT_UNSPLASH_PAGE_SIZE):
        """Fetch image from Unsplash or Pexels API."""
        # Try Unsplash first if API key is available
        if self.unsplash_access_key:
            image_data = self.fetch_unsplash_image(keyword, per_page)
            if image_data and image_data != "Invalid keyword" and image_data != "API error":
                return image_data
        
        # Fall back to Pexels for images
        if self.pexels_api_key:
            return self.fetch_pexels_image(keyword, per_page)
        
        return "Missing API keys"
    
    def fetch_unsplash_image(self, keyword, per_page=DEFAULT_UNSPLASH_PAGE_SIZE):
        """Fetch image from Unsplash API."""
        if not keyword or not self.unsplash_access_key:
            return "Invalid keyword or missing API key"
            
        url = f"https://api.unsplash.com/search/photos?query={keyword}&per_page={per_page}"
        headers = {"Authorization": f"Client-ID {self.unsplash_access_key}"}
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            if 'results' in data and len(data['results']) > 0:
                image = data['results'][0]
                return {
                    'type': 'IMAGE',
                    'url': image['urls']['regular'],
                    'thumbnail': image['urls']['thumb'],
                    'width': image['width'],
                    'height': image['height'],
                    'source': 'unsplash',
                    'source_id': image['id'],
                    'source_url': image['links']['html'],
                    'photographer': image['user']['name'],
                    'photographer_url': image['user']['links']['html']
                }
            else:
                return "Invalid keyword"
        except Exception as e:
            logger.error(f"Error fetching Unsplash image for keyword '{keyword}': {e}")
            return "API error"
    
    def fetch_pexels_image(self, keyword, per_page=DEFAULT_PEXELS_PAGE_SIZE):
        """Fetch image from Pexels API."""
        if not keyword or not self.pexels_api_key:
            return "Invalid keyword or missing API key"
            
        url = f"https://api.pexels.com/v1/search?query={keyword}&per_page={per_page}"
        headers = {"Authorization": self.pexels_api_key}
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            if 'photos' in data and len(data['photos']) > 0:
                photo = data['photos'][0]
                return {
                    'type': 'IMAGE',
                    'url': photo['src']['large'],
                    'thumbnail': photo['src']['small'],
                    'width': photo['width'],
                    'height': photo['height'],
                    'source': 'pexels',
                    'source_id': photo['id'],
                    'source_url': photo['url'],
                    'photographer': photo['photographer'],
                    'photographer_url': photo['photographer_url']
                }
            else:
                return "Invalid keyword"
        except Exception as e:
            logger.error(f"Error fetching Pexels image for keyword '{keyword}': {e}")
            return "API error"
    
    def generate_text_overlay(self, suggestion):
        """Generate text overlay content."""
        segment_text = suggestion.get('text', '')
        keyword = suggestion.get('keyword', '')
        
        # For TEXT type, we generate content based on the segment and keyword
        system_prompt = """You are an expert at creating concise, impactful text overlays for videos.
Your task is to create a text overlay for a podcast segment that will appear on screen.

Guidelines:
1. Keep text concise (max 15 words)
2. Focus on key facts, statistics, or quotable statements
3. Format it attractively for on-screen display
4. Use factual, clear language
5. Include attribution if it's a quote
"""

        user_prompt = f"""Create a text overlay for this podcast segment:

"{segment_text}"

Focus keyword: {keyword}

The overlay should highlight the most important point or quote from this segment.
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = self.call_openai_with_retry(
                messages=messages, 
                model="gpt-3.5-turbo",
                temperature=0.7
            )

            text_content = response.choices[0].message.content.strip()
            # Remove quotes if they're just wrapping the whole text
            if text_content.startswith('"') and text_content.endswith('"'):
                text_content = text_content[1:-1]
                
            return {
                'type': 'TEXT',
                'url': None,  # No URL for text overlays
                'text_content': text_content,
                'source': 'openai',
                'keyword': keyword
            }
        
        except Exception as e:
            logger.error(f"Error generating text overlay: {str(e)}")
            return {
                'type': 'TEXT',
                'url': None,
                'text_content': keyword,  # Fallback to using the keyword
                'source': 'fallback'
            }
    
    def process_chapter(self, chapter_index, chapter, transcript_data):
        """Process a single chapter for B-roll suggestions."""
        logger.info(f"Processing chapter {chapter_index}: {chapter['title']}")
        
        start_time = time.time()
        
        # Get transcript for this chapter
        chapter_transcript = self.get_chapter_transcript(transcript_data, chapter)
        
        # Generate B-roll suggestions
        suggestions = self.generate_broll_suggestions(chapter, chapter_transcript)
        
        # Fetch media for each suggestion
        for suggestion in suggestions:
            try:
                logger.info(f"Fetching media for suggestion: {suggestion.get('keyword', '')}")
                media_data = self.fetch_media_for_suggestion(suggestion)
                
                if media_data and media_data != "Invalid keyword" and media_data != "API error":
                    # Attach media data to suggestion
                    suggestion['media_data'] = media_data
                    suggestion['media_found'] = True
                else:
                    suggestion['media_found'] = False
                    suggestion['media_error'] = str(media_data) if isinstance(media_data, str) else "Unknown error"
            except Exception as e:
                logger.error(f"Error processing media for suggestion: {str(e)}")
                suggestion['media_found'] = False
                suggestion['media_error'] = str(e)
        
        processing_time = time.time() - start_time
        
        return {
            "chapter_index": chapter_index,
            "chapter_title": chapter['title'],
            "chapter_start": float(chapter['start']),
            "chapter_end": float(chapter['end']),
            "suggestions": suggestions,
            "processing_time": processing_time
        }
    
    def process_podcast(self, input_path, output_path=None):
        """Process a podcast transcript file with B-roll suggestion generation."""
        logger.info(f"Processing podcast file: {input_path}")
        
        # Reset stats for this run
        self.stats = {
            "api_calls": 0,
            "tokens_used": 0,
            "processing_time": 0,
            "chapters_processed": 0,
            "broll_suggestions": 0
        }
        
        overall_start_time = time.time()
        
        # Load input data
        input_data = self.load_json_data(input_path)
        
        # Extract transcript components
        transcript_data = self.extract_transcript_data(input_data)
        chapters = transcript_data['chapters']
        
        # Process chapters in parallel
        results = []
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all chapters for processing
            future_to_chapter = {
                executor.submit(self.process_chapter, i, chapter, transcript_data): (i, chapter)
                for i, chapter in enumerate(chapters)
            }
            
            # Process results as they complete
            for future in as_completed(future_to_chapter):
                i, chapter = future_to_chapter[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Completed processing for chapter {i}: {chapter['title']}")
                except Exception as e:
                    logger.error(f"Error processing chapter {i}: {chapter['title']} - {str(e)}")
                    results.append({
                        "chapter_index": i,
                        "chapter_title": chapter["title"],
                        "chapter_start": float(chapter['start']),
                        "chapter_end": float(chapter['end']),
                        "suggestions": [],
                        "error": str(e)
                    })
        
        # Sort results by chapter index
        results.sort(key=lambda x: x["chapter_index"])
        
        # Calculate total processing time
        self.stats["processing_time"] = time.time() - overall_start_time
        self.stats["chapters_processed"] = len(chapters)
        
        logger.info(f"Total processing time: {self.stats['processing_time']:.2f} seconds")
        logger.info(f"Total API calls: {self.stats['api_calls']}")
        logger.info(f"Total B-roll suggestions: {self.stats['broll_suggestions']}")
        
        # Prepare output data
        output_data = {
            "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "input_file": input_path,
            "stats": self.stats,
            "chapters": results
        }
        
        # Save results if output path is provided
        if output_path:
            with open(output_path, 'w') as file:
                json.dump(output_data, file, indent=2)
            
            logger.info(f"Results saved to: {output_path}")
        
        return output_data
    
    def flatten_results_for_legacy_compatibility(self, results):
        """Convert chapter-based results to flat segment list for compatibility."""
        flat_segments = []
        
        for chapter in results["chapters"]:
            for suggestion in chapter.get("suggestions", []):
                if suggestion.get("media_found", False):
                    media_data = suggestion.get("media_data", {})
                    
                    segment = {
                        'start': suggestion['segment_start'],
                        'end': suggestion['segment_end'],
                        'text': suggestion.get('text', ''),
                        'use_broll': True,
                        'broll_type': suggestion.get('type', 'VIDEO'),
                        'keyword': suggestion.get('keyword', ''),
                        'chapter_title': chapter.get('chapter_title', '')
                    }
                    
                    # Add media-specific fields
                    if media_data.get('type') == 'VIDEO':
                        segment['broll_url'] = media_data.get('url')
                        segment['thumbnail'] = media_data.get('thumbnail')
                        segment['duration'] = media_data.get('duration')
                    elif media_data.get('type') == 'IMAGE':
                        segment['image_url'] = media_data.get('url')
                        segment['thumbnail'] = media_data.get('thumbnail')
                    elif media_data.get('type') == 'TEXT':
                        segment['text_overlay'] = media_data.get('text_content')
                    
                    flat_segments.append(segment)
        
        return flat_segments

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='B-roll Generator for Podcasts')
    parser.add_argument('-i', '--input', required=True, help='Input JSON file path')
    parser.add_argument('-o', '--output', help='Output JSON file path (optional)')
    parser.add_argument('-w', '--workers', type=int, default=MAX_WORKERS, help=f'Number of worker threads (default: {MAX_WORKERS})')
    parser.add_argument('--pexels-key', help='Pexels API key (overrides env var)')
    parser.add_argument('--unsplash-access-key', help='Unsplash access key (overrides env var)')
    parser.add_argument('--unsplash-secret-key', help='Unsplash secret key (overrides env var)')
    parser.add_argument('--legacy-output', action='store_true', help='Generate legacy flat output format')
    return parser.parse_args()

def main():
    """Main function to run the B-roll generator."""
    # Parse arguments
    args = parse_arguments()
    
    # Initialize generator
    generator = BrollGenerator(
        pexels_key=args.pexels_key,
        unsplash_access_key=args.unsplash_access_key,
        unsplash_secret_key=args.unsplash_secret_key
    )
    
    # Process podcast
    results = generator.process_podcast(args.input, args.output)
    
    # Create legacy output if requested
    if args.legacy_output:
        legacy_output = generator.flatten_results_for_legacy_compatibility(results)
        legacy_output_path = args.output.replace('.json', '_legacy.json') if args.output else 'broll_legacy_output.json'
        with open(legacy_output_path, 'w') as file:
            json.dump(legacy_output, file, indent=2)
        logger.info(f"Legacy format results saved to: {legacy_output_path}")
    
    # Print summary
    print("\nProcessing Summary:")
    print(f"Chapters processed: {results['stats']['chapters_processed']}")
    print(f"Total B-roll suggestions: {results['stats']['broll_suggestions']}")
    print(f"Total processing time: {results['stats']['processing_time']:.2f} seconds")
    
    for chapter in results['chapters']:
        print(f"\nChapter: {chapter['chapter_title']}")
        print(f"  Suggestions: {len(chapter['suggestions'])}")
        print(f"  Processing time: {chapter['processing_time']:.2f} seconds")

if __name__ == "__main__":
    main() 