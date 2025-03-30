"""
Text-Based B-roll Generator for Podcast Videos (v0.1)

This script provides a simple workflow for generating text-based B-roll suggestions for podcast videos.
It processes a podcast transcript and generates text overlay suggestions with alternatives.
Version 0.1 now includes caption highlighting and word-level timestamp accuracy.

Usage:
    python enhanced_broll_generator.py -i input.json -o output.json
"""

import json
import logging
import time
import argparse
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("text_broll")

# Constants
MAX_RETRIES = 3
ALTERNATIVE_OPTIONS_COUNT = 3

# API Key


class TextBrollGenerator:
    """A simple class to generate text-based B-roll suggestions for podcast videos."""
    
    def __init__(self, openai_key=None):
        """Initialize with OpenAI API key."""
        self.openai_client = OpenAI(api_key=openai_key or OPENAI_API_KEY)
    
    def load_json_data(self, file_path):
        """Load and parse JSON data from a file."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading JSON data: {str(e)}")
            raise
    
    def extract_transcript_data(self, data):
        """Extract transcript data including chapters, sentences, and words."""
        try:
            channels = data['data'][1]['channels_present']
            
            # Check for chapters
            has_chapters = 'chapter_data' in channels
            
            # Check for sentence-level transcript
            has_sentence_transcript = 'sentence_level_transcript' in channels
            
            # Check for word-level transcript
            has_word_transcript = 'word_level_transcripts' in channels
            
            result = {}
            
            # Extract chapters
            if has_chapters:
                result["chapters"] = data['data'][2]['channels']['chapter_data']
            
            # Extract sentence-level transcript
            if has_sentence_transcript:
                result["sentences"] = data['data'][2]['channels']['sentence_level_transcript']
            
            # Extract word-level transcript
            if has_word_transcript:
                result["words"] = data['data'][2]['channels']['word_level_transcripts']
            
            # If no chapters, create one chapter for the entire content
            if not has_chapters and (has_sentence_transcript or has_word_transcript):
                if has_sentence_transcript:
                    sentences = result["sentences"]
                    result["chapters"] = [{
                        "title": "Full Podcast",
                        "start": float(sentences[0]['start']),
                        "end": float(sentences[-1]['end'])
                    }]
                elif has_word_transcript:
                    words = result["words"]
                    result["chapters"] = [{
                        "title": "Full Podcast",
                        "start": float(words[0]['start']),
                        "end": float(words[-1]['end'])
                    }]
            
            # Validate that we have necessary data
            if "chapters" not in result:
                raise ValueError("No chapter data found")
                
            if not has_sentence_transcript and not has_word_transcript:
                raise ValueError("No transcript data found (neither sentence nor word level)")
                
            return result
            
        except Exception as e:
            logger.error(f"Error extracting transcript data: {str(e)}")
            raise
    
    def get_chapter_transcript(self, transcript_data, chapter):
        """Extract transcript content for a specific chapter with word-level precision if available."""
        chapter_start = float(chapter['start'])
        chapter_end = float(chapter['end'])
        
        result = {
            "text": "",
            "word_segments": []
        }
        
        # First try using word-level transcript for greater precision
        if "words" in transcript_data:
            word_segments = []
            chapter_words = []
            
            # Find words that fall within this chapter
            for word in transcript_data["words"]:
                word_start = float(word['start'])
                word_end = float(word['end'])
                
                if word_start >= chapter_start and word_end <= chapter_end:
                    chapter_words.append(word['text'])
                    word_segments.append({
                        "text": word['text'],
                        "start": word_start,
                        "end": word_end
                    })
            
            result["text"] = " ".join(chapter_words)
            result["word_segments"] = word_segments
            
        # Fall back to sentence-level if word-level not available
        elif "sentences" in transcript_data:
            chapter_sentences = [
                sentence for sentence in transcript_data['sentences']
                if float(sentence['start']) >= chapter_start and float(sentence['end']) <= chapter_end
            ]
            
            result["text"] = " ".join([sentence['text'] for sentence in chapter_sentences])
            result["sentences"] = chapter_sentences
        
        return result
    
    def get_precise_timestamps(self, text, word_segments):
        """Get precise start and end timestamps for a piece of text using word-level data."""
        # Simple sliding window to find the closest match
        text_words = text.split()
        text_length = len(text_words)
        
        best_match = None
        best_match_score = 0
        
        for i in range(len(word_segments) - text_length + 1):
            segment_window = word_segments[i:i + text_length]
            segment_text = " ".join([w["text"] for w in segment_window])
            
            # Compute similarity score (simple word overlap for now)
            common_words = set(segment_text.lower().split()) & set(text.lower().split())
            score = len(common_words)
            
            if score > best_match_score:
                best_match_score = score
                best_match = {
                    "start": segment_window[0]["start"],
                    "end": segment_window[-1]["end"],
                    "text": segment_text
                }
        
        # If no good match found, return None
        if best_match_score < len(text_words) / 2:
            return None
            
        return best_match
    
    def call_openai(self, messages, model="gpt-4o", temperature=0.5):
        """Call OpenAI API."""
        try:
            response = self.openai_client.chat.completions.create(
                model=model, 
                messages=messages, 
                temperature=temperature,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            return None
    
    def find_text_opportunities(self, chapter, chapter_transcript):
        """Find opportunities for text overlays in the transcript."""
        system_prompt = """You are an expert video editor. Find segments in this podcast transcript where text overlays would enhance viewer understanding.

Focus on:
- Key statistics or numbers
- Important quotes
- Technical terms or definitions
- Key points or takeaways
- Important dates or facts
- Emotional or impactful statements

For each segment, identify the highlight type from these categories:
- key_insight (for important realizations or perspectives)
- important_point (for main arguments or facts)
- technical_concept (for specialized terms or concepts)
- emotional_moment (for high impact emotional statements)
- key_statistic (for numbers, percentages, or data)
- practical_advice (for actionable recommendations)

Return a JSON array of objects with:
- text: The transcript text that should be highlighted
- idea: Why this needs a text overlay
- highlight_type: The type of highlight from the categories above
"""

        user_prompt = f"""Find text overlay opportunities in this podcast segment:

Chapter: {chapter['title']}
Content: {chapter_transcript['text']}

Return only the best 3-5 opportunities with appropriate highlight types."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        result = self.call_openai(messages)
        if not result or "opportunities" not in result:
            return []
            
        return result["opportunities"]
    
    def generate_text_options(self, segment_text, highlight_type="key_insight"):
        """Generate text overlay options for a segment with caption highlighting for the primary option."""
        system_prompt = """You are an expert at creating text overlays for videos.
Create 3 versions of a text overlay for this segment:

1. CAPTION HIGHLIGHT: Create a highlighted caption that preserves the exact spoken words but marks the most important part with *asterisks* (primary option)
2. STANDARD TEXT: Make a concise text overlay under 10 words that focuses on clarity and impact
3. ALTERNATIVE TEXT: Create another creative version with different wording but equal impact

For caption highlights:
- Maintain the exact wording from the transcript 
- Use *asterisks* to mark ONLY the key phrases (usually 3-7 words)
- Don't highlight the entire text, only the most impactful part

Return a JSON object with:
- text1: Caption highlight with *asterisks* around key phrases
- text2: Standard concise text overlay
- text3: Alternative creative version"""

        user_prompt = f"""Create text overlay versions for this segment:
"{segment_text}"

Highlight Type: {highlight_type}

1. For text1, keep the spoken text but mark the key parts with *asterisks*
2. For text2, make a concise version that is easily readable on screen
3. For text3, create a creative alternative that captures the same idea"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        result = self.call_openai(messages)
        if not result:
            # Create a simple highlight if API fails
            words = segment_text.split()
            if len(words) > 6:
                # Highlight the middle part if text is long enough
                middle_start = max(0, len(words) // 3)
                middle_end = min(len(words), middle_start + 3)
                highlighted_words = words.copy()
                highlighted_words[middle_start] = "*" + highlighted_words[middle_start]
                highlighted_words[middle_end-1] = highlighted_words[middle_end-1] + "*"
                text1 = " ".join(highlighted_words)
            else:
                # Highlight the whole text if it's short
                text1 = "*" + segment_text + "*"
                
            return {
                "text1": text1,
                "text2": "Key point: " + segment_text[:40] + "...",
                "text3": "Important: " + segment_text[:40] + "..."
            }
            
        return result
    
    def _get_highlight_color(self, highlight_type):
        """Get appropriate highlight color based on highlight type."""
        color_map = {
            "key_insight": "#FFD700",      # Gold
            "important_point": "#FF6347",  # Tomato
            "technical_concept": "#4682B4", # Steel Blue
            "emotional_moment": "#DA70D6", # Orchid
            "key_statistic": "#32CD32",    # Lime Green
            "practical_advice": "#FF8C00"  # Dark Orange
        }
        
        return color_map.get(highlight_type, "#FFFFFF")  # Default to white
        
    def process_podcast(self, input_path, output_path=None):
        """Process podcast transcript and generate text overlay suggestions."""
        logger.info(f"Processing podcast file: {input_path}")
        
        # Load and extract data
        input_data = self.load_json_data(input_path)
        transcript_data = self.extract_transcript_data(input_data)
        
        text_highlights = []
        
        # Process each chapter
        for chapter in transcript_data["chapters"]:
            chapter_transcript = self.get_chapter_transcript(transcript_data, chapter)
            
            # Find opportunities for text overlays
            opportunities = self.find_text_opportunities(chapter, chapter_transcript)
            
            # Generate text options for each opportunity
            for opp in opportunities:
                highlight_type = opp.get('highlight_type', 'key_insight')
                text = opp['text']
                
                # Generate text options
                text_options = self.generate_text_options(text, highlight_type)
                
                # Process caption highlight to extract highlighted portion
                caption_highlight = text_options.get('text1', '')
                
                # Calculate highlighted portion from asterisks
                highlighted_text = ""
                if '*' in caption_highlight:
                    parts = caption_highlight.split('*')
                    if len(parts) >= 3:
                        highlighted_text = parts[1]
                
                # Get precise timestamps if word-level transcript available
                timestamp_info = None
                show_for = ""
                
                if "word_segments" in chapter_transcript and chapter_transcript["word_segments"]:
                    timestamp_info = self.get_precise_timestamps(text, chapter_transcript["word_segments"])
                    
                if timestamp_info:
                    start_time = timestamp_info["start"]
                    end_time = timestamp_info["end"]
                    show_for = f"{start_time:.2f} - {end_time:.2f}"
                
                # Add to results with simplified format
                highlight = {
                    "chapter": chapter['title'],
                    "idea": opp['idea'],
                    "type": "TEXT",
                    "highlight_type": highlight_type,
                    "broll_text": {
                        "text1": caption_highlight,
                        "text2": text_options.get('text2', ''),
                        "text3": text_options.get('text3', '')
                    },
                    "highlight_color": self._get_highlight_color(highlight_type)
                }
                
                # Add show_for field if we have timestamp info
                if show_for:
                    highlight["show_for"] = show_for
                
                text_highlights.append(highlight)
        
        # Create output with new schema
        output_data = {"text_highlights": text_highlights}
            
        # Save results if output path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            logger.info(f"Results saved to: {output_path}")
        
        return text_highlights

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Text B-roll Generator with Caption Highlighting')
    parser.add_argument('-i', '--input', required=True, help='Input JSON file path')
    parser.add_argument('-o', '--output', help='Output JSON file path')
    parser.add_argument('--openai-key', help='OpenAI API key')
    args = parser.parse_args()
    
    generator = TextBrollGenerator(openai_key=args.openai_key)
    results = generator.process_podcast(args.input, args.output)
    
    print(f"\nProcessed {len(results)} text highlight suggestions")

if __name__ == "__main__":
    main() 