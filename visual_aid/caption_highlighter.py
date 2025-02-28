from typing import List
from pydantic import BaseModel, Field, field_validator
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from dotenv import load_dotenv
import json
import os

load_dotenv()

class CaptionHighlight(BaseModel):
    """Structured output for caption highlighting"""
    start_time: float = Field(description="Start timestamp to show highlight")
    end_time: float = Field(description="End timestamp to hide highlight")
    highlight_type: str = Field(description="Type of highlight (important_point, key_insight, emotional_moment, etc)")
    highlight_text: str = Field(description="Text to display as highlighted caption")
    speaker: str = Field(description="Speaker who said the highlighted text")

    @field_validator('start_time', 'end_time')
    def validate_timestamps(cls, v):
        if v < 0:
            raise ValueError("Timestamps must be non-negative")
        return v

class CaptionHighlightOutput(BaseModel):
    """Output structure for caption highlight segments"""
    highlights: List[CaptionHighlight]

class CaptionProcessor:
    def __init__(self):
        self.highlight_generator = Agent(
            model=OpenAIChat(id="gpt-4o", temperature=0.5),
            description="Caption Highlight Generator",
            response_model=CaptionHighlightOutput,
            instructions="""You are a Caption Highlight Specialist. Analyze the transcript to identify important statements or phrases that should be highlighted alongside normal captions to enhance viewer engagement.

            Analyze the transcript to identify:
            1. Important points or key takeaways
            2. Emotional or impactful statements
            3. Insightful comments or revelations
            4. Notable quotes that capture core ideas
            5. Technical terms or concepts that deserve emphasis
            
            Guidelines:
            1. Focus on statements that carry significant weight or meaning
            2. Identify content that viewers should pay special attention to
            3. Use the exact sentence timestamps for start and end times
            4. Ensure highlights represent the most impactful moments
            5. Include speaker information with each highlight
            6. Select highlights that work well displayed next to the speaker
            7. Choose complete sentences or key phrases within sentences
            
            Output JSON array of highlight segments:
            {{"highlights": [{{"start_time": 5.32, "end_time": 8.1, "highlight_type": "key_insight", "highlight_text": "The solution was right in front of us", "speaker": "John"}}]}}"""
        )

    def parse_sentence_transcript(self, sentence_transcript: dict) -> List[dict]:
        """Convert sentence-level transcript to structured format with speaker info"""
        sentences = []
        for sentence in sentence_transcript.get('text', []):
            # Extract speaker information if available, otherwise use default
            speaker = sentence.get('speaker', 'Unknown')
            if not speaker or speaker == "":
                speaker = 'Unknown'
                
            sentences.append({
                'start': float(sentence.get('start', 0)),
                'end': float(sentence.get('end', 0)),
                'text': sentence.get('text', ''),
                'speaker': speaker
            })
        return sentences

    def find_sentence_in_segment(self, sentence: dict, segment_start: float, segment_end: float) -> bool:
        """Check if a sentence falls within a segment's timeframe"""
        return (sentence['start'] >= segment_start and 
                sentence['end'] <= segment_end)

    def process_transcript_segment(self, 
                                 sentences: List[dict],
                                 segment_start: float,
                                 segment_end: float) -> List[CaptionHighlight]:
        # Collect relevant sentences for this segment
        segment_sentences = [
            sent for sent in sentences 
            if self.find_sentence_in_segment(sent, segment_start, segment_end)
        ]
        
        # Skip processing if no sentences in this segment
        if not segment_sentences:
            return []
            
        # Combine segment sentences with speaker info
        segment_content = []
        for sent in segment_sentences:
            segment_content.append(f"{sent['speaker']}: {sent['text']}")
        
        segment_text = "\n".join(segment_content)

        # Generate highlight suggestions
        result = self.highlight_generator.run(
            f"""Analyze this transcript segment and identify important statements to highlight:
            
            {segment_text}""",
            max_retries=3,
            validation_error_prompt="Fix JSON formatting and timestamp validation"
        )

        # Match highlights to sentences
        refined_highlights = []
        for highlight in result.content.model_dump()["highlights"]:
            highlight_text = highlight['highlight_text']
            
            # Find the sentence containing this highlight
            for sentence in segment_sentences:
                if highlight_text in sentence['text']:
                    refined_highlight = CaptionHighlight(
                        start_time=sentence['start'],
                        end_time=sentence['end'],
                        highlight_type=highlight['highlight_type'],
                        highlight_text=highlight_text,
                        speaker=sentence['speaker']
                    )
                    refined_highlights.append(refined_highlight)
                    break
            
        return refined_highlights

    def process_transcript(self, 
                         sentence_json_path: str, 
                         output_path: str,
                         segment_duration: float = 300.0):  # Process in 5-minute segments by default
        try:
            # Load sentence transcript file
            with open(sentence_json_path, 'r') as f:
                sentence_data = json.load(f)
                sentences = self.parse_sentence_transcript(sentence_data[0])

            # Get total duration
            if sentences:
                transcript_start = sentences[0]['start']
                transcript_end = sentences[-1]['end']
            else:
                print("No sentences found in transcript")
                return

            all_highlights = []
            
            # Process transcript in segments
            segment_start = transcript_start
            while segment_start < transcript_end:
                segment_end = min(segment_start + segment_duration, transcript_end)
                
                print(f"Processing segment from {segment_start:.2f}s to {segment_end:.2f}s")
                
                segment_highlights = self.process_transcript_segment(
                    sentences,
                    segment_start,
                    segment_end
                )
                all_highlights.extend(segment_highlights)
                
                segment_start = segment_end

            # Sort by timestamp
            all_highlights.sort(key=lambda x: x.start_time)

            # Remove duplicates
            unique_highlights = []
            seen_texts = set()
            
            for highlight in all_highlights:
                # Create a key using the highlight text and timestamps
                key = f"{highlight.highlight_text}_{highlight.start_time}_{highlight.end_time}"
                if key not in seen_texts:
                    seen_texts.add(key)
                    unique_highlights.append(highlight)

            # Save results
            with open(output_path, 'w') as f:
                json.dump({
                    "target_id": sentence_data[0].get('target_id'),
                    "media_id": sentence_data[0].get('media_id', {}).get('$oid') if isinstance(sentence_data[0].get('media_id'), dict) else sentence_data[0].get('media_id'),
                    "highlights": [h.model_dump() for h in unique_highlights]
                }, f, indent=2)

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {str(e)}")
        except KeyError as e:
            print(f"Missing expected key in JSON: {str(e)}")
        except Exception as e:
            print(f"Unexpected error processing transcript: {str(e)}")

if __name__ == "__main__":
    import time
    start_time = time.time()

    processor = CaptionProcessor()
    
    try:
        print("Starting caption highlight processing...")
        processor.process_transcript(
            sentence_json_path="/home/tanmay/Desktop/Ukumi_Tanmay/output/sen_AI.json",
            output_path="/home/tanmay/Desktop/Ukumi_Tanmay/output/caption_highlights.json"
        )
        print(f"Successfully processed caption highlights")
        
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds")
    except Exception as e:
        print(f"Error processing caption highlights: {str(e)}")
        raise