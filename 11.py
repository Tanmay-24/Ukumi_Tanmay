from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from dotenv import load_dotenv
import json
import os

load_dotenv()

class RemovedSegment(BaseModel):
    """Structured output for removed segments"""
    start: float = Field(description="Start timestamp of removed segment")
    end: float = Field(description="End timestamp of removed segment")

    @field_validator('start', 'end')
    def validate_timestamps(cls, v):
        if v < 0:
            raise ValueError("Timestamps must be non-negative")
        return v

class RemovedSegmentsOutput(BaseModel):
    """Output structure for removed segments"""
    removed: List[RemovedSegment]

class TranscriptProcessor:
    def __init__(self):
        self.segment_generator = Agent(
            model=OpenAIChat(id="o3-mini"),
            description="Podcast Transcript Cleaning Specialist",
            response_model=RemovedSegmentsOutput,
            instructions="""You are a Podcast Transcript Cleaning and Optimization Specialist.
With the combined expertise of a meticulous Video Editor and an engaged Viewer.

Your task is to analyze the provided transcript and
identify segments that should be removed to enhance the natural flow of the podcast video
without compressing or over-densifying the conversation.

Guidelines:
1. REMOVAL OF NON-ESSENTIAL CONTENT:
   - Identify segments that do not contribute meaningfully to the conversation(water breaks,mic checks,equipment tests)
   - Preserve core content, including complete question answer pairs and any substantive dialogue.
   - If an answer spans multiple segments, ensure the entire answer remains intact.
   - DO NOT REMOVE the podcast intro, guest intro, speaker intro, etc

2. TIMESTAMP ACCURACY:
   - Use the exact timestamps from the original transcript.
   - If multiple consecutive unwanted segments occur, merge them into a single removal segment.
   - Ensure a minimum granularity of 1 second and that all timestamps are in seconds.

3. NATURAL EDITING:
   - Follow natural, human-like editing practices so that the pacing of the conversation is maintained.
   - Adjust segmentation according to the variable length of the video to preserve its natural flow.
   - Keep the parts where speaker or guest give their intro, podcast intro, topic intro of podcast etc
   - Keep the ending remarks given by host and the guest in the final output
   - Podcast generally follow many type for format such as interview style, discussion style etc
   - use this knowledge to give better outputs
"""
        )

    def parse_word_level_transcript(self, word_transcript: dict) -> List[dict]:
        """Convert word-level transcript to structured format"""
        words = []
        for word in word_transcript.get('text', []):
            words.append({
                'start': float(word.get('start', 0)),
                'end': float(word.get('end', 0)),
                'text': word.get('text', '')
            })
        return words

    def find_precise_word_timing(self, start_time: float, end_time: float, 
                               words: List[dict]) -> Optional[tuple]:
        """Find precise start and end times within word-level transcript"""
        precise_start = None
        precise_end = None

        # Find closest word start time
        for word in words:
            if word['start'] <= start_time <= word['end']:
                precise_start = word['start']
                break
            if word['start'] > start_time:
                precise_start = word['start']
                break

        # Find closest word end time
        for word in reversed(words):
            if word['start'] <= end_time <= word['end']:
                precise_end = word['end']
                break
            if word['end'] < end_time:
                precise_end = word['end']
                break

        if precise_start is not None and precise_end is not None:
            return (precise_start, precise_end)
        return None

    def process_chapter(self, 
                       chapter: dict, 
                       sentences: List[dict], 
                       words: List[dict]) -> List[RemovedSegment]:
        """Process a single chapter to identify segments for removal"""
        # Collect relevant sentences for this chapter
        chapter_sentences = []
        for sent in sentences:
            if (sent['start'] >= chapter['start'] and 
                sent['end'] <= chapter['end']):
                chapter_sentences.append(sent)
        
        # Combine chapter content
        chapter_content = {
            'title': chapter['title'],
            'description': chapter.get('description', ''),
            'content': '\n'.join(sent['text'] for sent in chapter_sentences)
        }

        # Generate removal suggestions
        result = self.segment_generator.run(
            f"""Analyze this chapter:
            Title: {chapter_content['title']}
            Description: {chapter_content['description']}
            Content: {chapter_content['content']}""",
            max_retries=3,
            validation_error_prompt="Fix JSON formatting and timestamp validation"
        )

        # Refine timestamps using word-level transcript
        refined_segments = []
        for segment in result.content.model_dump()["removed"]:
            precise_timing = self.find_precise_word_timing(
                segment['start'],
                segment['end'],
                words
            )
            if precise_timing:
                refined_segment = RemovedSegment(
                    start=precise_timing[0],
                    end=precise_timing[1]
                )
                refined_segments.append(refined_segment)

        return refined_segments

    def process_transcript(self, 
                         chapter_json_path: str, 
                         sentence_json_path: str, 
                         word_json_path: str, 
                         output_path: str):
        """Process complete transcript"""
        try:
            # Load all transcript files
            with open(chapter_json_path, 'r') as f:
                chapter_data = json.load(f)
                chapters = chapter_data[0]['chapters']

            with open(sentence_json_path, 'r') as f:
                sentence_data = json.load(f)
                sentences = sentence_data[0]['text']

            with open(word_json_path, 'r') as f:
                word_data = json.load(f)
                words = self.parse_word_level_transcript(word_data[0])

            all_segments = []
            
            # Process each chapter
            for chapter in chapters:
                chapter_segments = self.process_chapter(
                    chapter, 
                    sentences, 
                    words
                )
                all_segments.extend(chapter_segments)

            # Sort by timestamp
            all_segments.sort(key=lambda x: x.start)

            # Save results
            with open(output_path, 'w') as f:
                json.dump({
                    "target_id": chapter_data[0].get('target_id'),
                    "media_id": chapter_data[0].get('media_id', {}).get('$oid'),
                    "removed": [segment.model_dump() for segment in all_segments]
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

    processor = TranscriptProcessor()
    
    try:
        print("Starting transcript processing...")
        processor.process_transcript(
            chapter_json_path="/home/tanmay/Desktop/Ukumi_Tanmay/output/chapters.json",
            sentence_json_path="/home/tanmay/Desktop/Ukumi_Tanmay/output/sentences.json",
            word_json_path="/home/tanmay/Desktop/Ukumi_Tanmay/output/words.json",
            output_path="/home/tanmay/Desktop/Ukumi_Tanmay/output/segments.json"
        )
        print(f"Successfully processed transcript")
        
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds")
    except Exception as e:
        print(f"Error processing transcript: {str(e)}")
        raise