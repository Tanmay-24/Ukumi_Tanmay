from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from dotenv import load_dotenv
import json
import os

load_dotenv()

class WordEmphasis(BaseModel):
    """Structured output for word emphasis overlay"""
    insert_start: float = Field(description="Start timestamp to show overlay")
    insert_end: float = Field(description="End timestamp to hide overlay")
    concept_type: str = Field(description="Type of emphasis (key_concept, quote, insight, etc)")
    overlay_text: str = Field(description="Text to display as overlay")

    @field_validator('insert_start', 'insert_end')
    def validate_timestamps(cls, v):
        if v < 0:
            raise ValueError("Timestamps must be non-negative")
        return v

class WordEmphasisOutput(BaseModel):
    """Output structure for word emphasis segments"""
    emphases: List[WordEmphasis]

class TranscriptProcessor:
    def __init__(self):
        self.emphasis_generator = Agent(
            model=OpenAIChat(id="gpt-4o", temperature=0.5),
            description="Podcast Transcript Emphasis Generator",
            response_model=WordEmphasisOutput,
            instructions="""You are a Podcast Content Emphasis Specialist. Analyze the transcript to identify impactful words or phrases that should be emphasized as overlays to enhance viewer engagement.

            For each chapter, analyze its description and content to identify:
            1. Key themes mentioned in the chapter description
            2. Important quotes that represent the main ideas
            3. Emotional or impactful moments
            4. Technical or specific terms that deserve emphasis
            
            Guidelines:
            1. Emphasize words that align with chapter descriptions
            2. Focus on moments of insight, revelation, or emotion
            3. Use precise timestamps from the word-level transcript
            4. Ensure overlays don't overlap
            5. Prefer shorter (1-3 word) phrases for impact
            
            Output JSON array of emphasis segments:
            {{"emphases": [{{"insert_start": 5.32, "insert_end": 8.1, "concept_type": "key_concept", "overlay_text": "Innovation"}}, ...]}}"""
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

    def find_sentence_in_chapter(self, sentence: dict, chapter: dict) -> bool:
        """Check if a sentence falls within a chapter's timeframe"""
        return (sentence['start'] >= chapter['start'] and 
                sentence['end'] <= chapter['end'])

    def find_precise_word_timing(self, phrase: str, words: List[dict], 
                               start_time: float, end_time: float) -> Optional[tuple]:
        """Find precise start and end times for a phrase within word-level transcript"""
        phrase_words = phrase.lower().split()
        for i in range(len(words)):
            if (words[i]['start'] >= start_time and 
                words[i]['end'] <= end_time):
                # Try to match the phrase starting at this word
                matched_words = []
                for j in range(len(phrase_words)):
                    if (i + j < len(words) and 
                        phrase_words[j] in words[i + j]['text'].lower()):
                        matched_words.append(words[i + j])
                    else:
                        break
                
                if len(matched_words) == len(phrase_words):
                    return (matched_words[0]['start'], 
                           matched_words[-1]['end'])
        return None

    def process_chapter(self, 
                       chapter: dict, 
                       sentences: List[dict], 
                       words: List[dict]) -> List[WordEmphasis]:
        # Collect relevant sentences for this chapter
        chapter_sentences = [
            sent for sent in sentences 
            if self.find_sentence_in_chapter(sent, chapter)
        ]
        
        # Combine chapter description and sentences
        chapter_content = {
            'title': chapter['title'],
            'description': '\n'.join(chapter['description']),
            'content': '\n'.join(sent['text'] for sent in chapter_sentences)
        }

        # Generate emphasis suggestions
        result = self.emphasis_generator.run(
            f"""Analyze this chapter:
            Title: {chapter_content['title']}
            Description: {chapter_content['description']}
            Content: {chapter_content['content']}""",
            max_retries=3,
            validation_error_prompt="Fix JSON formatting and timestamp validation"
        )

        # Refine timestamps using word-level transcript
        refined_emphases = []
        for emphasis in result.content.model_dump()["emphases"]:
            precise_timing = self.find_precise_word_timing(
                emphasis['overlay_text'],
                words,
                chapter['start'],
                chapter['end']
            )
            if precise_timing:
                refined_emphasis = WordEmphasis(
                    insert_start=precise_timing[0],
                    insert_end=precise_timing[1],
                    concept_type=emphasis['concept_type'],
                    overlay_text=emphasis['overlay_text']
                )
                refined_emphases.append(refined_emphasis)

        return refined_emphases

    def process_transcript(self, 
                         chapter_json_path: str, 
                         sentence_json_path: str, 
                         word_json_path: str, 
                         output_path: str):
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

            all_emphases = []
            
            # Process each chapter
            for chapter in chapters:
                chapter_emphases = self.process_chapter(
                    chapter, 
                    sentences, 
                    words
                )
                all_emphases.extend(chapter_emphases)

            # Sort by timestamp
            all_emphases.sort(key=lambda x: x.insert_start)

            # Save results
            with open(output_path, 'w') as f:
                json.dump({
                    "target_id": chapter_data[0].get('target_id'),
                    "media_id": chapter_data[0].get('media_id', {}).get('$oid'),
                    "emphases": [e.model_dump() for e in all_emphases]
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
            output_path="/home/tanmay/Desktop/Ukumi_Tanmay/output/emphasis.json"
        )
        print(f"Successfully processed transcript")
        
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds")
    except Exception as e:
        print(f"Error processing transcript: {str(e)}")
        raise
