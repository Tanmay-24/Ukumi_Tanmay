from typing import List
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
    title: str = Field(description="Description of removed content")

    @field_validator('start', 'end')
    def validate_timestamps(cls, v):
        if v < 0:
            raise ValueError("Timestamps must be non-negative")
        return v

class RemovedSegmentsOutput(BaseModel):
    """Output structure for removed segments"""
    removed: List[RemovedSegment]

# Create the processing agent with structured output
transcript_processor = Agent(
    model=OpenAIChat(id="gpt-4o",temperature=0.14,ap),
    description="Podcast Transcript Cleaning Specialist",
    response_model=RemovedSegmentsOutput,
    instructions="""You are a Podcast Transcript Cleaning and Optimization Specialist. Analyze the transcript and identify segments that should be removed to enhance natural flow while preserving core content.
Use your persona of a Video Editor
Guidelines:
1. Remove non-essential content (technical issues, off-topic discussions)
2. Preserve intros, outros, and substantive dialogue
3. Use exact original timestamps
4. Merge consecutive segments
5. Minimum 1-second granularity

Output JSON array of removal segments with start/end times and titles in this format:
{"removed": [{"start": 5.32, "end": 10.1, "title": "Reason"}, ...]}"""
)

def process_transcript(transcript_path: str, output_path: str):
    # Read transcript
    with open(transcript_path, 'r') as f:
        transcript = f.read()

    # Process with automatic retries
    result = transcript_processor.run(
        f"Analyze this podcast transcript:\n{transcript}",
        max_retries=3,
        validation_error_prompt="Fix JSON formatting and timestamp validation"
    )

    # Save results
    with open(output_path, 'w') as f:
        json.dump(result.content.model_dump()["removed"], f, indent=2)

if __name__ == "__main__":
    import time
    start_time = time.time()
    transcript_path = "/home/tanmay/Desktop/Ukumi_Tanmay/combined_transcript.txt"
    output_path = "/home/tanmay/Desktop/Ukumi_Tanmay/extras/phi.json"
    
    try:
        print("Starting transcript processing...")
        process_transcript(transcript_path, output_path)
        print(f"Successfully saved results to {output_path}")
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds")
    except Exception as e:
        print(f"Error processing transcript: {str(e)}")
        raise