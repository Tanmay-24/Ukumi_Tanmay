from typing import Annotated, TypedDict, List, Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os
import json
import time
import traceback
from pydantic import BaseModel, Field, field_validator, ValidationError

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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

class UnifiedState(TypedDict):
    """Combined processing state"""
    messages: Annotated[List[BaseMessage], add_messages]
    transcript: str
    analysis: Optional[Dict[str, Any]]
    final_output: Optional[Dict[str, Any]]
    retry_count: int

class OptimizedTranscriptProcessor:
    def __init__(self, max_retries: int = 3):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.1, api_key=OPENAI_API_KEY,seed=5567)
        self.max_retries = max_retries
        self.workflow = self.setup_workflow()

    def unified_agent(self, state: UnifiedState) -> UnifiedState:
        """Combined editor/viewer analysis agent"""
        unified_prompt = ChatPromptTemplate.from_messages([
            ("system", """

You are a Podcast Transcript Cleaning and Optimization Specialist with the combined expertise of a meticulous Video Editor and an engaged Viewer. 
Your task is to analyze the provided transcript and identify segments that should be removed to enhance the natural flow of the podcast video without compressing or over-densifying the conversation.

Guidelines:
1. REMOVAL OF NON-ESSENTIAL CONTENT:
   - Identify segments that do not contribute meaningfully to the conversation (e.g., water breaks, mic checks, equipment tests, technical disruptions, background noise).
   - Preserve core content, including complete question answer pairs and any substantive dialogue. If an answer spans multiple segments, ensure the entire answer remains intact.
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
   - Podcast generaly follow many type for format such as interview style, discussion style etc
   - use this knowledge to give better outputs
   

OUTPUT INSTRUCTIONS:
- Provide a JSON array of segments that are recommended for removal.
- Format each segment as a JSON object with the keys: {{"start": float, "end": float,"title": str}}.
- Sort the segments chronologically.

Example Output:
[
  {{"start": 5.32, "end": 10.105, "title": Unwanted parts in the guest intro}},
  {{"start": 150.0, "end": 155.0, "title": off-topic discussion}}
]
"""),
            ("human", "Analyze the following transcript and identify segments for removal that are clearly unnecessary, following natural human editing choices without compressing the dialogue too much:\n{transcript}")
        ])

        
        chain = unified_prompt | self.llm
        analysis = chain.invoke({"transcript": state["transcript"]})
        # print(analysis.content)
        
        return {
            "messages": [*state["messages"], AIMessage(content=analysis.content)],
            "transcript": state["transcript"],
            "analysis": analysis.content,
            "final_output": None,
            "retry_count": state.get("retry_count", 0)
        }
    
    def setup_workflow(self) -> StateGraph:
        """Optimized processing pipeline"""
        workflow = StateGraph(UnifiedState)
        workflow.add_node("analyzer", self.unified_agent)
        workflow.add_node("formatter", self.formatter_agent)
        
        workflow.add_edge("analyzer", "formatter")
        workflow.add_edge("formatter", END)
        
        workflow.set_entry_point("analyzer")
        return workflow.compile()

    def formatter_agent(self, state: UnifiedState) -> UnifiedState:
        """Direct formatting with advanced retry mechanism"""
        format_prompt = ChatPromptTemplate.from_messages([
            ("system", """STRUCTURED OUTPUT FORMATTER
ENHANCED CONVERSION TASKS:
1. Transform text analysis to precise JSON
2. Rigorous timestamp validation
3. Ensure segment non-overlap
4. Final structure: {{"removed": [{{"start": float, "end": float}}, ...]}}

RETRY PROTOCOL:
- Analyze previous parsing challenges
- Refine extraction strategy
- Emphasize clean, consistent formatting"""),
            ("human", "{analysis}")
        ])
        
        retry_count = state.get("retry_count", 0)
        
        try:
            chain = format_prompt | self.llm.with_structured_output(RemovedSegmentsOutput)
            formatted = chain.invoke({"analysis": state["analysis"]})
            
            return {**state, "final_output": formatted.model_dump()}
        
        except (ValidationError, ValueError) as e:
            if retry_count < self.max_retries:
                print(f"Validation error (Attempt {retry_count + 1}): {e}")
                print(traceback.format_exc())
                
                # Retry with increased specificity
                retry_prompt = f"""PARSING ERROR CONTEXT:
Previous Parsing Attempt Failed
Error Details: {str(e)}

REFINED EXTRACTION REQUEST:
- Review the original analysis
- Strictly adhere to float timestamp format
- Ensure each segment has valid start/end times
- Remove any ambiguous or malformed entries

Original Analysis:
{state['analysis']}"""
                
                return {
                    **state, 
                    "retry_count": retry_count + 1,
                    "analysis": retry_prompt
                }
            else:
                print(f"Max retries ({self.max_retries}) exceeded. Returning last state.")
                return state
        
        except Exception as e:
            print(f"Unexpected formatting error: {e}")
            return state
        
    def process_transcript(self, transcript: str):
        """Single-pass processing with comprehensive validation"""
        try:
            initial_state = {
                "messages": [HumanMessage(content=transcript)],
                "transcript": transcript,
                "analysis": None,
                "final_output": None,
                "retry_count": 0
            }
            
            result = self.workflow.invoke(initial_state)
            
            if not result.get("final_output"):
                raise ValueError("Processing pipeline failed to generate valid output")
                
            return result
        
        except Exception as e:
            print(f"Processing failed: {str(e)}")
            return None

def main():
    start_time = time.time()
    try:
        print("Starting transcript processing pipeline...")
        transcript_path = "/home/tanmay/Desktop/Ukumi_Tanmay/extras/ai.txt"
        
        print("Reading transcript file...")
        with open(transcript_path, 'r') as file:
            transcript = file.read()

        processor = OptimizedTranscriptProcessor(max_retries=3)
        print("Analyzing transcript...")
        result = processor.process_transcript(transcript)
        
        if not result or not result.get("final_output"):
            raise ValueError("Processing failed to produce valid output")
            
        output_path = "/home/tanmay/Desktop/Ukumi_Tanmay/extras/removed_segments_output.json"
        print("Saving results...")
        with open(output_path, 'w') as file:
            json.dump(result["final_output"]["removed"], file, indent=2)
            
        print(f"\nProcessing complete. Output saved to: {output_path}")
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds")

    except FileNotFoundError:
        print(f"Error: Transcript file not found at {transcript_path}")
    except Exception as e:
        print(f"Processing error: {str(e)}")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()