from typing import Annotated, TypedDict, List, Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os
import json
from models import TranscriptAnalysis, parse_text_format
import json
from exclude import find_segments_to_remove

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Fixed state definitions with proper typing
class EditorState(TypedDict):
    """State for editor agent processing"""
    messages: Annotated[List[BaseMessage], add_messages]
    transcript: str  # Changed to required field
    editor_analysis: Optional[Dict[str, Any]]

class ViewerState(TypedDict):
    """State for viewer agent processing"""
    messages: Annotated[List[BaseMessage], add_messages]
    transcript: str  # Changed to required field
    viewer_analysis: Optional[Dict[str, Any]]

class CombinedState(TypedDict):
    """State for combined processing"""
    messages: Annotated[List[BaseMessage], add_messages]
    editor_analysis: Dict[str, Any]  # Changed to required field
    viewer_analysis: Dict[str, Any]  # Changed to required field
    combined_analysis: Optional[Dict[str, Any]]
    final_output: Optional[Dict[str, Any]]

class ParallelTranscriptProcessor:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.1, api_key=OPENAI_API_KEY)
        
        # Initialize separate graphs
        self.editor_graph = self.setup_editor_workflow()
        self.viewer_graph = self.setup_viewer_workflow()
        self.final_graph = self.setup_final_workflow()

    def setup_editor_workflow(self) -> StateGraph:
        """Setup the editor agent workflow"""
        editor_graph = StateGraph(EditorState)
        editor_graph.add_node("editor", self.editor_agent)
        editor_graph.add_edge("editor", END)
        editor_graph.set_entry_point("editor")
        return editor_graph.compile()

    def setup_viewer_workflow(self) -> StateGraph:
        """Setup the viewer agent workflow"""
        viewer_graph = StateGraph(ViewerState)
        viewer_graph.add_node("viewer", self.viewer_agent)
        viewer_graph.add_edge("viewer", END)
        viewer_graph.set_entry_point("viewer")
        return viewer_graph.compile()

    def setup_final_workflow(self) -> StateGraph:
        """Setup the final combination workflow"""
        final_graph = StateGraph(CombinedState)
        final_graph.add_node("combiner", self.combiner_agent)
        final_graph.add_node("formatter", self.formatter_agent)
        final_graph.add_edge("combiner", "formatter")
        final_graph.add_edge("formatter", END)
        final_graph.set_entry_point("combiner")
        return final_graph.compile()

    def editor_agent(self, state: EditorState) -> EditorState:
        """Editor persona agent analyzing content quality and structure"""
        editor_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            Analyze the podcast transcript and identify question-answer pairs.
            Rules:
            1. Maintain exact timestamps from the original transcript
            2. Do not generate new text or modify existing content
            3. Your persona is of a Video Editor who edits podcast videos, and you are processing the transcript for video editing purposes. 
            4. You should work on removing unwanted content and identifying the core video segments from the transcript. Use your persona of a Video Editor to guide your decisions.
            
            The goal is to remove content which does not contribute any significance to the converstaion. The transcript may contain stops between actual questions and answers. The task is to remove segments like water breaks, mic checks, etc.
            
            Think step by step to solve this problem and process the entire transcript. 
            
            Important Structure Elements to KEEP:
            1. Show introduction and episode setup
            2. Initial guest introduction
            3. Opening question and response
            4. Final conclusions
            5. Thank you messages and goodbyes
            
            NOTE:
            Keep in mind that the answer to a question may be spread across multiple segments. 
            Ensure that answers are completly captured and not cut off. 
            Since most of the time questions are small but the answers are long, make sure to capture the entire answer.
            
            Ensure smooth transitions by:
            - Keeping context between segments
            - Maintaining conversation flow
            - Preserving question-answer pairs
            
            OUTPUT->
            - question_start,question_end,answer_start,answer_end,topic name
            - question_start,question_end,answer_start,answer_end,topic name
            
            Example Output:
            
            KEPT SEGMENTS:
            10.32,18.105,18.885,155.65001,"Introduction"
            169.205,172.98,174.34,257.01,"Discussion on AI"
            
            """),
            ("human", "{transcript}")
        ])

        editor_chain = editor_prompt | self.llm

        try:
            analysis = editor_chain.invoke({"transcript": state['transcript']})
            return {
                "messages": [*state['messages'], AIMessage(content=str(analysis))],
                "transcript": state['transcript'],
                "editor_analysis": analysis.content  # Ensure we're getting the content
            }
        except Exception as e:
            print(f"Editor agent error: {str(e)}")
            raise  # Raise the error instead of returning incomplete state

    def viewer_agent(self, state: ViewerState) -> ViewerState:
        """Viewer persona agent analyzing audience appeal and engagement"""
        viewer_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            Clean the identified segments:
            1. Your persona is of a Viewer who likes to watch podcast videos, you have access to the orignal transcript
            2. Use your persona to find the content you want to keep and, remove content which a viewer would not like to see. For this task use the original transcript.
            3. Maintain exact timestamps
            4. Do not modify core content of the transcript
            
            Essential Elements to KEEP:
            1. Show introduction and context setting
            2. Guest introductions and initial rapport
            3. Main topic introduction
            4. Key discussion points
            5. Natural conclusion
            6. Closing remarks and gratitude
            Consider:
            - Natural conversation progression
            - Emotional resonance
            - Story arc completion
            - Professional closing
            
            OUTPUT->
            - question_start,question_end,answer_start,answer_end,topic name
            - question_start,question_end,answer_start,answer_end,topic name
            
            Example Output:
            10.32,18.105,18.885,155.65001,"Introduction"
            169.205,172.98,174.34,257.01,"Discussion on AI"
            
            """),
            ("human", "{transcript}")
        ])

        viewer_chain = viewer_prompt | self.llm

        try:
            analysis = viewer_chain.invoke({"transcript": state['transcript']})
            return {
                "messages": [*state['messages'], AIMessage(content=str(analysis))],
                "transcript": state['transcript'],
                "viewer_analysis": analysis.content  # Ensure we're getting the content
            }
        except Exception as e:
            print(f"Viewer agent error: {str(e)}")
            raise  # Raise the error instead of returning incomplete state

    def combiner_agent(self, state: CombinedState) -> CombinedState:
        """Combines editor and viewer analyses"""
        combiner_prompt = ChatPromptTemplate.from_messages([
            ("system", """
             You are a Combiner agent responsible for merging the editor and viewer analyses.
            and add ratings (1-10) for each segment based on quality and engagement.
            
            Essential Structure to Preserve:
            1. Show introduction 
            2. Guest introduction and opening question
            3. Natural topic transitions
            4. Concluding remarks
            5. Thank you and goodbye segments 
            
            Format output:
            - question_start,question_end,answer_start,answer_end,topic name
            - rating
            """),
            ("human", "Editor Analysis: {editor_analysis}\nViewer Analysis: {viewer_analysis}")
        ])

        combiner_chain = combiner_prompt | self.llm

        try:
            combined = combiner_chain.invoke({
                "editor_analysis": state['editor_analysis'],
                "viewer_analysis": state['viewer_analysis']
            })
            print(combined.content)
            return {
                **state,
                "combined_analysis": combined.content
            }
            
        except Exception as e:
            print(f"Combiner agent error: {str(e)}")
            raise

    def formatter_agent(self, state: CombinedState) -> CombinedState:
        """Formats the final output using Pydantic validation"""
        formatter_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            Format into a structured output.
            
            Required format:
            KEPT SEGMENTS:
            start,end,question_start,answer_end,title,rating
            
            REMOVED SEGMENTS:
            start,end
            
            Rules:
            1. All timestamps must be valid numbers
            2. Ratings must be between 0-10
            3. Each segment must have a title
            4. Title should be descriptive but concise
            5. Order of the segments should be according to the timestamps
            
            """),
            ("human", "{combined_analysis}")
        ])

        formatter_chain = formatter_prompt | self.llm

        try:
            formatted = formatter_chain.invoke({
                "combined_analysis": state['combined_analysis']
            })
            
            # Parse the LLM output into our schema format
            parsed_data = parse_text_format(formatted.content)
            
            # Validate using Pydantic
            try:
                validated_data = TranscriptAnalysis(**parsed_data)
                return {
                    **state,
                    "final_output": validated_data.model_dump()
                }
            except Exception as e:
                print(f"Validation error: {str(e)}")
                raise
            
        except Exception as e:
            print(f"Formatter agent error: {str(e)}")
            raise

    def process_transcript(self, transcript: str) -> Dict[str, Any]:
        """Main processing method with proper error handling and state validation"""
        try:
            # Run editor and viewer analyses in parallel
            editor_state = {
                "messages": [HumanMessage(content=transcript)],
                "transcript": transcript,
                "editor_analysis": None
            }
            
            viewer_state = {
                "messages": [HumanMessage(content=transcript)],
                "transcript": transcript,
                "viewer_analysis": None
            }

            editor_result = self.editor_graph.invoke(editor_state)
            viewer_result = self.viewer_graph.invoke(viewer_state)

            # Validate results before proceeding
            if not editor_result.get("editor_analysis") or not viewer_result.get("viewer_analysis"):
                raise ValueError("Parallel processing failed to produce valid analysis")

            # Combine results and run final processing
            combined_state = {
                "messages": [],  # Reset messages for final processing
                "editor_analysis": editor_result["editor_analysis"],
                "viewer_analysis": viewer_result["viewer_analysis"],
                "combined_analysis": None,
                "final_output": None
            }

            final_result = self.final_graph.invoke(combined_state)
            
            # Validate final output
            if not final_result.get("final_output"):
                raise ValueError("Final processing failed to produce valid output")

            return final_result

        except Exception as e:
            print(f"Processing error: {str(e)}")
            raise

def main():
    try:
        transcript_path = "/home/tanmay/Desktop/Ukumi_Tanmay/riverside_shri.txt"
        with open(transcript_path, 'r') as file:
            transcript = file.read()

        processor = ParallelTranscriptProcessor()
        result = processor.process_transcript(transcript)
        
        print("Final Output:")
        print(json.dumps(result["final_output"], indent=2))
        
        def calculate_kept_duration(segments):
            total_kept_duration = 0
            for segment in segments:
                start = segment["timestamp"]["start"]
                end = segment["timestamp"]["end"]
                duration = end - start
                total_kept_duration += duration
            return total_kept_duration
        
        # Calculate removed segments
        total_duration = 5950
        removed_segments = find_segments_to_remove(result["final_output"], total_duration)
        print(removed_segments)
        # Create combined output dictionary with both kept and removed segments
        final_output = {
            "kept_segments": result["final_output"]["segments"],
            "removed_segments": removed_segments,
            "metadata": {
                "total_duration": total_duration,
                "total_kept_duration": calculate_kept_duration(result["final_output"]["segments"]),
                "total_kept_segments": len(result["final_output"]["segments"]),
                "total_removed_segments": len(removed_segments)
            }
        }
        
        # Save the complete output with both kept and removed segments
        with open("script_3_output_shri.json", 'w') as file:
            json.dump(final_output, file, indent=2)
            
        print(f"Saved output with {len(removed_segments)} removed segments")

    except FileNotFoundError:
        print(f"Error: Transcript file not found at {transcript_path}")
    except Exception as e:
        print(f"Processing error: {str(e)}")

if __name__ == "__main__":
    main()