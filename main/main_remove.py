from typing import Annotated, TypedDict, List, Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os
import json
from models import parse_text_format_2
import json


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
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=OPENAI_API_KEY)
        
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
        editor_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            As a professional Video Editor, analyze the podcast transcript and identify segments that should be removed while preserving natural flow and structure.

            Important Structure Elements to KEEP:
            1. Show introduction and episode setup
            2. Initial guest introduction
            3. Opening question and response
            4. Final conclusions
            5. Thank you messages and goodbyes
            
            Remove segments with:
            1. Audio/technical issues
            2. Off-topic discussions
            3. Dead air or silence
            4. Repetitive content
            5. Setup/preparation segments
            6. Mic checks or technical adjustments
            7. Interruptions

            Ensure smooth transitions by:
            - Keeping context between segments
            - Maintaining conversation flow
            - Preserving question-answer pairs
            
            Output Format:
            SEGMENTS TO REMOVE:
            start_time,end_time,"reason for removal"
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
        viewer_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            As an avid podcast viewer/listener, analyze the transcript while preserving natural episode structure.

            Essential Elements to KEEP:
            1. Show introduction and context setting
            2. Guest introductions and initial rapport
            3. Main topic introduction
            4. Key discussion points
            5. Natural conclusion
            6. Closing remarks and gratitude

            Remove segments that:
            1. Lack value or substance
            2. Are confusing or unclear
            3. Feel disconnected from main topics
            4. Contain filler content
            5. Have poor pacing
            6. Are redundant
            7. Break conversation flow

            Consider:
            - Natural conversation progression
            - Emotional resonance
            - Story arc completion
            - Professional closing
            
            Output Format:
            SEGMENTS TO REMOVE:
            start_time,end_time,"reason for removal"
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
        combiner_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            Analyze and combine segment removal suggestions while prioritizing video smoothness and completeness.

            Essential Structure to Preserve:
            1. Show introduction (first 1-2 minutes)
            2. Guest introduction and opening question
            3. Natural topic transitions
            4. Concluding remarks
            5. Thank you and goodbye segments (last 1-2 minutes)

            Combining Rules:
            1. Never remove introduction or conclusion segments
            2. Group small cuts that are less than 30 seconds apart into single larger segments
            3. Keep all questions with their complete answers
            4. Never cut in the middle of an answer
            5. If multiple small issues occur close together, remove the entire segment
            6. Prioritize keeping content over removing minor issues
            7. Only remove segments if they significantly impact viewing experience
            
            Hierarchy of what to keep (most important first):
            1. Complete question-answer pairs
            2. Story arcs and key points
            3. Context-setting segments
            4. Transitions between topics
            
            Output Format:
            SEGMENTS TO REMOVE:
            start_time,end_time,"reason for removal with grouping justification"
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
        formatter_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            Format the combined removal segments list with intelligent grouping.
            
            Rules:
            1. Merge segments that are less than 30 seconds apart
            2. Combine overlapping segments
            3. Update removal reasons to reflect grouped segments
            4. Ensure segments are ordered chronologically
            5. Never split question-answer pairs
            6. Prefer fewer longer segments over many short ones
            
            Required format:
            REMOVED SEGMENTS:
            start,end,"comprehensive reason for grouped removal"

            Example:
            178.5,242.3,"Combined technical issues and off-topic discussion"
            """),
            ("human", "{combined_analysis}")
        ])

        formatter_chain = formatter_prompt | self.llm

        try:
            formatted = formatter_chain.invoke({"combined_analysis": state["combined_analysis"]})
            parsed_data = parse_text_format_2(formatted.content)
            
            return {
                **state,
                "final_output": {
                    "removed_segments": parsed_data["removed_segments"],
                    "metadata": {
                        "total_segments": len(parsed_data["removed_segments"])
                    }
                }
            }
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

    def process_transcript(self, transcript: str) -> Dict[str, Any]:
        try:
            print("Starting parallel processing...")
            
            print("Running editor analysis...")
            editor_state = {
                "messages": [HumanMessage(content=transcript)],
                "transcript": transcript,
                "editor_analysis": None
            }
            
            print("Running viewer analysis...")
            viewer_state = {
                "messages": [HumanMessage(content=transcript)],
                "transcript": transcript,
                "viewer_analysis": None
            }

            editor_result = self.editor_graph.invoke(editor_state)
            print("Editor analysis complete.")
            
            viewer_result = self.viewer_graph.invoke(viewer_state)
            print("Viewer analysis complete.")

            if not editor_result.get("editor_analysis") or not viewer_result.get("viewer_analysis"):
                raise ValueError("Parallel processing failed to produce valid analysis")

            print("Combining results...")
            combined_state = {
                "messages": [],
                "editor_analysis": editor_result["editor_analysis"],
                "viewer_analysis": viewer_result["viewer_analysis"],
                "combined_analysis": None,
                "final_output": None
            }

            final_result = self.final_graph.invoke(combined_state)
            print("Final processing complete.")
            
            if not final_result.get("final_output"):
                raise ValueError("Final processing failed to produce valid output")

            return final_result

        except Exception as e:
            print(f"Processing error: {str(e)}")
            raise
                
def main():
    try:
        print("Starting transcript processing pipeline...")
        transcript_path = "/home/tanmay/Desktop/Ukumi_Tanmay/riverside_shri.txt"
        
        print("Reading transcript file...")
        with open(transcript_path, 'r') as file:
            transcript = file.read()

        processor = ParallelTranscriptProcessor()
        print("Analyzing transcript...")
        result = processor.process_transcript(transcript)
        
        if not result or not result.get("final_output"):
            raise ValueError("Processing failed to produce valid output")
            
        output_path = "removed_segments_output.json"
        print("Saving results...")
        with open(output_path, 'w') as file:
            json.dump(result["final_output"], file, indent=2)
            
        print(f"\nProcessing complete. Output saved to: {output_path}")
        print(f"Total segments removed: {len(result['final_output']['removed_segments'])}")

    except FileNotFoundError:
        print(f"Error: Transcript file not found at {transcript_path}")
    except Exception as e:
        print(f"Processing error: {str(e)}")
        raise

if __name__ == "__main__":
    main()