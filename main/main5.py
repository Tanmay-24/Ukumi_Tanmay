from typing import Annotated, TypedDict, List, Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os

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
            
            The goal is to remove content which does not contribute any significance to the converstaion. The transcript may contain stops between actual questions and answers. The task is to remove segments like water breaks, mic checks, and other unwanted content.
            
            Think step by step to solve this problem and process the entire transcript. 
            
            NOTE:
            Keep in mind that the answer to a question may be spread across multiple segments. 
            Ensure that answers are completly captured and not cut off. 
            Since most of the time questions are small but the answers are long, make sure to capture the entire answer.
            
            
            Output Format (structure only, do not output the structure):
            - question_start,question_end,answer_start,answer_end, topic name
            - question_start,question_end,answer_start,answer_end, topic name
            - list of segments that were removed, and the reason for removal
            
            
            Example output:
            10.32,18.105,18.885,155.65001,"Introduction"
            169.205,172.98,174.34,257.01,"Discussion on AI"
            
            Removed segments:
            5.32,10.105, "Water break"
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
            2. Use your persona to find the content you want to keep and, remove unwanted content. For this task use the original transcript.
            3. Remove unwanted words and profanity
            4. Maintain exact timestamps
            5. Do not modify core content of the transcript
            
            Output Format:
            - question_start,question_end,answer_start,answer_end, topic name
            
            Example output:
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
            Combine the editorial analysis and viewer feedback to:
            1. Identify highest potential segments
            2. Balance professional quality with audience appeal
            3. Create comprehensive segment ratings
            
            Format output:
            - question_start,question_end,answer_start,answer_end, topic name
            - Combined segment ratings
            - Overall recommendations
            - Priority segments
            """),
            ("human", "Editor Analysis: {editor_analysis}\nViewer Analysis: {viewer_analysis}")
        ])

        combiner_chain = combiner_prompt | self.llm

        try:
            combined = combiner_chain.invoke({
                "editor_analysis": state['editor_analysis'],
                "viewer_analysis": state['viewer_analysis']
            })
            return {
                **state,
                "combined_analysis": combined.content
            }
        except Exception as e:
            print(f"Combiner agent error: {str(e)}")
            raise

    def formatter_agent(self, state: CombinedState) -> CombinedState:
        """Formats the final output in the required structure"""
        formatter_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            Format the combined analysis into the final output structure:
            
            Required format:
            question_start,question_end,answer_start,answer_end,potential,title
            
            Rules:
            1. Extract precise timestamps
            2. Calculate potential scores (1-10)
            3. Create concise titles
            4. Sort by potential score
            
            Output only the formatted lines, nothing else.
            """),
            ("human", "{combined_analysis}")
        ])

        formatter_chain = formatter_prompt | self.llm

        try:
            formatted = formatter_chain.invoke({
                "combined_analysis": state['combined_analysis']
            })
            return {
                **state,
                "final_output": formatted.content
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

def main():
    try:
        transcript_path = "/home/tanmay/Desktop/Ukumi_Tanmay/data/output_saket.txt"
        with open(transcript_path, 'r') as file:
            transcript = file.read()

        processor = ParallelTranscriptProcessor()
        result = processor.process_transcript(transcript)
        
        print("Final Output:")
        print(result["final_output"])
        
        with open("script_output_2.txt", 'w') as file:
            file.write(result["final_output"])

    except FileNotFoundError:
        print(f"Error: Transcript file not found at {transcript_path}")
    except Exception as e:
        print(f"Processing error: {str(e)}")

if __name__ == "__main__":
    main()