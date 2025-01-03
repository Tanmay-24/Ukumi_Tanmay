import os
import streamlit as st
from typing import Annotated, TypedDict, List, Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from graphviz import Digraph


import os
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class State(TypedDict):
    """State management"""
    messages: Annotated[List[BaseMessage], add_messages]
    transcript: Optional[str]
    segments: Optional[List[Dict[str, Any]]]
    qa_pairs: Optional[List[Dict[str, Any]]]
    final_output: Optional[Dict[str, Any]]

class PodcastTranscriptProcessor:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, api_key=OPENAI_API_KEY)
        
        # Building  the Graph
        self.graph_builder = StateGraph(State)
        self.setup_workflow()

    def setup_workflow(self):
        self.graph_builder.add_node("flow_detector", self.flow_detector_agent)
        self.graph_builder.add_node("content_refinement", self.content_refinement_agent)
        self.graph_builder.add_node("timestamp_precision", self.timestamp_precision_agent)

        
        self.graph_builder.add_edge("flow_detector", "content_refinement")
        self.graph_builder.add_edge("content_refinement", "timestamp_precision")
        self.graph_builder.add_edge("timestamp_precision", END)

        
        self.graph_builder.set_entry_point("flow_detector")

        #finally complie
        self.graph = self.graph_builder.compile()

    def flow_detector_agent(self, state: State) -> State:
        flow_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            Analyze the podcast transcript and identify question-answer pairs.
            Rules:
            1. Maintain exact timestamps from the original transcript
            2. Do not generate new text or modify existing content
            3. Focus on clear question-answer segments
            
            The goal is to remove unwanted content and identify the core Q&A pairs. The transcript may contain stops between actual questions and answers. The task is to remove segments like water breaks, mic checks, and other non-Q&A content.
            Think step by step and process the entire transcript.
            
            Output Format:
            - Question segment with start/end timestamps
            - Answer segment with start/end timestamps
            """),
            ("human", "{transcript}")
        ])

        flow_chain = flow_prompt | self.llm


        try:
            segments = flow_chain.invoke({
                "transcript": state['transcript']
            })

            return {
                **state,
                "messages": [
                    *state['messages'],
                    AIMessage(content=str(segments))
                ],
                "segments": segments
            }
        except Exception as e:
            st.error(f"Flow detection error: {str(e)}")
            return state
        
    def content_refinement_agent(self, state: State) -> State:
        refinement_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            Clean the identified segments:
            1. Remove filler words and profanity
            2. Maintain exact timestamps
            3. Ensure question-answer alignment
            4. Do not modify core content of the transcript
            
            Output Format:
            - Cleaned question text with original timestamps
            - Cleaned answer text with original timestamps
            """),
            ("human", "{segments}")
        ])


        refinement_chain = refinement_prompt | self.llm

        try:
            refined_segments = refinement_chain.invoke({
                "segments": state['segments']
            })

            return {
                **state,
                "messages": [
                    *state['messages'],
                    AIMessage(content=str(refined_segments))
                ],
                "qa_pairs": refined_segments
            }
        except Exception as e:
            st.error(f"Content refinement error: {str(e)}")
            return state

    def timestamp_precision_agent(self, state: State) -> Dict[str, Any]:
        """Finalizes timestamp boundaries and formats output"""
        precision_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            Format final Q&A pairs with precise timestamps:
            1. Verify timestamp accuracy
            2. Ensure proper segmentation
            3. Format for video editor use
            
            Output Format:
            question_start,question_end,answer_start,answer_end
            """),
            ("human", "{qa_pairs}")
        ])

        precision_chain = precision_prompt | self.llm

        try:
            final_output = precision_chain.invoke({
                "qa_pairs": state['qa_pairs']
            })

            return {
                **state,
                "messages": [
                    *state['messages'],
                    AIMessage(content=str(final_output))
                ],
                "final_output": final_output
            }
        except Exception as e:
            st.error(f"Timestamp precision error: {str(e)}")
            return state

    def process_transcript(self, transcript: str):
        
        """Main method"""
        
        initial_state = {
            "messages": [HumanMessage(content=transcript)],
            "transcript": transcript,
            "segments": None,
            "qa_pairs": None,
            "final_output": None
        }

        return self.graph.invoke(initial_state)
    
def generate_workflow_diagram(): ## Just to get the diagram
    dot = Digraph(comment='Podcast Transcript Processing Workflow')
    
    
    dot.attr(rankdir='LR')
    dot.attr('node', shape='rectangle')  
    
    
    dot.node('flow_detector', 'Flow Detector')
    dot.node('content_refinement', 'Content\nRefinement')
    dot.node('timestamp_precision', 'Timestamp\nPrecision')
    dot.node('END', 'End')
    
    
    dot.edge('flow_detector', 'content_refinement', 'Segments identified')
    dot.edge('content_refinement', 'timestamp_precision', 'Content refined')
    dot.edge('timestamp_precision', 'END', 'Final output')
    
    #Save the png file
    dot.render('agent_workflow', format='png', cleanup=True)
    return 'agent_workflow.png'

def main():
    st.set_page_config(page_title="Podcast Transcript Processor", layout="wide")
    
    
    st.title("Podcast Transcript Processor")
    if st.button("Generate Workflow Diagram"):
        
        try:
            diagram_path = generate_workflow_diagram()
            st.image(diagram_path, caption="Agent Workflow Diagram")
            
            with open(diagram_path, "rb") as file:
                btn = st.download_button(
                    label="Download Workflow Diagram",
                    data=file,
                    file_name="agent_workflow.png",
                    mime="image/png"
                )
        except Exception as e:
            st.error(f"Error generating diagram: {str(e)}")
    
    # Input section
    st.sidebar.header("Input Transcript")
    input_method = st.sidebar.radio(
        "Choose Input Method",
        ["Paste Transcript", "Upload File"]
    )
    
    transcript = ""
    if input_method == "Paste Transcript":
        transcript = st.sidebar.text_area(
            "Paste the transcript here",
            height=300
        )
    else:
        uploaded_file = st.sidebar.file_uploader(
            "Upload transcript file",
            type=["txt"]
        )
        if uploaded_file:
            transcript = uploaded_file.getvalue().decode("utf-8")
    
    if st.sidebar.button("Process Transcript"):
        if not transcript:
            st.error("Please provide a transcript!")
            return
        
        with st.spinner("Processing transcript..."):
            processor = PodcastTranscriptProcessor()
            result = processor.process_transcript(transcript)
        
        # Display outputs 
        st.header("Processing Results")
        
        with st.expander("1. Detected Segments"):
            st.write(result.get("segments", "No segments detected"))
        
        with st.expander("2. Refined Q&A Pairs"):
            st.write(result.get("qa_pairs", "No Q&A pairs generated"))
        
        with st.expander("3. Final Output"):
            final_output = result.get("final_output", "No final output")
            st.write(final_output)
        
        
        st.download_button(
            "Download Results",
            data=str(final_output),
            file_name="processed_transcript.txt",
            mime="text/plain"
        )

if __name__ == "__main__":
    main()