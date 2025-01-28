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
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.2, api_key=OPENAI_API_KEY)
        
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
            
            The goal is to remove content which does not contribute any significance to the converstaion and identify the core Q&A pairs. The transcript may contain stops between actual questions and answers. The task is to remove segments like water breaks, mic checks, and other non-Q&A content.
            
            Think step by step and process the entire transcript. 
            
            NOTE:
            Keep in mind that the answer to a question may be spread across multiple segments. 
            Ensure that answrs are completly captured and not cut off. 
            Since most of the time questions are small but the answers are long, make sure to capture the entire answer.
            
            
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
            1. Remove unwanted words and profanity
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
            Also give, 
            potential: (number from 1-10) indicating the potential of the Q&A pair based on ->
            - Content depth and uniqueness
            - Emotional impact and relatability 
            - Educational or entertainment value
            - Shareability and trend relevance
            - Overall audience appeal
            - Title: A short title for each segment
            
            Output Format:
            question_start,question_end,answer_start,answer_end,potential,title,recommended
            
            DO NOT give anything other than the timestamps, potential score, title, and recommendation flag in the output.
            
            Example output: 
            10.32,18.105,18.885,155.65001,5,"Introduction"
            169.205,172.98,174.34,257.01,4,"Discussion on AI"
            271.965,279.745,280.41998,413.165,8,"Future of Technology"
            907.815,916.31,916.93,1052.7899,9,"Closing Remarks"
            
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
    
    # Initialize session state for results
    if 'processing_results' not in st.session_state:
        st.session_state.processing_results = None
    
    if st.sidebar.button("Process Transcript"):
        if not transcript:
            st.error("Please provide a transcript!")
            return
        
        with st.spinner("Processing transcript..."):
            processor = PodcastTranscriptProcessor()
            result = processor.process_transcript(transcript)
            st.session_state.processing_results = result
    
    # Display results if they exist in session state
    if st.session_state.processing_results:
        result = st.session_state.processing_results
        
        st.header("Processing Results")
        
        with st.expander("1. Detected Segments"):
            st.write(result.get("segments", "No segments detected"))
        
        with st.expander("2. Refined Q&A Pairs"):
            st.write(result.get("qa_pairs", "No Q&A pairs generated"))
        
        with st.expander("3. Final Output"):
            final_output = result.get("final_output", "No final output")
            st.write(final_output)
        
        # Segment selection section
        if final_output:
            st.header("Select Segments")
            
            # Initialize session state for selections if not exists
            if 'selected_segments' not in st.session_state:
                st.session_state.selected_segments = {}
            
            timestamps = final_output.content.strip('```').split('\n')
            selected_segments = []
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                for i, timestamp_group in enumerate(timestamps):
                    try:
                        # Split and validate the timestamp group
                        parts = timestamp_group.strip().split(',')
                        if len(parts) < 6:
                            continue
                        
                        times = [float(t) for t in parts[:4]]
                        potential_score = int(parts[4])
                        title = parts[5].strip('"')
                        
                        duration = (times[1] - times[0]) + (times[3] - times[2])
                        duration_str = f"{int(duration // 60)}:{int(duration % 60):02d}"
                        
                        score_color = f"{'#' + hex(int(255 * (potential_score/10)))[2:].zfill(2) + 'ff' + hex(int(255 * (1-potential_score/10)))[2:].zfill(2)}"
                        
                        seg_col1, seg_col2, seg_col3 = st.columns([4, 1, 1])
                        
                        with seg_col1:
                            key = f"segment_{i}"
                            # Initialize key if not exists
                            if key not in st.session_state.selected_segments:
                                st.session_state.selected_segments[key] = False
                            
                            if st.checkbox(
                                f"{title} ({duration_str})",
                                value=st.session_state.selected_segments[key],
                                key=key
                            ):
                                selected_segments.append(timestamp_group)
                                st.session_state.selected_segments[key] = True
                            else:
                                st.session_state.selected_segments[key] = False
                        
                        with seg_col2:
                            st.metric("Score", f"{potential_score}/10")
                        
                        with seg_col3:
                            if st.button("ℹ️", key=f"info_{i}"):
                                st.info(f"This segment has a potential score of {potential_score}/10 based on its content depth, emotional impact, educational value, and shareability.")
                    
                    except (ValueError, IndexError) as e:
                        st.error(f"Error processing segment {i}: {str(e)}")
                        continue
            
            # Rest of the code remains the same
            # Calculate total duration and average potential
            total_duration = 0
            total_potential = 0
            if selected_segments:
                for segment in selected_segments:
                    parts = segment.strip().split(',')
                    times = [float(t) for t in parts[:4]]
                    total_duration += (times[1] - times[0]) + (times[3] - times[2])
                    total_potential += int(parts[4])
                
                avg_potential = total_potential / len(selected_segments)
                
                st.metric(
                    "Total Duration",
                    f"{int(total_duration // 60)}:{int(total_duration % 60):02d}"
                )
                st.metric(
                    "Average Potential",
                    f"{avg_potential:.1f}/10"
                )
        
        # Download buttons
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "Download All Results",
                data=str(final_output.content),
                file_name="processed_transcript.txt",
                mime="text/plain"
            )
        
        with col2:
            if selected_segments:
                selected_output = "\n".join(selected_segments)
                st.download_button(
                    "Download Selected Segments",
                    data=selected_output,
                    file_name="selected_segments.txt",
                    mime="text/plain"
                )
                    
if __name__ == "__main__":
    main()