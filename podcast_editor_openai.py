import json
import os
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
import openai
import tiktoken

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def get_words_from_json(json_data: Dict[str, Any]) -> List[Dict]:
    """Parse the input JSON and extract the word-level transcript."""
    try:
        return json_data['data'][2]['channels']['word_level_transcripts']
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError(f"Invalid input JSON structure for word transcripts: {str(e)}")

def get_sen_from_json(json_data: Dict[str, Any]) -> List[Dict]:
    """Parse the input JSON and extract the sentence-level transcript."""
    try:
        return json_data['data'][2]['channels']['sentence_level_transcript']
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError(f"Invalid input JSON structure for sentence transcripts: {str(e)}")

def get_chapters_from_json(json_data: Dict[str, Any]) -> List[Dict]:
    """Parse the input JSON and extract the chapters."""
    try:
        return json_data['data'][2]['channels']['chapter_data']
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError(f"Invalid input JSON structure for chapters: {str(e)}")

def analyze_podcast_format(sentence_transcript: List[Dict]) -> str:
    """Analyze the entire podcast format using the OpenAI API with o3-mini model."""

    # Create a concise representation of the entire transcript without timestamps
    transcript_str = "\n".join([sentence.get("text", "") for sentence in sentence_transcript])
    
    prompt = """Analyze this podcast transcript and determine its overall format.
Possible formats: Q&A Format, Narrative Format, Panel Discussion, Interview, Technical Explanation, Monologue.
"""

    try:
        response = client.responses.create(
            model="gpt-4o-mini",
            temperature=0.3,
            input=[
                {"role": "user", "content": prompt + "\n\nTRANSCRIPT:\n" + transcript_str}  # Limit size if needed
            ],
            text={
                "format":{
                    "type": "json_schema",
                    "name": "podcast_format",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "format": {"type": "string","description": "The primary format of the podcast", "enum": ["Q&A Format", "Narrative Format", "Panel Discussion", "Interview", "Technical Explanation", "Monologue"]},
                        },
                        "required": ["format"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            }
        )
        
        result = json.loads(response.output_text)
        return result['format']
    
    except Exception as e:
        return Exception(f"Error analyzing podcast format: {e}")


def process_chapter_edits(chapter_data: dict, word_transcript: list[Dict], format_type: str) -> dict:
    """Process both content and filler word edits for a chapter using the OpenAI API with the o3-mini model."""
    combined_removals = []
    # TESTING: Only process first 3 chapters
    chapter_data = chapter_data[:3]
    total_chapters = len(chapter_data)
    
    for i, chapter in enumerate(chapter_data, 1):
        print(f"\nProcessing chapter {i}/{total_chapters}: {chapter['title']}")
        
        chapter_transcript = [
            word
            for word in word_transcript
            if float(word["start"]) >= float(chapter["start"]) and float(word["end"]) <= float(chapter["end"])
        ]
        
        transcript_str = (
            f"Chapter: {chapter['title']} (Start: {chapter['start']}, End: {chapter['end']})\n" +
            "\n".join(
                f"[{sentence['start']}-{sentence['end']}] {sentence['text']}" for sentence in chapter_transcript
            )
        )
        
        # Count tokens for this chapter
        token_count = count_tokens(transcript_str)
        print(f"Token count for chapter {i}: {token_count}")
        if token_count > 8000:  # Approximate context limit for o3-mini
            print(f"Warning: Chapter {i} exceeds recommended token limit. Consider splitting into smaller chunks.")
        
        # Enhanced format-specific strategies with clear delimiters
        format_lower = format_type.lower()
        if "q&a" in format_lower or "interview" in format_lower:
            strategy = """
            <format_strategy>
            Q&A/Interview Format:
            - Preserve complete question-answer pairs
            - Keep natural transitions between questions
            - Maintain interviewer-interviewee rapport
            - Preserve genuine reactions and emotions
            - Remove content not part of Q&A pairs
            - Keep authentic expressions of enthusiasm
            - Preserve personal connections
            - Remove technical setup noise
            </format_strategy>
            """
        elif "narrative" in format_lower:
            strategy = """
            <format_strategy>
            Narrative Format:
            - Focus on story flow and key plot points
            - Preserve authentic reactions and emotions
            - Keep natural story progression
            - Maintain engaging descriptions
            - Remove excessive descriptions
            - Preserve personal connections
            - Keep authentic expressions
            - Remove technical setup noise
            </format_strategy>
            """
        elif "technical" in format_lower:
            strategy = """
            <format_strategy>
            Technical Format:
            - Focus on technical accuracy
            - Maintain engaging explanations
            - Preserve authentic reactions
            - Keep natural progression
            - Remove redundant explanations
            - Preserve personal connections
            - Keep authentic expressions
            - Remove technical setup noise
            </format_strategy>
            """
        else:
            strategy = """
            <format_strategy>
            General Format:
            - Focus on clarity and removing redundancy
            - Preserve authentic voice
            - Keep main points and insights
            - Maintain natural flow
            - Remove tangents
            - Preserve personal connections
            - Keep authentic expressions
            - Remove technical setup noise
            </format_strategy>
            """
        
        # Updated prompt with enhanced format-specific instructions and clear delimiters
        prompt = f"""
<task>
You are an expert podcast editor for a {format_type} podcast. Your goal is to enhance the listening experience by removing content that might bore or distract viewers while preserving the authentic voice and important elements.
</task>

{strategy}

<content_guidelines>
REMOVE:
- Technical setup phrases ("okay", "yeah", "great")
- Incomplete sentences and thoughts
- Repetitive words/phrases
- Off-topic tangents
- Factually incorrect statements

KEEP:
- Expressions of gratitude and politeness
- Personal connections and authentic reactions
- Key personality traits of speakers
- Main discussion points and arguments
- Natural conversation flow
</content_guidelines>

<quality_requirements>
- Each remaining sentence must be complete and coherent
- Maintain the natural flow of conversation
- Preserve the speaker's authentic voice
- Keep the emotional authenticity
- Ensure content remains engaging
</quality_requirements>

<analysis_requirements>
1. Identify segments containing filler words, false starts, or disjointed phrasing
2. Evaluate if the entire chapter is non-essential
3. Preserve vital content for understanding
4. Ensure removal segments don't overlap
5. Use exact source timestamps
</analysis_requirements>

<output_format>
Return a JSON object with edits that include:
- start: timestamp
- end: timestamp
- reason: brief explanation
- chapter_title: chapter name
</output_format>

<chapter_data>
{transcript_str}
</chapter_data>
"""
        
        try:
            response = client.responses.create(
                model="o3-mini",
                reasoning={"effort": "high"},
                input=[
                    {"role": "user", "content": prompt}
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "chapter_edits",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "edits": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "start": {"type": "number"},
                                            "end": {"type": "number"},
                                            "reason": {"type": "string"},
                                            "chapter_title": {"type": "string"}
                                        },
                                        "required": ["start", "end", "reason", "chapter_title"],
                                        "additionalProperties": False
                                    },
                                    "strict": True
                                },
                            },
                            "required": ["edits"],
                            "additionalProperties": False
                        },
                        "strict": True
                    }
                }
            )
        
            result = json.loads(response.output_text)
            combined_removals.extend(result["edits"])
           

        except Exception as e:
            return Exception(f"Error processing chapter edits: {e}")
        
    return combined_removals

def reconstruct_transcript(chapter_data: list[dict], word_transcript: list[dict], edit_result: List[Dict[str, Any]]) -> str:
    """Reconstruct the refined transcript by applying chapter-level removal edits."""
    all_removals = edit_result
    refined_chapters = []
    # TESTING: Only process first 3 chapters
    chapter_data = chapter_data[:3]
    
    for chapter in chapter_data:
        chapter_title = chapter["title"]
        chapter_start = chapter["start"]
        chapter_end = chapter["end"]
        
        # Filter removal segments for this chapter using the chapter title
        chapter_removals = [
            removal for removal in all_removals
            if removal.get("chapter_title") == chapter_title
        ]
        
        # Get word-level segments that fall within this chapter's boundaries
        chapter_words = [
            word for word in word_transcript
            if chapter_start <= float(word.get("start", 0)) and float(word.get("end", 0)) <= chapter_end
        ]
        
        refined_words = []
        # Iterate over each word and check against all removal segments for this chapter
        for word in chapter_words:
            word_start = float(word.get("start", 0))
            word_end = float(word.get("end", 0))
            word_duration = word_end - word_start
            remove_word = False
            
            for removal in chapter_removals:
                rem_start = float(removal.get("start", 0))
                rem_end = float(removal.get("end", 0))
                # Calculate the overlap between this word and the removal segment
                overlap = max(0, min(word_end, rem_end) - max(word_start, rem_start))
                if word_duration > 0 and (overlap / word_duration) > 0.5:
                    remove_word = True
                    break
            
            if not remove_word:
                refined_words.append(word.get("text", ""))
        
        # Reconstruct the chapter transcript with a header and join the remaining words
        chapter_text = f"Chapter: {chapter_title}\n" + " ".join(refined_words)
        refined_chapters.append(chapter_text)
    
    # Join all chapter transcripts with double newlines
    edit_applied_transcript = "\n\n".join(refined_chapters)
    return edit_applied_transcript


def reflect_on_transcript(chapter_data: List[Dict[str, Any]], word_transcript: List[Dict[str, Any]], edited_transcript: str, edits: List[Dict[str, Any]]) -> str:
    """Analyze the coherence, flow, and broken sentences for each chapter."""
    print("\n=== Starting Transcript Reflection ===")
    edited_chapters = edited_transcript.split("Chapter:")
    reflection_notes = []
    # TESTING: Only process first 3 chapters
    chapter_data = chapter_data[:3]
    total_chapters = len(chapter_data)
    
    for i, chapter in enumerate(chapter_data, 1):
        print(f"\nReflecting on chapter {i}/{total_chapters}: {chapter['title']}")
        
        # Count tokens for reflection input
        chapter_start = float(chapter["start"])
        chapter_end = float(chapter["end"])
        
        # Reconstruct original chapter transcript
        chapter_words = [
            word["text"] for word in word_transcript
            if chapter_start <= float(word["start"]) and float(word["end"]) <= chapter_end
        ]
        original_chapter = " ".join(chapter_words)
        
        edited_chapter = edited_chapters[i] if i < len(edited_chapters) else ""
        chapter_edits = [edit for edit in edits if edit.get("chapter_title") == chapter["title"]]
        
        reflection_input = f"""
        Original Chapter {i} ({chapter["title"]}):
        {original_chapter.strip()}

        Edited Chapter {i}:
        {edited_chapter.strip()}

        Edits made:
        {json.dumps(chapter_edits, indent=2)}
        """
        
        token_count = count_tokens(reflection_input)
        print(f"Token count for chapter {i} reflection: {token_count}")
        if token_count > 8000:
            print(f"Warning: Chapter {i} reflection exceeds recommended token limit")
        
        prompt = f"""
        Analyze the following original and edited podcast chapter transcripts:

        Original Chapter {i} ({chapter["title"]}):
        {original_chapter.strip()}

        Edited Chapter {i}:
        {edited_chapter.strip()}

        Edits made:
        {json.dumps(chapter_edits, indent=2)}

         Provide a detailed analysis focusing on the following aspects,
         Include timestamps or segment of sentences in the output to give a better idea of what part of the transcript we are talking about in the analysis.

        1. Coherence:
           - Has the overall coherence of the chapter improved or deteriorated?
           - Are the main ideas and arguments presented more clearly in the edited version?
           - Is there a logical progression of thoughts throughout the chapter?

        2. Flow:
           - Is the flow of ideas smoother in the edited version?
           - Are transitions between topics more natural and effective?
           - Has the removal of content affected the narrative structure positively or negatively?

        3. Broken Sentences:
           - Identify any sentences that became broken or incomplete due to the edits.
           - Assess if these broken sentences impact the understanding of the content.
           - Suggest potential fixes for any problematic sentences.

        4. Edit Impact:
           - Evaluate how the specific edits have affected the chapter's overall quality.
           - Determine if important information was inadvertently removed.
           - Assess if the edits have improved the clarity and conciseness of the content.

        5. Content Preservation:
           - Ensure that key points, main arguments, and unique insights are preserved in the edited version.
           - Identify any instances where crucial context or examples may have been lost.

        Format your analysis as a concise yet comprehensive paragraph for each of the five aspects mentioned above.
        """

        try:
            response = client.responses.create(
                model="gpt-4o",
                temperature=0.5,
                input=[
                    {"role": "user", "content": prompt}
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "chapter_reflection",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "coherence": {"type": "string"},
                                "flow": {"type": "string"},
                                "broken_sentences": {"type": "string"},
                                "edit_impact": {"type": "string"},
                                "content_preservation": {"type": "string"},
                            },
                            "required": ["coherence", "flow", "broken_sentences", "edit_impact", "content_preservation"],
                            "additionalProperties": False
                        },
                        "strict": True
                    }
                }
            )
            
            result = json.loads(response.output_text)
            reflection_notes.append(f"""Reflection on Chapter {i} :
                                    
Coherence: {result['coherence']}

Flow: {result['flow']}

Broken Sentences: {result['broken_sentences']}

Edit Impact: {result['edit_impact']}

Content Preservation: {result['content_preservation']}
""")
            
        except Exception as e:
            reflection_notes.append(f"Chapter {i} Reflection: Error occurred during analysis - {str(e)}\n")
    
    return "\n\n".join(reflection_notes)

def apply_final_edits(word_transcript: List[Dict[str, Any]], chapter_data: List[Dict[str, Any]], final_edits: List[Dict[str, Any]]) -> str:
    """Apply the final edits to the word-level transcript and reconstruct the final edited transcript."""
    final_chapters = []
    # TESTING: Only process first 3 chapters
    chapter_data = chapter_data[:3]
    
    for chapter, chapter_edits in zip(chapter_data, final_edits):
        chapter_start = float(chapter["start"])
        chapter_end = float(chapter["end"])
        print(f"\nProcessing final edits for chapter: {chapter['title']}")
        print(f"Start time: {chapter_start}, End time: {chapter_end}")
        print(f"Number of edits to apply: {len(chapter_edits['final_chapter_edits'])}")
        
        chapter_words = [
            word for word in word_transcript
            if chapter_start <= float(word["start"]) and float(word["end"]) <= chapter_end
        ]
        print(f"Original words in chapter: {len(chapter_words)}")
        
        # Apply edits to the chapter
        for edit in chapter_edits["final_chapter_edits"]:
            edit_start = float(edit["start"])
            edit_end = float(edit["end"])
            print(f"Applying edit: {edit_start} - {edit_end}, Reason: {edit['reason']}")
            chapter_words = [
                word for word in chapter_words
                if float(word["start"]) < edit_start or float(word["end"]) > edit_end
            ]
        
        print(f"Words remaining after edits: {len(chapter_words)}")
        
        # Reconstruct the chapter text
        chapter_text = f"Chapter: {chapter['title']}\n" + " ".join(word["text"] for word in chapter_words)
        final_chapters.append(chapter_text)
    
    # Join all chapter texts with double newlines
    return "\n\n".join(final_chapters)

def generate_final_output(chapter_data: List[Dict[str, Any]], word_transcript: List[Dict[str, Any]], edited_transcript: str, edits: List[Dict[str, Any]], review_notes: List[str]) -> Dict:
    """Generate the final edits by processing each chapter individually."""
    final_output = {"final_edits": []}
    # TESTING: Only process first 3 chapters
    chapter_data = chapter_data[:3]
    
    for i, chapter in enumerate(chapter_data):
        chapter_title = chapter.get("title", f"Chapter {i+1}")
        chapter_start = float(chapter["start"])
        chapter_end = float(chapter["end"])
        
        # Reconstruct original chapter transcript
        original_chapter_words = [
            word["text"] for word in word_transcript
            if chapter_start <= float(word["start"]) and float(word["end"]) <= chapter_end
        ]
        original_chapter_transcript = " ".join(original_chapter_words)
        
        # Extract edited chapter transcript
        edited_chapters = edited_transcript.split("Chapter:")
        edited_chapter_transcript = edited_chapters[i+1].strip() if i+1 < len(edited_chapters) else ""
        
        # Filter initial edits for this chapter
        chapter_initial_edits = [edit for edit in edits if edit.get("chapter_title") == chapter_title]
        
        # Get chapter-specific review notes
        chapter_review_notes = review_notes[i] if i < len(review_notes) else ""
        
        # Determine format from chapter title or content
        format_type = "General"  # Default format
        if any(keyword in chapter_title.lower() for keyword in ["interview", "qa", "question", "answer"]):
            format_type = "Q&A/Interview"
        elif any(keyword in chapter_title.lower() for keyword in ["story", "narrative", "tale"]):
            format_type = "Narrative"
        elif any(keyword in chapter_title.lower() for keyword in ["technical", "explanation", "guide"]):
            format_type = "Technical"
        
        # Enhanced format-specific strategies with clear delimiters
        if format_type == "Q&A/Interview":
            strategy = """
            <format_strategy>
            Q&A/Interview Format:
            - Preserve complete question-answer pairs
            - Keep natural transitions between questions
            - Maintain interviewer-interviewee rapport
            - Preserve genuine reactions and emotions
            - Remove content not part of Q&A pairs
            - Keep authentic expressions of enthusiasm
            - Preserve personal connections
            - Remove technical setup noise
            </format_strategy>
            """
        elif format_type == "Narrative":
            strategy = """
            <format_strategy>
            Narrative Format:
            - Focus on story flow and key plot points
            - Preserve authentic reactions and emotions
            - Keep natural story progression
            - Maintain engaging descriptions
            - Remove excessive descriptions
            - Preserve personal connections
            - Keep authentic expressions
            - Remove technical setup noise
            </format_strategy>
            """
        elif format_type == "Technical":
            strategy = """
            <format_strategy>
            Technical Format:
            - Focus on technical accuracy
            - Maintain engaging explanations
            - Preserve authentic reactions
            - Keep natural progression
            - Remove redundant explanations
            - Preserve personal connections
            - Keep authentic expressions
            - Remove technical setup noise
            </format_strategy>
            """
        else:
            strategy = """
            <format_strategy>
            General Format:
            - Focus on clarity and removing redundancy
            - Preserve authentic voice
            - Keep main points and insights
            - Maintain natural flow
            - Remove tangents
            - Preserve personal connections
            - Keep authentic expressions
            - Remove technical setup noise
            </format_strategy>
            """
        
        # Construct the prompt for this chapter with clear delimiters
        prompt = f"""
<task>
You are an expert podcast editor for a {format_type} podcast. Your goal is to enhance the listening experience by removing content that might bore or distract viewers while preserving the authentic voice and important elements.
</task>

{strategy}

<content_guidelines>
REMOVE:
- Technical setup phrases
- Incomplete sentences and thoughts
- Repetitive words/phrases
- Off-topic tangents
- Factually incorrect statements

KEEP:
- Expressions of gratitude and politeness
- Personal connections and authentic reactions
- Key personality traits of speakers
- Main discussion points and arguments
- Natural conversation flow
</content_guidelines>

<quality_requirements>
- Each remaining sentence must be complete and coherent
- Maintain the natural flow of conversation
- Preserve the speaker's authentic voice
- Keep the emotional authenticity
- Ensure content remains engaging
</quality_requirements>

<chapter_data>
Title: {chapter_title}
Original: {original_chapter_transcript}
Initial Edits: {json.dumps(chapter_initial_edits, indent=2)}
Edited: {edited_chapter_transcript}
Review Notes: {chapter_review_notes}
</chapter_data>

<output_requirements>
Provide final edits that:
1. Address specific issues from review notes
2. Follow content guidelines
3. Maintain {format_type} format
4. Include brief reasoning for each edit
5. Only include removal edits (no insertions)
</output_requirements>

<output_format>
Return a JSON object with:
- chapter_title: string
- final_chapter_edits: array of objects with:
  - start: number
  - end: number
  - reason: string
</output_format>
"""
        
        try:
            response = client.responses.create(
                model="o3-mini",
                reasoning={"effort": "high"},
                input=[
                    {"role": "user", "content": prompt}
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "chapter_final_edits",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "chapter_title": {"type": "string"},
                                "final_chapter_edits": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "start": {"type": "number"},
                                            "end": {"type": "number"},
                                            "reason": {"type": "string"}
                                        },
                                        "required": ["start", "end", "reason"],
                                        "additionalProperties": False
                                    },
                                    "strict": True
                                }
                            },
                            "required": ["chapter_title", "final_chapter_edits"],
                            "additionalProperties": False
                        },
                        "strict": True
                    }
                }
            )
            
            chapter_final_edits = json.loads(response.output_text)
            # Filter out any edits where start and end times are the same (insertions)
            chapter_final_edits["final_chapter_edits"] = [
                edit for edit in chapter_final_edits["final_chapter_edits"]
                if edit["end"] > edit["start"]
            ]
            final_output["final_edits"].append(chapter_final_edits)
            
        except Exception as e:
            print(f"Error processing chapter {chapter_title}: {e}")
            final_output["final_edits"].append({
                "chapter_title": chapter_title,
                "final_chapter_edits": [],
                "error": str(e)
            })
    
    # Apply final edits to create the final transcript
    final_transcript = apply_final_edits(word_transcript, chapter_data, final_output["final_edits"])
    
    # Add the final transcript to the output
    final_output["final_transcript"] = final_transcript
    
    return final_output



def process_transcript(input_json_path: str) -> Dict:
    """Process a podcast transcript to improve its quality."""
    try:
        print("\n=== Starting Podcast Transcript Processing ===")
        print(f"Loading input JSON from: {input_json_path}")
        
        # Load the input JSON
        with open(input_json_path, 'r') as f:
            json_data = json.load(f)
        
        print("\n=== Extracting Data from JSON ===")
        # Extract data from the JSON
        chapter_data = get_chapters_from_json(json_data)
        print(f"Found {len(chapter_data)} chapters")
        
        sentence_transcript = get_sen_from_json(json_data)
        print(f"Found {len(sentence_transcript)} sentences")
        
        word_transcript = get_words_from_json(json_data)
        print(f"Found {len(word_transcript)} words")
        
        print("\n=== Analyzing Podcast Format ===")
        # Analyze overall podcast format once (instead of per chapter)
        podcast_format = analyze_podcast_format(sentence_transcript)
        print(f"Detected format: {podcast_format}")
        
        print("\n=== Processing Chapter Edits ===")
        edits = process_chapter_edits(chapter_data, word_transcript, podcast_format)
        print(f"Generated {len(edits)} edit segments")
        
        print("\n=== Reconstructing Transcript ===")
        cleaned_transcript = reconstruct_transcript(chapter_data, word_transcript, edits)
        print("Transcript reconstruction complete")

        print("\n=== Reflecting on Transcript ===")
        reflection_result = reflect_on_transcript(chapter_data, word_transcript, cleaned_transcript, edits)
        print("Reflection analysis complete")
        
        print("\n=== Generating Final Output ===")
        final_output = generate_final_output(chapter_data, word_transcript, cleaned_transcript, edits, reflection_result)
        print("Final output generation complete")
        
        print("\n=== Processing Complete ===")
        return final_output
    
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        return {"error": str(e)}

def main():
    print("\n=== Starting Podcast Editor ===")
    print("Using conda environment: local")
    
    result = process_transcript(input_json_path="/home/tanmay/Desktop/Ukumi_Tanmay/output/sahu.json")
    
    print("\n=== Saving Results ===")
    # Print or save the final output and transcript
    print("\nFinal Output:")
    print(json.dumps(result["final_edits"], indent=2))
    
    print("\nFinal Transcript:")
    print(result["final_transcript"])
    
    # Save the final transcript to a file
    output_path = "/home/tanmay/Desktop/Ukumi_Tanmay/output/final_transcript.txt"
    with open(output_path, "w") as f:
        f.write(result["final_transcript"])
    print(f"\nFinal transcript saved to: {output_path}")
    
    print("\n=== Processing Complete ===")

if __name__ == "__main__":
    main()
