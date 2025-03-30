import json
import os
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
import openai

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)

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
    """Process both content and filler word edits for a chapter using the OpenAI API with the o3-mini model.
       If the chapter is largely non-essential or filled with disjointed phrases, consider skipping the entire chapter."""
    combined_removals = []
    for chapter in chapter_data:
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
        
        format_lower = format_type.lower()
        if "q&a" in format_lower or "interview" in format_lower:
            strategy = (
                "Focus on preserving complete question-answer pairs. Remove content not part of a Q&A pair or essential bridging statement."
            )
        elif "narrative" in format_lower:
            strategy = (
                "Focus on story flow and key plot points. Remove excessive descriptions or tangential sub-plots."
            )
        elif "technical" in format_lower:
            strategy = (
                "Focus on technical accuracy and clarity. Remove redundant explanations while ensuring completeness."
            )
        else:
            strategy = (
                "Focus on improving clarity and removing redundancy. Preserve main points and unique insights. Remove tangents and verbosity."
            )
        
        # Updated prompt with enhanced instructions
        prompt = f"""
You are an expert podcast editor. Your task is to analyze the following podcast chapter transcript and determine the best editing approach to create a concise and engaging transcript.

Your analysis must:
1. Identify segments within the chapter that should be removed because they contain filler words (e.g., "um", "uh", "like", "you know", "sort of", "kind of"), false starts, or disjointed phrasing.
2. Evaluate if the entire chapter is non-essential or composed predominantly of filler content; if so, mark the entire chapter for removal.
3. Preserve all content that is vital for understanding, including key points, main arguments, unique insights, important examples, and necessary context.
4. Ensure that removal segments do not overlap and use exact source timestamps without rounding.

INPUT:
- A string representation of a single chapter's transcript, including:
    * Chapter title, start time, and end time.
    * A series of sentences/words, each with its own start and end timestamp.

OUTPUT:
Return a JSON object with combined edits that include both content and filler word removal segments. If your analysis suggests the chapter should be entirely skipped, output an edit covering the whole chapter's time range and explain your decision in the "reason" field.

FORMAT: {format_type}
STRATEGY: {strategy}

Additional Guidelines:
- Think step-by-step like a podcast viewer: retain segments that add value and are likely to engage the listener.
- When determining removals, consider if skipping the entire chapter is warranted because it lacks meaningful content.
- Justify every removal decision with a clear explanation in the "reason" field.
- Follow the exact timestamp protocol: use the exact start time of the first sentence and the exact end time of the last sentence in any removal segment.

Transcript:
{transcript_str}
"""
        
        try:
            response = client.responses.create(
                model="o3-mini",
                reasoning={"effort": "medium"},
                input=[
                    {"role": "user", "content": "Think step-by-step as a podcast viewer and only keep those segments which a viewer might want to listen to.\n" + prompt + "\nTranscript:\n" + transcript_str}
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
            break  # Process only the first chapter

        except Exception as e:
            return Exception(f"Error processing chapter edits: {e}")
        
    return combined_removals

def reconstruct_transcript(chapter_data: list[dict], word_transcript: list[dict], edit_result: List[Dict[str, Any]]) -> str:
    """
    Reconstruct the refined transcript by applying chapter-level removal edits on a word-level transcript.
    
    For each chapter:
      - Extract the words that fall within the chapter's time range.
      - Remove words whose timestamp overlaps >50% with any removal segment for that chapter.
      - Prepend a chapter header and join the remaining words to form a refined chapter transcript.
    
    Args:
        chapter_data: A list of chapters, each with at least "start", "end", and "title" keys.
        word_transcript: A list of word-level transcript segments, each with "start", "end", and "text".
        edit_result: A list of dictionaries containing the chapter edit removals.
    
    Returns:
        A string of the refined transcript, with each chapter processed separately.
    """

    all_removals = edit_result
    refined_chapters = []
    
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
    """
    Analyze the coherence, flow, and broken sentences for each chapter, comparing the original and edited transcripts.
    
    Args:
        chapter_data (List[Dict[str, Any]]): List of chapters with their metadata.
        word_transcript (List[Dict[str, Any]]): Word-level transcript with timestamps.
        edited_transcript (str): The edited podcast transcript.
        edits (List[Dict[str, Any]]): List of edits made to the transcript.
    
    Returns:
        str: A string containing reflection notes for each chapter.
    """
    edited_chapters = edited_transcript.split("Chapter:")
    reflection_notes = []
    
    for i, chapter in enumerate(chapter_data, 1):
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
            break
        except Exception as e:
            reflection_notes.append(f"Chapter {i} Reflection: Error occurred during analysis - {str(e)}\n")
    
    return "\n\n".join(reflection_notes)

def generate_final_output(chapter_data: List[Dict[str, Any]], word_transcript: List[Dict[str, Any]], edited_transcript: str, edits: List[Dict[str, Any]], review_notes: List[str]) -> Dict:
    """
    Generate the final edits by processing each chapter individually, considering the original transcript,
    edited transcript, initial edits, and chapter-specific review notes.
    
    Args:
        chapter_data: List of chapters with their metadata.
        word_transcript: Word-level transcript with timestamps.
        edited_transcript: The edited podcast transcript.
        edits: List of initial edits made to the transcript.
        review_notes: List of review notes for each chapter.
    
    Returns:
        A dictionary containing final edits for each chapter.
    """
    final_output = {"final_edits": []}
    
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
        
        # Construct the prompt for this chapter
        prompt = f"""
        You are an expert podcast editor. Analyze the following chapter data and provide final edits:

        Chapter Title: {chapter_title}
        
        Original Transcript:
        {original_chapter_transcript}
        
        Initial Edits:
        {json.dumps(chapter_initial_edits, indent=2)}

        Edited Transcript by applying Initial Edits Original Transcript:
        {edited_chapter_transcript}
        
        Review Notes on Edited Transcript:
        {chapter_review_notes}
        
        Based on this information, provide final edits for this chapter. Focus on addressing the specific issues mentioned in the review notes. For each edit, provide detailed reasoning.

        IMPORTANT: Only provide removal edits. Do not suggest any insertions.
        Ensure that your edits improve clarity, flow, and content preservation while addressing the issues raised in the review notes.
        """
        
        try:
            response = client.responses.create(
                model="o3-mini",
                reasoning={"effort": "medium"},
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
            with open("/home/tanmay/Desktop/Ukumi_Tanmay/1.json", 'w') as f:
                json.dump(final_output, f, indent=2)
            break
        except Exception as e:
            print(f"Error processing chapter {chapter_title}: {e}")
            final_output["final_edits"].append({
                "chapter_title": chapter_title,
                "final_chapter_edits": [],
                "error": str(e)
            })
    
    return final_output



def process_transcript(input_json_path: str) -> Dict:
    """Process a podcast transcript to improve its quality."""
    try:
        # Load the input JSON
        with open(input_json_path, 'r') as f:
            json_data = json.load(f)
        
        # Extract data from the JSON
        chapter_data = get_chapters_from_json(json_data)
        sentence_transcript = get_sen_from_json(json_data)
        word_transcript = get_words_from_json(json_data)
        
        
        # Analyze overall podcast format once (instead of per chapter)
        podcast_format = analyze_podcast_format(sentence_transcript)
        edits = process_chapter_edits(chapter_data, word_transcript, podcast_format)
        cleaned_transcript = reconstruct_transcript(chapter_data,word_transcript, edits)

        reflection_result = reflect_on_transcript(chapter_data,word_transcript,cleaned_transcript,edits)
        final_output = generate_final_output(chapter_data,word_transcript,cleaned_transcript,edits,reflection_result)
        print(final_output)
        return final_output
    
    except Exception as e:
        return {"error": str(e)}

def main():
    process_transcript(input_json_path="/home/tanmay/Desktop/Ukumi_Tanmay/output/sahu.json")

if __name__ == "__main__":
    main()
