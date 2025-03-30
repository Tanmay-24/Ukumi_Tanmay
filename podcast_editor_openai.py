import json
import os
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import openai

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)


class RemovalSegment(BaseModel):
    """A segment of transcript to be removed."""
    start: float
    end: float
    reason: str
    chapter_title: Optional[str] = None
    text: Optional[str] = None 

class EditResult(BaseModel):
    """Result of content or filler word editing, containing segments to remove."""
    removed: List[RemovalSegment] = [] 

class ReflectionResult(BaseModel):
    """Result of transcript reflection, containing assessment and review notes."""
    coherence_score: int = Field(description="Score from 1-10 on transcript coherence")
    review_notes: str = Field(description="Detailed review notes on issues found")



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
            print(result)
            combined_removals.extend(result)
            break  # Process only the first chapter

        except Exception as e:
            return Exception(f"Error processing chapter edits: {e}")

    return combined_removals

def reconstruct_transcript(sentence_transcript: List[Dict], edit_result: Optional[Dict]) -> str:
    """Reconstruct the transcript after applying edits."""
    
    # Extract removal segments
    all_removals = edit_result.get("removed", []) if edit_result else []
    
    # Create a clean transcript by filtering out the removal segments
    clean_sentences = []
    for sentence in sentence_transcript:
        sentence_start = float(sentence.get("start", 0))
        sentence_end = float(sentence.get("end", 0))
        
        # Check if this sentence should be removed based on content edit segments
        should_remove = False
        for removal in all_removals:
            removal_start = float(removal.get("start", 0))
            removal_end = float(removal.get("end", 0))
            
            # If there's a significant overlap, remove the sentence
            if (max(sentence_start, removal_start) < min(sentence_end, removal_end)) and \
               (min(sentence_end, removal_end) - max(sentence_start, removal_start)) > 0.5 * (sentence_end - sentence_start):
                should_remove = True
                break
        
        if not should_remove:
            clean_sentences.append(sentence)
    
    # Build the clean transcript string
    clean_transcript = "\n".join([sentence.get("text", "") for sentence in clean_sentences])
    return clean_transcript

def reflect_on_transcript(original_transcript: str, edited_transcript: str) -> ReflectionResult:
    """Reflect on the quality of the edited transcript using the OpenAI API with o3-mini model."""
    
    prompt = """You are an expert podcast editor evaluating transcript quality.
Compare the original and edited transcripts to evaluate improvement in clarity, flow, and overall quality.
Provide a structured assessment with specific scores and detailed review notes.

Your analysis should be a valid JSON object with this structure:
{
  "coherence_score": integer from 1-10,
  "flow_score": integer from 1-10,
  "overall_score": integer from 1-10,
  "review_notes": "detailed notes on issues found",
  "requires_revision": boolean,
  "revision_suggestions": [
    {"issue": "specific issue", "suggestion": "how to fix it", "severity": integer from 1-5}
  ]
}"""

    try:
        response = client.responses.create(
            model="o3-mini",
            reasoning={"effort": "medium"},
            input=[
                {"role": "user", "content": prompt + f"\n\nORIGINAL TRANSCRIPT:\n{original_transcript}\n\nEDITED TRANSCRIPT:\n{edited_transcript}"}
            ]
        )
        
        result = json.loads(response.output_text)
        reflection_result = ReflectionResult(
            coherence_score=result.get("coherence_score", 5),
            flow_score=result.get("flow_score", 5),
            overall_score=result.get("overall_score", 5),
            review_notes=result.get("review_notes", "No notes provided."),
            requires_revision=result.get("requires_revision", False),
            revision_suggestions=result.get("revision_suggestions", [])
        )
        

        return reflection_result
    except json.JSONDecodeError as e:
        return ReflectionResult(
            coherence_score=5,
            flow_score=5,
            overall_score=5,
            review_notes=f"Error parsing reflection results: {e}",
            requires_revision=False,
            revision_suggestions=[]
        )
    except Exception as e:
        return ReflectionResult(
            coherence_score=5,
            flow_score=5,
            overall_score=5,
            review_notes=f"Error during reflection: {e}",
            requires_revision=False,
            revision_suggestions=[]
        )

def generate_final_output(edits: Optional[Dict], clean_transcript: Optional[str], reflection_result: Optional[ReflectionResult]) -> Dict:
    """Generate the final output including all edits and the clean transcript."""
    # Extract removal segments
    all_removals = edits.get("removed", []) if edits else []
    
    # Categorize removals as content or filler
    content_removals = []
    filler_removals = []
    
    for removal in all_removals:
        if "text" in removal and isinstance(removal.get("text"), str):
            filler_removals.append(removal)
        else:
            content_removals.append(removal)
    
    # Count statistics
    content_edit_count = len(content_removals)
    filler_edit_count = len(filler_removals)
    total_edit_count = content_edit_count + filler_edit_count
    
    # Create the final output dictionary
    output = {
        "edits": {
            "content_edits": content_removals,
            "filler_edits": filler_removals,
            "total_edits": total_edit_count
        },
        "statistics": {
            "content_edit_count": content_edit_count,
            "filler_edit_count": filler_edit_count
        },
        "clean_transcript": clean_transcript or ""
    }
    
    # Add reflection results if available
    if reflection_result:
        output["reflection"] = reflection_result.dict()
    return output

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
        # cleaned_transcript = reconstruct_transcript(sentence_transcript, edits)
        # reflection_result = reflect_on_transcript(original_transcript, cleaned_transcript)
        # final_output = generate_final_output(edits, original_transcript, reflection_result)
        
        return edits
    
    except Exception as e:
        return {"error": str(e)}

def main():
    process_transcript(input_json_path="/home/tanmay/Desktop/Ukumi_Tanmay/output/sahu.json")

if __name__ == "__main__":
    main() 