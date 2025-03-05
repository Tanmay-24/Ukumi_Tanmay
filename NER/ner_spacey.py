import json
import re
from typing import Dict, List, Any, Union

import spacy

class PodcastTranscriptAnalyzer:
    def __init__(self):
        """
        Initialize the analyzer with SpaCy NLP model
        """
        try:
            # Load the large English model
            self.nlp = spacy.load("en_core_web_trf")
        except OSError:
            print("Downloading SpaCy English model...")
            raise ValueError("SpaCy English model not found")
    
    def clean_and_normalize_text(self, text: str) -> str:
        """
        Clean and normalize input text
        """
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text)
        
        
        return text.strip()
    
    def extract_text_from_transcript(self, transcript_data: Union[Dict, List, str]) -> str:
        """
        Extract text from different transcript formats
        """
        try:
            all_text = ""
            
            # Handle new JSON structure
            if isinstance(transcript_data, dict) and 'data' in transcript_data:
                for item in transcript_data['data']:
                    if 'channels' in item:
                        paragraph_transcript = item['channels'].get('paragraph_level_transcript', [])
                        all_text = " ".join([p.get('text', '') for p in paragraph_transcript])
                        break
            
            # Handle list of dictionaries (old format)
            elif isinstance(transcript_data, list) and len(transcript_data) > 0:
                for item in transcript_data:
                    if isinstance(item, dict) and 'text' in item:
                        if isinstance(item['text'], list):
                            all_text += " ".join([
                                segment.get("text", "") if isinstance(segment, dict) 
                                else str(segment) 
                                for segment in item['text']
                            ])
                        else:
                            all_text += str(item['text']) + " "
            
            # Handle single dictionary (old format)
            elif isinstance(transcript_data, dict):
                if 'text' in transcript_data:
                    all_text = str(transcript_data['text'])
            
            # Handle string
            elif isinstance(transcript_data, str):
                all_text = transcript_data
            
            # Clean and normalize text
            return self.clean_and_normalize_text(all_text)
        
        except Exception as e:
            print(f"Error extracting text: {str(e)}")
            return ""
    
    def process_ner_results(self, doc) -> List[Dict[str, Any]]:
        """
        Process and clean NER results from SpaCy
        """
        processed_entities = []
        seen_entities = set()
        
        # Entity description mapping relevant for podcast viewers
        entity_descriptions = {
            'PERSON': 'Person',
            'ORG': 'Organization', 
            'LOC': 'Location',
            'GPE': 'Geopolitical Entity',
            'EVENT': 'Named Events (e.g., Conferences, Historical Events)',
            'WORK_OF_ART': 'Titles of Books, Songs, Movies, etc.',
            'DATE': 'Dates Mentioned',
            'TIME': 'Specific Times Mentioned'
        }
        
        for ent in doc.ents:
            # Extract entity details
            entity_text = ent.text.strip()
            entity_label = ent.label_
            
            # Skip very short entities
            if len(entity_text) <= 1:
                continue
            
            # Create a unique key to avoid duplicates
            entity_key = (entity_text.lower(), entity_label)
            
            # Avoid duplicates
            if entity_key in seen_entities:
                continue
            
            # Only include entities that are in our entity_descriptions dictionary
            if entity_label in entity_descriptions:
                description = entity_descriptions[entity_label]
                
                processed_entities.append({
                    "text": entity_text,
                    "label": entity_label,
                    "description": description
                })
                
                seen_entities.add(entity_key)
        
        # Sort entities by text length (descending) to prioritize more specific entities
        processed_entities.sort(key=lambda x: len(x['text']), reverse=True)
        
        return processed_entities
    
    def analyze_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from text using SpaCy
        """
        # Process the entire text at once (SpaCy is more efficient)
        doc = self.nlp(text)
        
        # Process and clean NER results
        processed_results = self.process_ner_results(doc)
        
        return processed_results
    
    def analyze(self, transcript_data: Union[Dict, List, str]) -> Dict[str, Any]:
        """
        Main analysis method
        """
        try:
            # Extract media_id if available
            media_id = None
            if isinstance(transcript_data, dict):
                media_id = transcript_data.get('media_id')
            
            # Extract and clean text
            raw_text = self.extract_text_from_transcript(transcript_data)
            if not raw_text:
                return {"error": "Could not extract text from the transcript."}
            
            # Perform NER analysis
            entities = self.analyze_entities(raw_text)
            
            # Compile results with more information
            results = {
                "media_id": media_id,
                "analysis_summary": {
                    "transcript_length": len(raw_text),
                    "entity_count": len(entities),
                    "top_entities": [e["text"] for e in entities[:10]],
                    "entity_categories": list(set(e["description"] for e in entities))
                },
                "named_entities": entities
            }
            
            return results
        
        except Exception as e:
            print(f"Error in transcript analysis: {str(e)}")
            return {"error": f"Analysis failed: {str(e)}"}

def main():
    # Initialize analyzer
    analyzer = PodcastTranscriptAnalyzer()
    
    # Read JSON input file
    input_file = "/home/tanmay/Desktop/Ukumi_Tanmay/output/sahu.json"
    try:
        with open(input_file, 'r') as f:
            input_data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {input_file}")
        return
    except FileNotFoundError:
        print(f"Error: File {input_file} not found")
        return

    # Analyze the transcript
    try:
        results = analyzer.analyze(input_data)
        
        # Print results
        print(json.dumps(results, indent=2))
        with open("ner_analysis.json", 'w') as f:
            json.dump(results, f, indent=2)
    
    except Exception as e:
        print(f"Analysis failed: {e}")

if __name__ == "__main__":
    main()
