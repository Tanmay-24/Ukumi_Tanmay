import json
import re
from typing import Dict, List, Any, Union

# Transformer and NLP libraries
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

class PodcastTranscriptAnalyzer:
    def __init__(self):
        """
        Initialize the analyzer with NLP models
        """
        # Use a more specialized and reliable NER model
        model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
        
        try:
            # Explicitly set device to CPU
            device = torch.device('cpu')
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(model_name)
            
            # Create pipeline with explicit device setting
            self.nlp = pipeline(
                "ner", 
                model=self.model, 
                tokenizer=self.tokenizer,
                aggregation_strategy="simple",
                device=device
            )
        except Exception as e:
            print(f"Error initializing NLP models: {e}")
            raise
    
    def clean_and_normalize_text(self, text: str) -> str:
        """
        Clean and normalize input text
        """
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize some common cases
        text = text.replace('ister', 'ister ')  # Fix partial words
        text = text.replace('ister Musk', 'Mister Musk')
        
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
    
    def process_ner_results(self, ner_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process and clean NER results
        """
        processed_entities = []
        seen_entities = set()
        
        # Comprehensive entity description mapping
        entity_descriptions = {
            'PER': 'Person',
            'ORG': 'Organization', 
            'LOC': 'Location',
            'MISC': 'Miscellaneous'
        }
        
        for entity in ner_results:
            # Extract entity details
            entity_text = entity.get('word', entity.get('text', '')).strip()
            entity_label = entity.get('entity', entity.get('label', 'O'))
            entity_score = float(entity.get('score', 1.0))
            
            # Skip very short entities
            if len(entity_text) <= 1:
                continue
            
            # Create a unique key to avoid duplicates
            entity_key = (entity_text.lower(), entity_label)
            
            # Avoid duplicates
            if entity_key in seen_entities:
                continue
            
            # Determine description
            description = entity_descriptions.get(entity_label, 'Other')
            
            processed_entities.append({
                "text": entity_text,
                "label": entity_label,
                "score": entity_score,
                "description": description
            })
            
            seen_entities.add(entity_key)
        
        # Sort entities by score in descending order
        processed_entities.sort(key=lambda x: x['score'], reverse=True)
        
        return processed_entities
    
    def analyze_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from text
        """
        # Process text in chunks to avoid token limits
        words = text.split()
        chunk_size = 200
        chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
        
        all_entities = []
        
        for chunk in chunks:
            if not chunk.strip():
                continue
            
            try:
                chunk_results = self.nlp(chunk)
                all_entities.extend(chunk_results)
            except Exception as e:
                print(f"Error processing chunk: {e}")
        
        # Process and clean NER results
        processed_results = self.process_ner_results(all_entities)
        
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