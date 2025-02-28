import json
import re
import pandas as pd
from collections import Counter
from typing import Dict, List, Any, Union, Tuple
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PodcastTranscriptAnalyzer:
    """
    A system for analyzing podcast transcripts to extract entities
    and key phrases for content enhancement using Hugging Face transformers.
    """
    
    def __init__(self, use_advanced_models=False):
        """
        Initialize the analyzer with necessary NLP models.
        
        Args:
            use_advanced_models: Whether to use larger, more accurate models
        """
        # Initialize NLP libraries
        try:
            # Load Hugging Face NER model
            if use_advanced_models:
                self.tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
                self.model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
                logger.info("Loaded advanced transformer-based NER model")
            else:
                self.tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
                self.model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
                logger.info("Loaded basic NER model")
            
            self.nlp = pipeline("ner", model=self.model, tokenizer=self.tokenizer, aggregation_strategy="simple")
            
            # Initialize NLTK for tokenization and n-grams
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english'))
            
        except Exception as e:
            logger.error(f"Error initializing NLP models: {str(e)}")
            raise
    
    def extract_text_from_transcript(self, transcript_data: Union[Dict, List, str]) -> str:
        """
        Extract text from different transcript formats.
        
        Args:
            transcript_data: The transcript data in various possible formats
            
        Returns:
            A string containing all the text content
        """
        try:
            all_text = ""
            
            # Handle list of dictionaries (as in the sen_AI.json file)
            if isinstance(transcript_data, list) and len(transcript_data) > 0 and isinstance(transcript_data[0], dict):
                for item in transcript_data:
                    if 'text' in item and isinstance(item['text'], list):
                        all_text += " ".join([segment.get("text", "") for segment in item['text'] if isinstance(segment, dict)])
            
            # Handle single dictionary with 'text' key
            elif isinstance(transcript_data, dict) and 'text' in transcript_data:
                if isinstance(transcript_data['text'], list):
                    all_text = " ".join([segment.get("text", "") for segment in transcript_data['text'] if isinstance(segment, dict)])
                elif isinstance(transcript_data['text'], str):
                    all_text = transcript_data['text']
            
            # Handle single string
            elif isinstance(transcript_data, str):
                all_text = transcript_data
            
            # Handle JSON string
            elif isinstance(transcript_data, str) and (transcript_data.startswith('{') or transcript_data.startswith('[')):
                try:
                    parsed_data = json.loads(transcript_data)
                    return self.extract_text_from_transcript(parsed_data)
                except json.JSONDecodeError:
                    all_text = transcript_data
            
            if not all_text:
                logger.warning("Could not extract text from transcript - unknown format")
                return ""
            
            return all_text
        
        except Exception as e:
            logger.error(f"Error extracting text from transcript: {str(e)}")
            return ""
            
    def clean_text(self, text: str, preserve_case: bool = True) -> str:
        """
        Clean and normalize text for analysis.
        
        Args:
            text: Raw text from transcript
            preserve_case: Whether to preserve the original case (for NER) or convert to lowercase
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove timestamps like [00:15:23] or (15:23)
        text = re.sub(r'[\[\(]?\d{1,2}:\d{2}(:\d{2})?[\]\)]?', '', text)
        
        # Remove speaker identification patterns like "Speaker 1:" or "[Host]:"
        text = re.sub(r'(\[|\()?((speaker|host|guest|interviewer|interviewee|moderator)\s*\d*|[a-z]+)(\:|\]|\))\s*', '', text)
        
        # Remove special characters but keep structure
        text = re.sub(r'[^\w\s\.\,\?\!]', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        if not preserve_case:
            text = text.lower()
        
        return text
    
    def process_ner_results(self, ner_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process and clean NER results from Hugging Face pipeline.
        
        Args:
            ner_results: Raw NER results from the Hugging Face pipeline
            
        Returns:
            Cleaned list of entity dictionaries
        """
        processed_entities = []
        seen_entities = set()
        
        for entity in ner_results:
            # Create a unique identifier for deduplication
            entity_key = (entity['word'].lower(), entity['entity_group'])
            
            # Skip if already seen or too short
            if entity_key in seen_entities or len(entity['word']) <= 1:
                continue
                
            # Add to processed entities
            processed_entities.append({
                "text": entity['word'],
                "label": entity['entity_group'],
                "score": float(entity['score']),  # Convert float32 to regular Python float
                "start_char": entity['start'],
                "description": self.get_entity_description(entity['entity_group'])
            })
            
            seen_entities.add(entity_key)
            
        return processed_entities
        
    def get_entity_description(self, label: str) -> str:
        """
        Get a human-readable description for an entity label.
        
        Args:
            label: Entity label from the NER model
            
        Returns:
            Human-readable description
        """
        descriptions = {
            'PER': 'Person',
            'PERSON': 'Person',
            'ORG': 'Organization',
            'LOC': 'Location',
            'LOCATION': 'Location',
            'MISC': 'Miscellaneous',
            'O': 'Other'
        }
        
        return descriptions.get(label, label)
    
    def analyze_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from text using Hugging Face NER pipeline.
        
        Args:
            text: Cleaned text from transcript
            
        Returns:
            List of entity dictionaries
        """
        # Split text into chunks to avoid exceeding model token limits
        # BERT models typically have a limit of 512 tokens
        words = text.split()
        chunk_size = 200  # A conservative chunk size
        chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
        
        all_entities = []
        offset = 0
        
        for chunk in chunks:
            if not chunk.strip():
                continue
                
            # Process chunk with NER pipeline
            chunk_results = self.nlp(chunk)
            
            # Debug: Print chunk and its results
            logger.debug(f"Chunk: {chunk}")
            logger.debug(f"Chunk results: {chunk_results}")
            
            # Adjust start_char positions based on current offset
            for entity in chunk_results:
                entity['start'] += offset
                entity['end'] += offset
                
            all_entities.extend(chunk_results)
            offset += len(chunk) + 1  # +1 for the space between chunks
        
        # Debug: Print all entities before processing
        logger.debug(f"All entities before processing: {all_entities}")
        
        # Process and clean NER results
        processed_results = self.process_ner_results(all_entities)
        
        # Debug: Print processed results
        logger.debug(f"Processed entities: {processed_results}")
        
        return processed_results
    
    def extract_key_phrases(self, text: str, min_phrase_length=2, max_phrase_length=4) -> List[str]:
        """
        Extract key phrases using NLTK.
        
        Args:
            text: Cleaned text from transcript
            min_phrase_length: Minimum words in a phrase
            max_phrase_length: Maximum words in a phrase
            
        Returns:
            List of key phrases
        """
        # Tokenize text
        tokens = word_tokenize(text)
        logger.debug(f"Tokenized text: {tokens[:20]}...")  # Log first 20 tokens
        
        # Filter out stopwords and short tokens
        filtered_tokens = [token.lower() for token in tokens if token.lower() not in self.stop_words 
                          and len(token) > 2 and token.isalpha()]
        logger.debug(f"Filtered tokens: {filtered_tokens[:20]}...")  # Log first 20 filtered tokens
        
        # Extract n-grams
        all_ngrams = []
        for n in range(min_phrase_length, min(max_phrase_length + 1, len(filtered_tokens))):
            ngrams_n = list(ngrams(filtered_tokens, n))
            all_ngrams.extend([' '.join(gram) for gram in ngrams_n])
            logger.debug(f"{n}-grams (first 5): {ngrams_n[:5]}")
        
        # Count phrase frequency
        phrase_counter = Counter(all_ngrams)
        logger.debug(f"Top 10 phrases by frequency: {phrase_counter.most_common(10)}")
        
        # Return top phrases, filtering out near-duplicates
        unique_phrases = []
        for phrase, count in phrase_counter.most_common(50):  # Increased from 30 to 50
            if count > 0:  # Changed from count > 1 to count > 0
                # Check if this phrase is not a subset of an already included phrase
                if not any(phrase in existing for existing in unique_phrases):
                    unique_phrases.append(phrase)
        
        logger.debug(f"Final unique phrases: {unique_phrases}")
        return unique_phrases[:15]  # Return top 15 unique phrases
    
    def analyze(self, transcript_data: Union[Dict, List, str]) -> Dict[str, Any]:
        """
        Main method to perform analysis on a podcast transcript using Hugging Face models.
        
        Args:
            transcript_data: Podcast transcript in various possible formats
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Extract and clean text
            raw_text = self.extract_text_from_transcript(transcript_data)
            if not raw_text:
                return {"error": "Could not extract text from the provided transcript data."}
                
            # Get a sample for logging
            text_sample = raw_text[:100] + "..." if len(raw_text) > 100 else raw_text
            logger.info(f"Extracted text sample: {text_sample}")
            
            # Clean text for NER (preserve case)
            cleaned_text_ner = self.clean_text(raw_text, preserve_case=True)
            
            # Clean text for key phrase extraction (lowercase)
            cleaned_text_phrases = self.clean_text(raw_text, preserve_case=False)
            
            # Perform analyses
            entities = self.analyze_entities(cleaned_text_ner)
            key_phrases = self.extract_key_phrases(cleaned_text_phrases)
            
            # Compile results
            results = {
                "analysis_summary": {
                    "transcript_length": len(raw_text),
                    "entity_count": len(entities),
                    "top_entities": [e["text"] for e in entities[:5]] if entities else []
                },
                "named_entities": entities,
                "key_phrases": key_phrases
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in transcript analysis: {str(e)}")
            return {"error": f"Analysis failed: {str(e)}"}


# Example usage
def main():
    # Initialize analyzer
    analyzer = PodcastTranscriptAnalyzer()
    
    # Use the example text directly
    example_text = """
    In this podcast episode, we discussed artificial intelligence with Dr. Jane Smith from MIT. 
    She explained how deep learning models like BERT and GPT have transformed natural language processing.
    We also talked about the ethical implications of AI in healthcare with representatives from Google and Microsoft.
    """
    with open("/home/tanmay/Desktop/Ukumi_Tanmay/output/sen_AI.json",'r') as f:
        text=json.load(f)
    
    # Analyze the transcript
    results = analyzer.analyze(text)
    
    # Print results to console
    print(json.dumps(results, indent=2))

    # Save results to a JSON file
    output_file = "/home/tanmay/Desktop/Ukumi_Tanmay/output/analysis_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nAnalysis results have been saved to: {output_file}")

if __name__ == "__main__":
    main()
