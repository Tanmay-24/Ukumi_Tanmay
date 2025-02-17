import json
from typing import List, Dict
import os

def parse_transcripts(sentence_transcript: List[Dict], word_transcript: List[Dict]) -> List[Dict]:
    """
    Combines sentence and word level transcripts by keeping sentence text but using
    word level timestamp information.
    
    Args:
        sentence_transcript: List of sentence level transcript entries
        word_transcript: List of word level transcript entries
        
    Returns:
        List of combined transcript entries with sentence text and word timestamps
    """
    
    def find_word_boundaries(sentence: Dict, word_entries: List[Dict]) -> tuple:
        """Find first and last word that matches the sentence timing."""
        sentence_start = sentence['start']
        sentence_end = sentence['end']
        
        first_word = None
        last_word = None
        
        for word in word_entries:
            # Find first word
            if abs(word['start'] - sentence_start) < 0.1 and first_word is None:
                first_word = word
            
            # Find last word that ends near sentence end
            if abs(word['end'] - sentence_end) < 0.1:
                last_word = word
                break
                
        return first_word, last_word

    combined_transcript = []
    
    # Extract just the text entries
    sentences = sentence_transcript[0]['text']
    words = word_transcript[0]['text']
    
    for sentence in sentences:
        first_word, last_word = find_word_boundaries(sentence, words)
        
        if first_word and last_word:
            # Create new entry with sentence text but word timestamps
            combined_entry = {
                'id': sentence['id'],
                'start': first_word['start'],
                'formatted_start': first_word['formatted_start'],
                'end': last_word['end'],
                'formatted_end': last_word['formatted_end'],
                'text': sentence['text']
            }
            combined_transcript.append(combined_entry)
    
    return combined_transcript

def write_transcript_to_file(transcript: List[Dict], output_file: str):
    """
    Writes the transcript to a text file in a readable format.
    
    Args:
        transcript: List of transcript entries
        output_file: Path to output file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in transcript:
            f.write(f"[{entry['start']} to {entry['end']}] ")
            f.write(f"{entry['text']}\n")

def main():
    # Read input files
    with open('/home/tanmay/Desktop/Ukumi_Tanmay/output/sen_AI.json', 'r', encoding='utf-8') as f:
        sentence_transcript = json.load(f)
    
    with open('/home/tanmay/Desktop/Ukumi_Tanmay/output/words_AI.json', 'r', encoding='utf-8') as f:
        word_transcript = json.load(f)
    
    # Process transcripts
    combined_transcript = parse_transcripts(sentence_transcript, word_transcript)
    
    # Write output
    write_transcript_to_file(combined_transcript, 'combined_transcript.txt')

if __name__ == "__main__":
    main()