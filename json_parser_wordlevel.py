import json

def parse_words(json_data):
    """
    Parse the JSON data to extract punctuated words from the transcript.
    Uses a similar chained get() approach as the sample code for safer navigation.
    """
    words = json_data.get('results', {}).get('channels', [])[0] \
            .get('alternatives', [])[0].get('words', [])
    
    parsed_data = []
    for word in words:
        word_data = {
            "text": word.get('punctuated_word', ''),
            "start": word.get('start', 0),
            "end": word.get('end', 0),
        }
        parsed_data.append(word_data)
    
    return parsed_data

def save_to_txt(parsed_data, output_file):
    """
    Save the parsed transcript to a text file.
    Creates both a simple plaintext version and a detailed version with timestamps.
    """
    # Save detailed version with timestamps
    with open(output_file, "w", encoding="utf-8") as f:
        # First write the complete transcript
        f.write("Complete Transcript:\n")
        f.write("==================\n")
        f.write(" ".join(word['text'] for word in parsed_data))
        f.write("\n\n")
        
        # Then write detailed word information
        f.write("Detailed Word Information:\n")
        f.write("========================\n")
        for word in parsed_data:
            f.write(f"Word: {word['text']}\n")
            f.write(f"Start: {word['start']}, End: {word['end']}\n")
            f.write("\n")

def get_plaintext(parsed_data):
    """
    Get just the plaintext transcript from the parsed data.
    """
    return " ".join(word['text'] for word in parsed_data)

if __name__ == "__main__":
    # Read and parse the JSON file
    try:
        with open("deepgram_response.json", "r", encoding="utf-8") as file:
            json_data = json.load(file)
        
        # Parse the transcript
        result = parse_words(json_data)
        
        # Save detailed output
        output_file = "transcript_output.txt"
        save_to_txt(result, output_file)
        print(f"Detailed transcript saved to {output_file}")
        
        # Print plaintext transcript
        print("\nPlaintext transcript:")
        print(get_plaintext(result))
        
    except FileNotFoundError:
        print("Error: Input JSON file not found")
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in input file")
    except Exception as e:
        print(f"Error: {str(e)}")