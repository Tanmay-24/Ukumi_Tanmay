import json

def parse_paragraphs(json_data):
    paragraphs = json_data.get('results', {}).get('channels', [])[0] \
        .get('alternatives', [])[0].get('paragraphs', {}).get('paragraphs', [])
    duration= json_data.get('metadata', {}).get('duration', [])
    print(f"Duration: {duration}")    
    
    parsed_data = []
    for para in paragraphs:
        paragraph_data = {
            "paragraph_start": para.get("start"),
            "paragraph_end": para.get("end"),
            "num_words": para.get("num_words"),
            "sentences": []
        }
        sentences = para.get("sentences", [])
        for sentence in sentences:
            sentence_data = {
                "text": sentence.get("text"),
                "start": sentence.get("start"),
                "end": sentence.get("end")
            }
            paragraph_data["sentences"].append(sentence_data)
        
        parsed_data.append(paragraph_data)
    return parsed_data

def save_to_txt(parsed_data, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for idx, para in enumerate(parsed_data, start=1):
            for sentence in para["sentences"]:
                f.write(f"{sentence['text']}")
                f.write(f" {sentence['start']},{sentence['end']}\n")
            f.write("\n")





if __name__ == "__main__":
    with open("/home/tanmay/Desktop/Ukumi_Tanmay/deepgram_response_riverside.json", "r") as file:
        json_data = json.load(file)




    result = parse_paragraphs(json_data)
    output_file = "riverside_shri.txt"
    save_to_txt(result, output_file)
    print(f"Data saved to {output_file}")
