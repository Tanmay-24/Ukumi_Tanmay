import json
import asyncio
import aiohttp
from time import time


OPENAI_API_KEY = ""

# Load input data
with open('/home/tanmay/Desktop/Ukumi_Tanmay/output/chapters.json', 'r') as f:
    data = json.load(f)

chapters = data[0]['chapters']

async def rate_chapter(session, chapter, index):
    """Rate a single podcast chapter asynchronously with enhanced criteria"""
    
    # Enhanced prompt with specific podcast rating criteria
    prompt = f"""You are a podcast critic. Rate the following podcast chapter on a scale of 1-10, where:
1-3: Poor quality content with little value
4-6: Average content with some interesting points
7-8: Good content that engages the listener
9-10: Exceptional content that provides unique insights

Consider these factors:
- Potential listener engagement
- Educational/entertainment value
- Relevance of the topic
- Depth of discussion (based on description)

Chapter Title: "{chapter['title']}"
Chapter Description: "{', '.join(chapter['description'])}"

Provide your rating as a single number (1-10) followed by one concise sentence explaining WHY you gave this rating. Your explanation should directly reflect the rating value.

Format your response exactly like this:
Rating: [number]
Reason: [Your one-sentence explanation that matches the rating level]"""
    
    # Prepare the request payload
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4000,  # Increased slightly for better reasoning
        "temperature": 0.7  # Balanced between creativity and consistency
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    
    start_time = time()
    try:
        async with session.post("https://api.openai.com/v1/chat/completions", 
                               headers=headers, 
                               json=payload) as response:
            result = await response.json()
            
        # Check if the response contains an error
        if 'error' in result:
            print(f"Error for chapter {index}: {result['error']}")
            rating = 5  # Default rating
            reason = f"Rating failed: {result.get('error', {}).get('message', 'Unknown error')}"
        else:
            # Extract rating and reason from response
            response_text = result['choices'][0]['message']['content']
            
            # Improved parsing - looking for specific format
            try:
                lines = response_text.strip().split('\n')
                for line in lines:
                    if line.startswith('Rating:'):
                        rating = int(line.replace('Rating:', '').strip())
                    elif line.startswith('Reason:'):
                        reason = line.replace('Reason:', '').strip()
                
                if 'rating' not in locals() or 'reason' not in locals():
                    # Fallback parsing if the format isn't as expected
                    parts = response_text.split('.')
                    rating_part = parts[0].strip()
                    rating = int(''.join(filter(str.isdigit, rating_part)))
                    
                    reason_parts = parts[1:] if len(parts) > 1 else ["Insufficient information provided"]
                    reason = '.'.join(reason_parts).strip()
            except:
                rating = 5  # Default if parsing fails
                reason = "Failed to parse rating: " + response_text
    except Exception as e:
        print(f"Exception for chapter {index}: {str(e)}")
        rating = 5  # Default rating
        reason = f"Rating failed: {str(e)}"
    
    elapsed = time() - start_time
    
    # Return the rated chapter with timing info
    return {
        "chapter_index": index,
        "title": chapter['title'],
        "start": chapter['start'],
        "end": chapter['end'],
        "rating": rating,
        "reason": reason,
        "processing_time": elapsed
    }

async def main():
    start_total = time()
    
    # Configurable concurrency - adjust based on your API rate limits
    max_concurrent = 5
    
    async with aiohttp.ClientSession() as session:
        # Process chapters in batches for better control
        results = []
        for i in range(0, len(chapters), max_concurrent):
            batch = chapters[i:i+max_concurrent]
            batch_tasks = [rate_chapter(session, chapter, i+idx) for idx, chapter in enumerate(batch)]
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)
            
            # Optional: add a small delay between batches to avoid rate limiting
            if i + max_concurrent < len(chapters):
                await asyncio.sleep(0.5)
    
    # Sort by original chapter order
    rated_chapters = sorted(results, key=lambda x: x['chapter_index'])
    
    # Create the final output
    output = {
        "media_id": data[0]['media_id']['$oid'],
        "target_id": data[0]['target_id'],
        "rated_chapters": [
            {
                "title": c['title'],
                "timestamps": {"start": c['start'], "end": c['end']},
                "rating": c['rating'],
                "reason": c['reason']
            } for c in rated_chapters
        ],
        "total_processing_time": time() - start_total
    }
    
    # Print the result
    print(json.dumps(output, indent=2))
    print(f"Total time: {output['total_processing_time']:.2f} seconds")
    
    # Optionally save to file
    with open('rated_chapters.json', 'w') as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    asyncio.run(main())