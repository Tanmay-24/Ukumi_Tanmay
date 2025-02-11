from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.fal import FalTools
import json
from dotenv import load_dotenv
load_dotenv()

#  type, media type,
# plan by tom
broll_agent = Agent(
    name="B-Roll Suggestion Agent",
    model=OpenAIChat(id="gpt-4o"),
    tools=[FalTools("fal-ai/hunyuan-video")],
    description="You suggest relevant B-roll transitions for podcast videos based on transcripts",
    instructions=[
        "Analyze the transcript to identify key moments needing B-roll",
        "For each suggestion:",
        "1. Use the exact timestamp from the transcript",
        "2. Create a visual prompt matching the content",
        "3. Keep prompts concise (e.g., 'AI robots in congressional hearing')",
        "4. Output JSON array with prompt, start/end timestamps,use the `generate_media` tool to create the video for each brool suggestions Return the URL in the json to the user.",
        "Don't convert video URL to markdown or anything else.",
        "Maintain chronological order from the transcript",
        "Prioritize transitions between speakers and topic changes",
        "Include 3-5 most important suggestions for a 30min podcast"
        "each generated video should be of exactly 3 seconds"

    ],
    markdown=False,
    debug_mode=True,
    show_tool_calls=True,
)

def format_broll_suggestion(transcript):
    response = broll_agent.run(
        f"""Analyze this podcast transcript and suggest B-roll transitions:
        
        {transcript}
        
        Output JSON format:
        {{
            "broll_suggestions": [
                {{
                    "prompt": "visual description",
                    "timestamp_start": 00.00,
                    "timestamp_end": 00.00,
                    "link":"url",
                }}
            ]
        }}"""
    )
    
    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        return {"error": "Failed to parse B-roll suggestions"}

# Example usage with sample transcript
sample_transcript = """
Welcome to the regulating AI podcast where we bring diverse global voices to discuss creating fair and equitable AI regulation. 2.72,10.98
Today, we have the privilege of hosting congressman Ted Lieu who represents California's 30 Sixth Congressional District. 11.5199995,18.585
As the cochair of the House's bipartisan task force on artificial intelligence, congressman Lieu has been at the forefront of efforts to create fair and equitable AI regulation. 18.885,29.545
We are thrilled to have him here to share his insights and perspectives on this crucial topic. 30.21,35.67
Congressman Liu, welcome to the regulating AI podcast. 36.05,39.269997

Thank you, Sanjay. 40.449997,41.43
Honored to be on the podcast. 41.489998,42.87
Wonderful. 43.704998,44.105
Congressman Liu, just to give you some perspective, we have a global audience. 44.585,48.265
We get over a million views. 48.265,49.964996

It's made up of policymakers, think tanks, CEOs of companies, AI aficionados. 50.585,56.125
So, they're all anxiously waiting to hear from you, and several have sent questions in anticipation knowing the great leader you are. 56.425,64.82
So, congressman, to begin with, as the cochair of the house's bipartisan task force on artificial intelligence, What do you consider to be the most pressing challenges in regulating AI technology? 64.82,77.085
Let me first thank speaker Johnson, Leo Jeffries for creating their bipartisan house task force on AI. 78.265,85.565
I'm honored to cochair the task force. 85.86,88.26

My counterpart is congressman Jay Obernotti out of, Southern California, and we have had a number of meetings already. 88.26,96.42
There's 24 members of the task force, 12 Democrats, 12 Republicans. 96.42,100.04
And we're in the process now of gathering information on AI in a number of different, fields and areas. 100.58501,106.685005
And our main mission is to come up with a report by the end of the year of what kinds of AI we might wanna regulate and how we might wanna go about doing so. 106.90501,116.125
Well, congressman, firstly, congratulations, and I think, speaks to the leaders. 117.200005,122.72

And we've been lucky enough to have a lot of your colleagues on the show, including congressman over naughty who was a great guest too. 122.72,129.86
Congressman, you have, you know, put forward several bills in congress. 130.985,136.905
So let's talk a few of them because they are very, very important. 136.905,139.965
Let's first talk about the block nuclear launch by autonomous artificial intelligence act, and I'm talking about it given, the current state of the world we are in. 140.665,149.98
It aims to prevent AI from making decisions to launch nuclear weapons without human control. 150.68,156.06

Can you explain just the motivation behind this bill and the potential risk that you're trying to mitigate? 156.605,161.825
Sure. 163.245,163.72499
One reason that also there are podcasts in AI now and why everyone is looking at this issue is because of a new change in AI technology that basically hit the world, less than two years ago, and it's these large language models. 164.045,179.68
You can call them transformer models. 179.98,181.5
You can call them neural networks. 181.5,182.8

"""
output_path = "/home/tanmay/Desktop/Ukumi_Tanmay/extras/phi-fal.json"
suggestions = format_broll_suggestion(sample_transcript)
print(json.dumps(suggestions, indent=2))
with open(output_path, 'w') as f:
        json.dump(suggestions, f, indent=2)