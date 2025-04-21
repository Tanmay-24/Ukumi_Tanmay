from browser_use import Agent, Browser, BrowserConfig, BrowserContextConfig
from langchain_openai import ChatOpenAI
import asyncio
from dotenv import load_dotenv
load_dotenv()
import os
# Configure the browser to connect to your Chrome instance
browser = Browser(
    config=BrowserConfig(
        chrome_instance_path="/usr/bin/google-chrome",
        new_context_config=BrowserContextConfig(save_downloads_path="/home/tanmay/Downloads/")
    )
)
    
async def run_download():
	agent = Agent(
    task="""
Navigate to https://drive.google.com/file/d/1De1uM11p4Dv7aYvlbDCw2tXTw9Oa3TRh/view?usp=drive_link
and download and save the mp4
""",
    llm=ChatOpenAI(model='gpt-4o'),
    browser=browser,
    use_vision=True,
)
	await agent.run(max_steps=25)
	await browser.close()


if __name__ == '__main__':
	asyncio.run(run_download())    