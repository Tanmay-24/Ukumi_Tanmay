from browser_use import Agent, Browser, BrowserConfig
from langchain_openai import ChatOpenAI
import asyncio
from dotenv import load_dotenv
load_dotenv()

# Configure the browser to connect to your Chrome instance
browser = Browser(
    config=BrowserConfig(
        chrome_instance_path="/usr/bin/brave-browser",
        _force_keep_brower_alive=True
    )
)

# Create the agent with your configured browser
agent = Agent(
    task="go to chatgpt and ask it to write a python script that uses the langchain library to create a simple agent that can browse the web",
    llm=ChatOpenAI(model='gpt-4o'),
    browser=browser,
)

async def main():
    await agent.run()

    input('Press Enter to close the browser...')
    await browser.close()

if __name__ == '__main__':
    asyncio.run(main())