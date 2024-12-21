import torch
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool

from datetime import datetime
import asyncio

async def main():
    @tool
    def get_weather(date:str, location:str) -> str:
        """Retrieve weather information for location(e.g. town name, address) for specific date.
        Args:
            date: (string) The date for the weather
            location: (string) the address for the weather. e.g. City, Country
        """
        return "-1 degrees celsius, snowy"
    @tool
    def get_date() -> str:
        """Retrieve the current date. This is realtime date information about what is the time now.
        """
        return datetime.today().strftime('%Y-%m-%d')
    tools = [get_weather,get_date]
    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Make sure to use the tools to help answer the human's questions. Make sure to chain the tools when necessary.",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
    )
    agent = create_tool_calling_agent(ChatOllama(model="llama3-groq-tool-use:8b", temperature=0), tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    while True:
        agent_executor.invoke({"input": input("User: ")})

if __name__ == "__main__":
    asyncio.run(main())