from dotenv import load_dotenv
load_dotenv()

from rich import print
import os
import requests

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from tavily import TavilyClient
from langchain.agents import create_agent

API_KEY = os.getenv("OPENWEATHER_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


@tool
def get_weather(city: str) -> str:
    """Get current weather of a city"""
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

    response = requests.get(url, timeout=10)
    data = response.json()

    if response.status_code != 200:
        return f"Error: {data.get('message', 'Could not fetch weather')}"

    city_name = data["name"]
    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]

    return f"Weather in {city_name}: {desc}, {temp}°C"


tavily_client = TavilyClient(api_key=TAVILY_API_KEY)


@tool
def get_news(city: str) -> str:
    """Get latest news about the city"""
    response = tavily_client.search(
        query=f"latest news about {city}",
        max_results=3,
        search_depth="basic"
    )

    results = response.get("results", [])

    if not results:
        return f"No news found for {city}"

    news_list = []

    for result in results:
        title = result.get("title", "No Title")
        url = result.get("url", "No Link")
        snippet = result.get("content", "")
        news_list.append(f"{title}\n- {url}\n{snippet[:300]}...")

    return f"Latest news about {city}:\n\n" + "\n\n".join(news_list)


llm = ChatMistralAI(model="mistral-small-2506")


agent = create_agent(
    llm,
    tools=[get_weather, get_news],
    system_prompt= "You are a good news reporter." 
)

print("City Agent | type exit to quit")
print("")

while  True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Good Bye!")
        break

    result = agent.invoke({
        "messages": [{"role" : "user", "content": user_input}]
    })

    print("BOT: ",result['messages'][-1].content)