from dotenv import load_dotenv
load_dotenv()

from rich import print
import os
import requests

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from tavily import TavilyClient

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

tools = {
    "get_weather": get_weather,
    "get_news": get_news
}

llm_with_tools = llm.bind_tools([get_weather, get_news])

messages = []

print("City Intelligence Agent")
print("Ask me about the weather and news of any city!")
print("Type 'exit' to quit.")

while True:
    user_input = input("\nYou: ").strip()

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    messages.append(HumanMessage(content=user_input))

    while True:
        result = llm_with_tools.invoke(messages)
        messages.append(result)

        if result.tool_calls:
            for tool_call in result.tool_calls:
                tool_name = tool_call["name"]

                confirm = input(f"Agent wants to call {tool_name}. Approve (yes/no): ").strip().lower()

                if confirm == "no":
                    messages.append(
                        ToolMessage(
                            content="Tool call denied by user.",
                            tool_call_id=tool_call["id"]
                        )
                    )
                    continue

                tool_res = tools[tool_name].invoke(tool_call["args"])

                messages.append(
                    ToolMessage(
                        content=str(tool_res),
                        tool_call_id=tool_call["id"]
                    )
                )

            continue

        else:
            print(result.content)
            break