from dotenv import load_dotenv
load_dotenv()

import os
import requests
from rich import print

from tavily import TavilyClient

from langchain.tools import tool
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda

API_KEY = os.getenv("OPENWEATHER_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not API_KEY:
    raise ValueError("OPENWEATHER_API_KEY is missing in environment variables")

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY is missing in environment variables")


@tool
def get_weather(city: str) -> str:
    """Get current weather of a city."""
    url = (
        "https://api.openweathermap.org/data/2.5/weather"
        f"?q={city}&appid={API_KEY}&units=metric"
    )

    try:
        response = requests.get(url, timeout=10)
        data = response.json()
    except requests.RequestException as e:
        return f"Error while fetching weather: {e}"

    if response.status_code != 200:
        return f"Error: {data.get('message', 'Could not fetch weather')}"

    city_name = data["name"]
    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]

    return f"Weather in {city_name}: {desc}, {temp}°C"


tavily_client = TavilyClient(api_key=TAVILY_API_KEY)


@tool
def get_news(city: str) -> str:
    """Get latest news about the city."""
    try:
        response = tavily_client.search(
            query=f"latest news about {city}",
            max_results=3,
            search_depth="basic",
        )
    except Exception as e:
        return f"Error while fetching news: {e}"

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


tools = [get_weather, get_news]
tool_map = {tool.name: tool for tool in tools}

llm = ChatMistralAI(
    model="mistral-small-2506",
    temperature=0,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a good news reporter."),
    MessagesPlaceholder("messages"),
])

# Runnable: prompt -> model with tools
model_runnable = prompt | llm.bind_tools(tools)


def execute_tool_calls(ai_message):
    """Run every tool requested by the model and convert results to ToolMessages."""
    tool_messages = []

    for tool_call in ai_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]

        selected_tool = tool_map[tool_name]

        try:
            result = selected_tool.invoke(tool_args)
        except Exception as e:
            result = f"Tool execution error in '{tool_name}': {e}"

        tool_messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_id,
            )
        )

    return tool_messages


tool_runner = RunnableLambda(execute_tool_calls)


def run_agent(user_text: str) -> str:
    """Runnable-style agent loop."""
    messages = [HumanMessage(content=user_text)]

    while True:
        ai_message = model_runnable.invoke({"messages": messages})
        messages.append(ai_message)

        if not ai_message.tool_calls:
            return ai_message.content

        tool_messages = tool_runner.invoke(ai_message)
        messages.extend(tool_messages)


print("City Agent | type exit to quit\n")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() == "exit":
        print("Good Bye!")
        break

    try:
        answer = run_agent(user_input)
        print("BOT:", answer)
    except Exception as e:
        print(f"[red]Error:[/red] {e}")