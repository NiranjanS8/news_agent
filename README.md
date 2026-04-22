# 📌 News Agent – LangChain (Agent vs Runnables)

A simple AI agent that can:

- 🌦️ Get current weather of a city  
- 📰 Fetch latest news about a city  

Built while learning **LangChain tool calling** and comparing two approaches:
1. High-level `create_agent()`
2. Low-level **Runnables (manual loop)**

---

## 🚀 Features

- Custom tools using `@tool`
- OpenWeather API integration
- Tavily search for news
- Two implementations:
  - ✅ Agent abstraction
  - ✅ Runnable-based control flow

---

## 🧠 What I Learned

- How LLMs decide when to call tools  
- The internal loop:  
  **User → Model → Tool → Model → Final Answer**
- Difference between:
  - `create_agent()` (easy, abstracted)
  - Runnables (flexible, transparent)

---

## ⚙️ Tech Stack

- Python  
- LangChain  
- Mistral (`ChatMistralAI`)  
- Tavily API  
- OpenWeather API  

---

## 🆚 Approaches

### 1️⃣ Agent (Simple)

```python
from langchain.agents import create_agent

agent = create_agent(
    llm,
    tools=[get_weather, get_news],
    system_prompt="You are a helpful news reporter."
)
```

**Pros:**
- ✔️ Easy to use  
- ✔️ Minimal setup  

**Cons:**
- ❌ Hides internal working  
- ❌ Less control  

---

### 2️⃣ Runnables (Advanced)

```python
model = prompt | llm.bind_tools(tools)
```

**Pros:**
- ✔️ Full control  
- ✔️ Clear understanding of tool-calling loop  

**Cons:**
- ❌ Slightly more code  

---

## 🔁 Flow (Runnable Version)

1. User input  
2. Model decides tool calls  
3. Execute tools  
4. Append results as `ToolMessage`  
5. Repeat until final response  

---

## 🔑 Setup

Install dependencies:

```bash
pip install langchain langchain-mistralai tavily-python python-dotenv requests rich
```

Create a `.env` file:

```
OPENWEATHER_API_KEY=your_key
TAVILY_API_KEY=your_key
```
