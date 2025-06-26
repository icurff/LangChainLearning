import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

search = TavilySearch(max_results=2)
tools = [search]


model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.environ["GOOGLE_API_KEY"]
)
memory = MemorySaver()

# model_with_tools = model.bind_tools(tools)
#
# response = model_with_tools.invoke([{"role": "user", "content":  "Search for the weather in Danang, Vietnam"}])
#
# print(f"Message content: {response.text()}\n")
# print(f"Tool calls: {response.tool_calls}")


agent_executor = create_react_agent(model, tools, checkpointer=memory)
config = {"configurable": {"thread_id": "abc123"}}
input_message = {
    "role": "user",
    "content": "Hi, I'm Hải and I life in Đà Nẵng, Vietnam.",
}
for step in agent_executor.stream(
    {"messages": [input_message]}, config, stream_mode="values"
):
    step["messages"][-1].pretty_print()

input_message = {
    "role": "user",
    "content": "What's the weather where I live?",
}

for step in agent_executor.stream(
    {"messages": [input_message]}, config, stream_mode="values"
):
    step["messages"][-1].pretty_print()
