from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

import getpass
import os

# Model
model = ChatOllama(
    # Modelo descargado desde Ollama (puedes cambiar el nombre según el modelo disponible)
    model="llama3.2",
    base_url="http://localhost:11434"  # URL local donde corre el servidor de Ollama
)

'''
END 2 END agent

memory = MemorySaver()
search = TavilySearchResults(max_results=2)
tools = [search]
agent_executor = create_react_agent(model, tools, checkpointer=memory)


config = {"configurable": {"thread_id": "abc123"}}
for step in agent_executor.stream(
    {"messages": [HumanMessage(content="hi im bob! and i live in sf")]},
    config,
    stream_mode="values",
):
    step["messages"][-1].pretty_print()


for step in agent_executor.stream(
    {"messages": [HumanMessage(content="whats the weather where I live?")]},
    config,
    stream_mode="values",
):
    step["messages"][-1].pretty_print()
'''

print("TAVILY_API_KEY:", os.getenv("TAVILY_API_KEY"))
os.environ["TAVILY_API_KEY"] = "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"

# Defining the search engine with max results of 2
search = TavilySearchResults(max_results=2)
# invoking the search engine with a question
search_results = search.invoke("what is the weather in SF")
print("\nResultado del motor de búsqueda\n")
# Pinrting the results
print(search_results)
# If we want, we can create other tools.
# Once we have all the tools we want, we can put them in a list that we will reference later.
tools = [search]


# Using the model
response = model.invoke([HumanMessage(content="hi!")])
print(response.content)

# Use the bind_tools to assing the tools to the model
model_with_tools = model.bind_tools(tools)

# Using a prompt template to gave a role to the agent
system_message = SystemMessage(content= "You're an excelent assistant expert in tool using. You only have to use tools if the user is questioning something you don't know.")

# Using the model with tools to see the response and the tool calls
print("\nResultado del modelo con herramientas a un mensaje de Hola\n")
response = model_with_tools.invoke([system_message,HumanMessage(content="Hi!")])

print(f"ContentString: {response.content}")
print(f"ToolCalls: {response.tool_calls}")

# Using the model with tools with a proper question
print("\nResultado del modelo con herramientas a una pregunta\n")
response = model_with_tools.invoke([system_message,HumanMessage(content="What's the weather in SF?")])

print(f"ContentString: {response.content}")
print(f"ToolCalls: {response.tool_calls}")


# Create the agent
print("\nCreando el agente\n")
# Note that we are passing in the model, not model_with_tools. That is because create_react_agent will call .bind_tools for us under the hood. Model|Tool|Model
agent_executor = create_react_agent(model, tools)

# Running the agent
response = agent_executor.invoke({"messages":[system_message,HumanMessage(content="hi!")]})

print(response["messages"])

#Streaming Messages
print("\nStreaming Messages\n")
for step in agent_executor.stream(
    {"messages": [system_message,HumanMessage(content="whats the weather in sf?")]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()


# Streaming Tokens
print("\nStreaming Tokens\n")
for step, metadata in agent_executor.stream(
    {"messages": [system_message, HumanMessage(content="whats the weather in sf?")]},
    stream_mode="messages",
):
    if metadata["langgraph_node"] == "agent" and (text := step.text()):
        print(text, end="|")

# Adding memory
print("\nAñadiendo memoria\n")
memory = MemorySaver()

agent_executor = create_react_agent(model, tools, checkpointer=memory)

config = {"configurable": {"thread_id": "abc123"}}
print(f"Tread id: {config["configurable"]["thread_id"]}")

for chunk in agent_executor.stream(
    {"messages": [system_message,HumanMessage(content="hi im bob!")]}, config
):
    print(chunk)
    print("----")


config = {"configurable": {"thread_id": "xyz123"}}
print(f"Tread id: {config["configurable"]["thread_id"]}")

for chunk in agent_executor.stream(
    {"messages": [system_message,HumanMessage(content="whats my name?")]}, config
):
    print(chunk)
    print("----")