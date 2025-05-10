from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage, trim_messages
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from typing import Sequence
from typing_extensions import Annotated, TypedDict
import transformers


# Model
model = ChatOllama(
    # Modelo descargado desde Ollama (puedes cambiar el nombre según el modelo disponible)
    model="llama3.2",
    base_url="http://localhost:11434"  # URL local donde corre el servidor de Ollama
)

# Calling the model
output = model.invoke([HumanMessage(content="Hi! I'm Bob")])
print("Llamando al modelo\n")
print(output)

# Model without state
output = model.invoke([HumanMessage(content="What's my name?")])
print("\nModelo sin estado\n")
print(output)

# Giving the context
output = model.invoke(
    [
        HumanMessage(content="Hi! I'm Bob"),
        AIMessage(content="Hello Bob! How can I assist you today?"),
        HumanMessage(content="What's my name?"),
    ]
)
print("\nPasando el contexto al modelo\n")
print(output)


# Message Persistence
print("\nPersistencia de Mensajes\n")
# Define a new graph
workflow = StateGraph(state_schema=MessagesState)


# Define the function that calls the model
def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": response}


# Define the (single) node in the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Creating the thread
config = {"configurable": {"thread_id": "abc123"}}

# New query
query = "Hi! I'm Bob."

# input uses the new query
input_messages = [HumanMessage(query)]

# invoke the workflow compile using the input and the thread
output = app.invoke({"messages": input_messages}, config)

print(
    f"\nConversación con el thread id: {config['configurable']['thread_id']}\n")

# Printing the last message of the state
output["messages"][-1].pretty_print()  # output contains all messages in state

# Following the thread
query = "What's my name?"

input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()

# Using a new thread
config = {"configurable": {"thread_id": "abc234"}}

input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)

print(
    f"\nConversación con el thread id: {config['configurable']['thread_id']}\n")
output["messages"][-1].pretty_print()

config = {"configurable": {"thread_id": "abc123"}}

input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)

print(
    f"\nConversación con el thread id: {config['configurable']['thread_id']}\n")
output["messages"][-1].pretty_print()


'''
# Async function for node:
async def call_model(state: MessagesState):
    response = await model.ainvoke(state["messages"])
    return {"messages": response}


# Define graph as before:
workflow = StateGraph(state_schema=MessagesState)
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)
app = workflow.compile(checkpointer=MemorySaver())

# Async invocation:
output = await app.ainvoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()
'''


# Prompt Templates
print("\nPrompt Templates\n")
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You talk like a pirate. Answer all questions to the best of your ability.",  # sytem prompt
        ),
        # from state(MessageState) we use the messages to insert as the placeholder, the MessageState automatically will add any other message
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Create the new workflow with the MessagesState schema
builder = StateGraph(state_schema=MessagesState)

# New function addgin the state


def call_model_template(state: MessagesState):
    # the prompt template will invoke the messages from the MessagesState schema
    prompt = prompt_template.invoke(state)
    # the model will invoke the prompt with the full conversation
    response = model.invoke(prompt)
    # updating the messages from MessagesState with the new response
    return {"messages": response}


builder.add_edge(START, "model_template")
builder.add_node("model_template", call_model_template)

memory = MemorySaver()
app = builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "abc345"}}
query = "Hi! I'm Jim."

input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
print(
    f"\nConversación con el thread id: {config['configurable']['thread_id']}\n")
output["messages"][-1].pretty_print()


query = "What is my name?"

input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()


print("\nPrompt Templates complicados\n")
# Complicate Prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Creating the new state class with TypedDict schema: use messages as an Annotaded List with a
# Sequence of BaseMessage using the method add_messages: Merges two lists of messages, updating existing messages by ID.
# Use the variable language to be use it in the prompt template


class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage],
                        add_messages]    # List of messages
    language: str                                               # lenguage variable


# New workflow using the State class as schema
flow = StateGraph(state_schema=State)


def call_model_complicated(state: State):
    prompt = prompt_template.invoke(state)
    response = model.invoke(prompt)
    return {"messages": [response]}


flow.add_edge(START, "model_complicated")
flow.add_node("model_complicated", call_model_complicated)

memory = MemorySaver()
app = flow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "abc456"}}
query = "Hi! I'm Bob."
# Defining the language
language = "Spanish"

input_messages = [HumanMessage(query)]
output = app.invoke(  # invoke the app with the two variables used in the schema and the thread
    {"messages": input_messages,
     "language": language},
    config,
)
print(
    f"\nConversación con el thread id: {config['configurable']['thread_id']}\n")
output["messages"][-1].pretty_print()

query = "What is my name?"

input_messages = [HumanMessage(query)]
output = app.invoke(
    {"messages": input_messages},
    config,
)
output["messages"][-1].pretty_print()


print("\nManejando el historial de la conversación\n")

# Managing Conversation History
# Use trim_messages to reduce how many messages we're sending to the model.
# use transformers
trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=count_tokens_approximately,  # use model if you have transformers
    include_system=True,
    allow_partial=False,
    start_on="human",
)

# List of messages
messages = [
    SystemMessage(content="you're a good assistant"),
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]

print("\nInvocando el trimmer\n")
print(trimmer.invoke(messages))

newworkflow = StateGraph(state_schema=State)


def call_model_trimmed(state: State):
    # Invoke the trimmer with the message of the state
    trimmed_messages = trimmer.invoke(state["messages"])
    prompt = prompt_template.invoke(
        # Passing the trimmed messages as messages in the prompt template
        {"messages": trimmed_messages, "language": state["language"]}
    )
    response = model.invoke(prompt)
    return {"messages": [response]}


newworkflow.add_edge(START, "model_trimmed")
newworkflow.add_node("model_trimmed", call_model_trimmed)

memory = MemorySaver()
app = newworkflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "abc567"}}
query = "What is my name?"
language = "English"

input_messages = messages + [HumanMessage(query)]
output = app.invoke(
    {"messages": input_messages, "language": language},
    config,
)
output["messages"][-1].pretty_print()


config = {"configurable": {"thread_id": "abc678"}}
query = "What math problem did I ask?"
language = "English"

input_messages = messages + [HumanMessage(query)]
output = app.invoke(
    {"messages": input_messages, "language": language},
    config,
)
output["messages"][-1].pretty_print()


# Streaming
print("\nStreaming\n")
config = {"configurable": {"thread_id": "abc789"}}
query = "Hi I'm Todd, please tell me a joke."
language = "English"

input_messages = [HumanMessage(query)]
for chunk, metadata in app.stream(
    {"messages": input_messages, "language": language},
    config,
    stream_mode="messages",
):
    if isinstance(chunk, AIMessage):  # Filter to just model responses
        print(chunk.content, end="|")
