from langchain_ollama import ChatOllama 
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# Model
model = ChatOllama(
    model="llama3.2",                  # Modelo descargado desde Ollama (puedes cambiar el nombre según el modelo disponible)
    base_url="http://localhost:11434"  # URL local donde corre el servidor de Ollama
)


# Using LLM
messages = [
    SystemMessage("Translate the following from English into Italian"),
    HumanMessage("hi!"),
]

# Using dict of messages
output = model.invoke(messages)
print(output)

# Using direct string
output = model.invoke("Hello")
print(output)

# Using dict with role and content
output = model.invoke([{"role": "user", "content": "Hello"}])
print(output)

# Using Humanmessage with the prompt
output = model.invoke([HumanMessage("Hello")])
print(output)


#Streaming
for token in model.stream(messages):
    print(token.content, end="|")


# Prompt template using direct prompt
'''
String PromptTemplates
These prompt templates are used to format a single string, and generally are used for simpler inputs. 
For example, a common way to construct and use a PromptTemplate is as follows:
'''
prompt_template = PromptTemplate.from_template("Tell me a joke about {topic}")

# Assing topic value with invoke
prompt = prompt_template.invoke({"topic": "cats"})
print(prompt)

# Prompt template with dict
'''
ChatPromptTemplates
These prompt templates are used to format a list of messages. 
These "templates" consist of a list of templates themselves. 
For example, a common way to construct and use a ChatPromptTemplate is as follows:
'''
prompt_template = ChatPromptTemplate([
    ("system", "You are a helpful assistant"),
    ("user", "Tell me a joke about {topic}")
])

prompt = prompt_template.invoke({"topic": "cats"})
print(prompt)

# Message PlaceHolder
'''
MessagesPlaceholder

This prompt template is responsible for adding a list of messages in a particular place. 
In the above ChatPromptTemplate, we saw how we could format two messages, each one a string. 
But what if we wanted the user to pass in a list of messages that we would slot into a particular spot? 
This is how you use MessagesPlaceholder.
'''
prompt_template = ChatPromptTemplate([
    ("system", "You are a helpful assistant"),
    MessagesPlaceholder("msgs")
])

prompt = prompt_template.invoke({"msgs": [HumanMessage(content="hi!")]})
print(prompt)

# Other way to use Message PlaceHolder
'''
This will produce a list of two messages, the first one being a system message, and the second one being the HumanMessage we passed in. 
If we had passed in 5 messages, then it would have produced 6 messages in total (the system message plus the 5 passed in). 
This is useful for letting a list of messages be slotted into a particular spot.

An alternative way to accomplish the same thing without using the MessagesPlaceholder class explicitly is:
'''
prompt_template = ChatPromptTemplate([
    ("system", "You are a helpful assistant"),
    ("placeholder", "{msgs}") # <-- This is the changed part
])

prompt = prompt_template.invoke({"msgs": [HumanMessage(content="hi!")]})
print(prompt)

# Prompt Template
system_template = "Translate the following from English into {language}"

# Prompt template with dict of messages using from_messages method
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

prompt = prompt_template.invoke({"language": "Italian", "text": "hi!"})
print(prompt)

# We can see that it returns a ChatPromptValue that consists of two messages. If we want to access the messages directly we do:
print(prompt.to_messages())

# invoke the chat model on the formatted prompt:
response = model.invoke(prompt)
print(response.content)
