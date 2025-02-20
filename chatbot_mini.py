import getpass
import os
from langchain_openai import ChatOpenAI #ChatOpenAI is a chat model, an instance of a class for interacting with an LLM
from langchain_core.messages import HumanMessage, SystemMessage

model = ChatOpenAI(model="gpt-4o-mini") #instantiate the ChatOpenAI model object

#System message is sent to the model without the user having to type it
#human message is like what the human would type
#Then if you invoke the model and feed it these two messages it'll respond.
messages = [
    SystemMessage("Translate the following from English into Italian"),
    HumanMessage("hi!")
]

model.invoke(messages)
model.invoke("hi")