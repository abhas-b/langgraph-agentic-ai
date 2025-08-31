from typing import TypedDict, List, Union
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os


load_dotenv(override=True)

class AgentState(TypedDict):
    messages : List[Union[HumanMessage, AIMessage]]


llm = ChatOpenAI(model='gpt-4o-mini')

def process_node(state: AgentState) -> AgentState:
    """This node will solve the request as per input"""
    response = llm.invoke(state['messages'])

    state["messages"].append(AIMessage(content=response.content))
    print(f"\nAI: {response.content}")
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process_node)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

conversation_history = []
user_input = input("Enter: ")
while user_input.lower() != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": conversation_history})

    print(result['messages'])
    conversation_history = result['messages']
    user_input = input("Enter: ")


with open("logging_basic_chatbot.txt", "w") as file:
    file.write("Your Conversation Log: \n")
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"You: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n")
    file.write("End of Conversation")

print("Conversation saved to logs")