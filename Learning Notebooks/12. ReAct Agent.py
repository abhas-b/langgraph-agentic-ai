from typing import TypedDict, List, Union, Annotated, Sequence
# Annotated allows us to add context to a datatype without changing the data type
# Sequence: Used to automatically handle the state updates for sequences such as by adding new messages to a chat history
from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage # Foundation class for all messages in Langchain
from langchain_core.messages import ToolMessage # passes data from tool to LLM
from langchain_core.messages import SystemMessage # Message for providing instruction to LLM
from langchain_core.tools import tool


from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
# add_messages is a reducer function.
# a reducer function basically controls how updates from nodes are combined with existing state.
# prevents us from overwriting the state entirely.
from langgraph.graph.message import add_messages


load_dotenv(override=True)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def add(a: int, b: int):
    """This is an addition function that adds 2 numbers together"""
    return a + b


@tool
def multiply(a: int, b: int):
    """This is an multiplication function that multiplies 2 numbers together"""
    return a * b


tools = [add, multiply]
model = ChatOpenAI(model='gpt-4o-mini').bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content="You are my AI assistant. Please answer my question to the best of your ability"
        )
    print(f"In Model Call function: {state['messages']} \n\n")
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]} # This is a result of using add_messages


def should_continue(state: AgentState) -> AgentState:
    messages = state["messages"]
    last_message = messages[-1]

    print(f"Messages from should continue: {messages}")
    print(f"Last message: {last_message}")

    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
    

graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "end": END,
        "continue":"tools"
    }
)


graph.add_edge("tools", "our_agent")


app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [("user", "Add 40 + 12.")]}
app.invoke(inputs)
# inputs = {"messages": [("user", "Add 40 + 12 and then multiply the result by 6. Also tell me a joke please.")]}
# print_stream(app.stream(inputs, stream_mode="values"))