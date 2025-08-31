from typing import TypedDict, List, Union, Annotated, Sequence
from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, HumanMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages


load_dotenv(override=True)

document_content = ""

class AgentState(TypedDict):
    messages = Annotated[Sequence(BaseMessage), add_messages]

@tool
def update_tool(content: str) -> str:
    """Updates the document with the required content"""
    global document_content
    document_content = content
    return f"Document has been updated successfully! The current content is \n{document_content}"


@tool
def save_tool(filename: str) -> str:
    """Saves the current document to a text file and finish the process
    
    Args:
        filename for the text file
    """

    if not filename.endswith('.txt'):
        filename = f"{filename}.txt"

    try:
        with open(filename, "w") as file:
            file.write(document_content)
        print("\n Document saved as {filename}")
        return "Document has been saved successfully to {filename}"
    except Exception as e:
        return f"Error in saving document: {str(e)}"
    


tools = [update_tool, save_tool]
model = ChatOpenAI(model='gpt-4o-mini').bind_tools(tools)

def agent_process(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content=f'''
                    You are a drafter, a helpful writing assistant. You are going to help the user update and modify documents.

                    - If the user wants to update the document, use the update tool.
                    - If the user wants to save the document, use the save tool.
                    - Make sure to always show the current document state after modifications.

                    The current document content is: {document_content}''')

    if not state["messages"]:
        user_input = "Im ready to help you update documents. What would you like to create"
        user_message = HumanMessage(content=user_input)
    else:
        user_input = input("\nWhat would you like to do with the document?")
        print(f"\n USER: {user_input}")
        user_message = HumanMessage(content=user_input)

    all_messages = system_prompt + list(state["messages"]) + [user_message]

    response = model.invoke(all_messages)
    print(f"AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"Using tools: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": list(state["messages"]) + [user_message, response]}


def should_continue(state: AgentState) -> AgentState:
    messages = state["messages"]

    if not messages:
        return "continue"
    
    for message in reversed(messages):
        if (isinstance(message, ToolMessage)) and ("saved" in message.content.lower()) and ("document" in message.content.lower()):
            return "end"
        
    return "continue"


def print_messages(messages):
    if not messages:
        return
    
    for message in messages[-3]:
        if isinstance(message, ToolMessage):
            print(f"Tool result: {message.content}")   



graph = StateGraph(AgentState)
graph.add_node("agent", agent_process)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.add_edge("agent", "tools")

graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "end": END,
        "continue":"agent"
    }
)

graph.set_entry_point("agent")
app = graph.compile()

def run_document_agent():
    print(f"\n===========DRAFTER============")
    state = {"messages": []}

    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])

    print("\n =========== DRAFTER FINISHED============")

if __name__ == "__main__":
    run_document_agent()