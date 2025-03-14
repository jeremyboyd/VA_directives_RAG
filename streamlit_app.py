# Author: Jeremy Boyd (jeremy.boyd@va.gov)
# Description: Screipt that extracts all pages of text from PDFs in a directory,
# merges pages into documents, splits documents into chunks, embeds chunks,
# stores chunks in a vector store, implements a RAG-chatbot that uses vector
# store to respond to queries.

# Packages
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing_extensions import List, TypedDict
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain import hub
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.graph import START, StateGraph, MessagesState, END
from typing_extensions import List, TypedDict
from langchain_chroma import Chroma
from langgraph.prebuilt import ToolNode, tools_condition
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
import chromadb

# Load environment variables from .env file
load_dotenv()

# Access env variables
openai_api_key = os.getenv("OPENAI_API_KEY")
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
DB_CONNECTION_STRING = os.getenv("DATABASE_URL")

# Embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Read vector store from postgreSQL db
vector_store = Chroma(persist_directory=DB_CONNECTION_STRING, embedding_function=embeddings)

# Initialize chat model
llm = init_chat_model("gpt-4o-mini", model_provider="openai")

# Graph to represent states of assistant
graph_builder = StateGraph(MessagesState)

# Turn retreival step into tool
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=10)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# Step 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}

# Step 2: Execute the retrieval.
tools = ToolNode([retrieve])

# Step 3: Generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know."
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # Run
    response = llm.invoke(prompt)
    return {"messages": [response]}

# Compile into graph
graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)
graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)
graph = graph_builder.compile()

# Implement chat memory
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "abc123"}}

# Add agent
from langgraph.prebuilt import create_react_agent
agent_executor = create_react_agent(llm, [retrieve], checkpointer=memory)
config = {"configurable": {"thread_id": "def234"}}

# Streamlit UI
st.title("Chat Assistant 💬")
input_message = st.text_input("You:", "")

# Send input to LLM on send
if st.button("Send"):
    if input_message:
        response_chunks = []
        for event in agent_executor.stream(
            {"messages": [{"role": "user", "content": input_message}]},
            stream_mode="values",
            config=config,
        ):
            response_chunks.append(event["messages"][-1])
        st.text_area("Assistant:", str(response_chunks[-1].content), height=200)
