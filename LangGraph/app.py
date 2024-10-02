import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from langchain.agents import create_structured_chat_agent, AgentExecutor, tool
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
import os
from langchain_chroma import Chroma
from langchain_community.document_loaders import PDFMinerLoader , CSVLoader
import tempfile
from chromadb.utils.embedding_functions import chroma_langchain_embedding_function
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch


# Load environment variables
load_dotenv()
# Get API keys from environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
# Configure the Google API key
genai.configure(api_key=GOOGLE_API_KEY)
# Define chat prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "Respond to the human as helpfully and accurately as possible. You have access to the following tools:\n\n{tools}\n\n and one of them uses vectorstore to searh the document.Use a JSON blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).\n\nValid 'action' values: 'Final Answer' or {tool_names}\n\nProvide only ONE action per $JSON_BLOB, as shown:\n\n```\n{{\n  \"action\": $TOOL_NAME,\n  \"action_input\": $INPUT\n}}\n```\n\nFollow this format:\n\nQuestion: input question to answer\nThought: consider previous and subsequent steps\nAction:\n```\n$JSON_BLOB\n```\nObservation: action result\n... (repeat Thought/Action/Observation N times)\nThought: I know what to respond\nAction:\n```\n{{\n  \"action\": \"Final Answer\",\n  \"action_input\": \"Final response to human\"\n}}\n\nBegin! Reminder to ALWAYS respond with a valid JSON blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Action:\n```$JSON_BLOB```\nthen Observation"),
    ("placeholder", "{chat_history}"),
    ("human", "{input}\n\n{agent_scratchpad}\n(reminder to respond in a JSON blob no matter what)")
])
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma("Documents",embedding_function=embeddings,persist_directory="./document_db")
# Define a custom Tavily Search tool using @tool decorator
@tool
def tavily_search(query):
    """Fetches search results from Tavily based on the query input."""
    from langchain_community.tools.tavily_search import TavilySearchResults
    tavily_tool = TavilySearchResults(api_key=TAVILY_API_KEY, max_results=2)
    return tavily_tool.invoke(query["title"])
@tool
def vectorstore_search(query):
    """Fetches search result from vectorstore based on query input."""
    retriver = vectorstore._similarity_search_with_relevance_scores(query=query['title'])
    return retriver
# Initialize embeddings and LLM
llm = ChatGoogleGenerativeAI(
    api_key=GOOGLE_API_KEY,
    model="gemini-1.5-flash",
    max_retries=5
)
# Define tools list using the @tool decorated function directly
tools = [tavily_search,vectorstore_search]
# Create the tool-calling agent using the direct @tool integration
agent = create_structured_chat_agent(llm, tools, prompt)
# Create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=5)
# Function to handle search queries using the agent executor
def search_the_web_or_vector_store(query):
    # Retrieve the chat history from session state
    chat_history = st.session_state.chat_history
    # Prepare the chat history for the agent executor
    agent_chat_history = []
    for entry in chat_history:
        if 'human' in entry:
            agent_chat_history.append(HumanMessage(content=entry['human']))
        elif 'ai' in entry:
            agent_chat_history.append(AIMessage(content=entry['ai']))
    # Call the agent executor with the current query and the chat history
    result = agent_executor.invoke({
        "input": query,
        "chat_history": agent_chat_history  
        # Pass the chat history to the agent
    })
    return result['output']
# Function to handle submit action when the user submits the form
def handle_submit(user_input):
    if user_input:
        # Add human message to the session state
        st.session_state.chat_history.append({'human': user_input})
        # Get the bot's response using search_the_web
        bot_output = search_the_web_or_vector_store(user_input)
        # Add AI response to the session state
        st.session_state.chat_history.append({'ai': bot_output})
def pdf_loader(files):
    loader = PDFMinerLoader(files)
    docs = loader.load()
    return docs[0]
def csv_loader(files):
    loader = CSVLoader(files)
    docs = loader.load()
    return docs[0]

# Set up Streamlit app configuration
st.set_page_config(page_title="Agent with History", layout="wide")
# Streamlit UI Title
st.title("Chatbot with Agent History")
# Initialize chat history and document in session state if not present
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
# Function to identify file type and save the file in session state
def save_file_by_type(files):
    for file in files:
        if file.type == "application/pdf":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file.getvalue())  # Write the uploaded file content to temp file
                tmp_file_path = tmp_file.name  # Get the file path
                # Load PDF using the PDF loader with the temp file path
                pdf_data = pdf_loader(tmp_file_path)
                # Add text content and metadata
                vectorstore.add_documents([pdf_data])
        elif file.type in ["image/jpeg", "image/png"]:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg" if file.type == "image/jpeg" else ".png") as tmp_file:
                tmp_file.write(file.getvalue())
                tmp_file_path = tmp_file.name
                # Handle image file (you can perform additional processing here)
                # Adjusted to directly pass the embeddings object without chroma_langchain_embedding_function
                vectorstore.add_images([tmp_file_path])
        elif file.type == "text/csv":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
                tmp_file.write(file.getvalue())
                tmp_file_path = tmp_file.name
                # Load CSV using the CSV loader with the temp file path
                csv_data = csv_loader(tmp_file_path)
                # Add csv data and metadata
                vectorstore.add_documents([csv_data])
        else:
            st.warning("Unsupported file format.")
# File uploader widget in the sidebar
files = st.sidebar.file_uploader("Upload file", type=['pdf', 'jpg', 'jpeg', 'csv', 'png'], accept_multiple_files=True)
# If files are uploaded, save them based on their type
if files:
    save_file_by_type(files)
    st.success("Files uploaded and categorized successfully!")
# Function to display chat history
def display_chat():
    for message in st.session_state.chat_history:
        if 'human' in message:
            st.write(f"**You:** {message['human']}")
        elif 'ai' in message:
            st.write(f"**Bot:** {message['ai']}")
# Define a form for the user input and submit button
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Type your message:")
    submit_button = st.form_submit_button(label="Send")
    # Handle submit action when the user clicks "Send"
    if submit_button:
        handle_submit(user_input)
        display_chat()                            
    # Display chat after submission
# Clear Chat button to reset the session state chat history
if st.button("Clear Chat"):
    st.session_state.chat_history.clear()