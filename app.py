import streamlit as st
import tempfile
import os
import urllib.request
import json
import ssl
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import YoutubeLoader, WebBaseLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# --- Configuration & Setup ---
st.set_page_config(page_title="Minimal AI Knowledge Base", page_icon="🧠", layout="wide")

def inject_minimal_css():
    st.markdown("""
    <style>
        /* Cinematic Obsidian Dark Theme & Micro-Animations */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
        
        .stApp {
            font-family: 'Inter', sans-serif;
            background-color: #0A0A0A;
        }
        
        /* Keyframe Animations */
        @keyframes slideUpFade {
            from { opacity: 0; transform: translateY(15px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* Sidebar layout */
        [data-testid="stSidebar"] {
            border-right: 1px solid #333333;
            background-color: #121212;
        }
        
        /* Top-tier glowing buttons */
        .stButton>button {
            border: 1px solid #00FFFF40;
            border-radius: 8px;
            font-weight: 500;
            background-color: #121212;
            color: #00FFFF;
            transition: all 0.3s ease-in-out;
        }
        .stButton>button:hover {
            border-color: #00FFFF;
            color: #00FFFF;
            transform: scale(1.03);
            box-shadow: 0 0 15px #00FFFF40;
        }
        .stButton>button:active {
            transform: scale(0.98);
        }
        
        /* Glassmorphic inputs */
        .stTextInput>div>div>input, .stFileUploader>div>div, .stSelectbox>div>div>div {
            border-radius: 8px;
            border: 1px solid #333333;
            background-color: #1E1E1E !important;
            color: #E0E0E0;
            transition: all 0.3s ease;
        }
        .stTextInput>div>div>input:focus {
            border-color: #00FFFF;
            box-shadow: 0 0 10px #00FFFF33;
        }
        
        /* Divider */
        hr {
            margin: 2em 0;
            border-bottom: 1px solid #333333;
            border-top: none;
        }
        
        /* Chat messages with staggered entrance */
        .stChatMessage {
            background-color: transparent;
            padding: 1.2rem 1rem;
            border-bottom: 1px solid #1E1E1E;
            animation: slideUpFade 0.5s ease-out forwards;
        }
        /* Assistant message styling */
        [data-testid="chatAvatarIcon-assistant"] {
            background-color: #00FFFF20;
            border: 1px solid #00FFFF;
            color: #00FFFF;
        }
        /* User message styling */
        [data-testid="chatAvatarIcon-user"] {
            background-color: #FF00FF20;
            border: 1px solid #FF00FF;
            color: #FF00FF;
        }
        
        /* Main chat input */
        .stChatInputContainer {
            border-top: 1px solid #333333;
            background-color: #0A0A0A;
            transition: all 0.3s ease;
        }
        .stChatInputContainer:focus-within {
            border-color: #00FFFF;
            box-shadow: 0 0 15px #00FFFF33;
            transform: translateY(-2px);
        }
        
        /* Typography */
        h1, h2, h3 {
            font-weight: 600 !important;
            letter-spacing: -0.02em;
            color: #E0E0E0;
        }
        p, li {
            color: #B0B0B0;
            line-height: 1.6;
        }
    </style>
    """, unsafe_allow_html=True)

inject_minimal_css()
st.title("Minimal AI Knowledge Base (Powered by DeepSeek)")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Helper Functions ---
def get_local_ollama_models():
    """Fetches installed local models from Ollama."""
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=2) as response:
            data = json.loads(response.read().decode())
            # exclude embedding models and vision models if possible
            models = [m["name"] for m in data.get("models", []) if "embed" not in m["name"].lower()]
            return models if models else ["gemma:4b", "deepseek-r1:latest"]
    except Exception:
        return ["gemma:4b", "deepseek-r1:latest"]

def run_perplexity_research(topic):
    """Hits the Perplexity Sonar API and returns a LangChain Document with facts."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {st.secrets.get('PPLX_API_KEY', 'your_perplexity_api_key_here')}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "sonar-pro",
        "messages": [
            {"role": "system", "content": "You are a professional research assistant. Provide a comprehensive, factual overview of the following topic based on web search results."},
            {"role": "user", "content": f"Research the following topic and provide a detailed summary: {topic}"}
        ]
    }
    req = urllib.request.Request(url, headers=headers, data=json.dumps(data).encode("utf-8"))
    with urllib.request.urlopen(req, context=ctx, timeout=60) as response:
        result = json.loads(response.read().decode())
        content = result["choices"][0]["message"]["content"]
        # Wrap the API response string into a Document for our RAG pipeline
        return [Document(page_content=content, metadata={"source": f"Perplexity Research: {topic}"})]

# Initialize settings
available_models = get_local_ollama_models()
if "selected_model" not in st.session_state:
    st.session_state.selected_model = available_models[0] if available_models else "deepseek-r1:latest"

# Setup DB models
EMBEDDING_MODEL = "nomic-embed-text"
DB_DIR = "./chroma_db"

def get_vectorstore():
    """Initializes or loads the local Chroma vector database."""
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

def process_and_store_data(docs):
    """Chunks documents and stores them in the vector database."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    vectorstore = get_vectorstore()
    vectorstore.add_documents(documents=splits)
    return len(splits)

# --- Sidebar: Knowledge Ingestion Zone ---
with st.sidebar:
    st.header("⚙️ Settings")
    # Determine safe index for selectbox
    try:
        default_index = available_models.index(st.session_state.selected_model)
    except ValueError:
        default_index = 0
        
    selected_ml = st.selectbox("Active AI Model", available_models, index=default_index)
    st.session_state.selected_model = selected_ml
    
    st.divider()

    st.header("🔍 Perplexity Auto-Research")
    st.write("Let AI browse the web for you.")
    topic_input = st.text_input("Enter a research topic:")
    if st.button("Research & Absorb"):
        if topic_input:
            with st.spinner("Researching the web..."):
                try:
                    docs = run_perplexity_research(topic_input)
                    chunks_added = process_and_store_data(docs)
                    st.success(f"Absorbed {chunks_added} chunks of research! ✅")
                except Exception as e:
                    st.error(f"Failed to research: {str(e)}")
        else:
            st.warning("Please enter a topic.")
            
    st.divider()

    st.header("📚 Add Knowledge")
    st.write("Feed your AI local data.")
    
    # 1. Web & YouTube Link Ingestion
    link_input = st.text_input("Paste YouTube URL or Website Link:")
    if st.button("Absorb Link"):
        if link_input:
            with st.spinner("Extracting web data..."):
                try:
                    if "youtube.com" in link_input or "youtu.be" in link_input:
                        loader = YoutubeLoader.from_youtube_url(link_input, add_video_info=False)
                    else:
                        loader = WebBaseLoader(link_input)
                    
                    docs = loader.load()
                    chunks_added = process_and_store_data(docs)
                    st.success(f"Absorbed {chunks_added} chunks of data! ✅")
                except Exception as e:
                    st.error(f"Failed to extract: {str(e)}")
        else:
            st.warning("Please enter a valid link.")

    st.divider()

    # 2. PDF Ingestion
    uploaded_file = st.file_uploader("Upload a PDF Document:", type=["pdf"])
    if st.button("Absorb PDF"):
        if uploaded_file is not None:
            with st.spinner("Processing PDF..."):
                try:
                    # Save uploaded file temporarily to read it
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    loader = PyPDFLoader(tmp_file_path)
                    docs = loader.load()
                    chunks_added = process_and_store_data(docs)
                    
                    os.remove(tmp_file_path) # Clean up temp file
                    st.success(f"Absorbed {chunks_added} chunks from PDF! ✅")
                except Exception as e:
                    st.error(f"Failed to process PDF: {str(e)}")
        else:
            st.warning("Please upload a file first.")

# --- Main Window: Chat Interface ---
# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask your custom knowledge base a question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate AI Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # 1. Setup Models & DB
                llm = OllamaLLM(model=st.session_state.selected_model)
                vectorstore = get_vectorstore()
                retriever = vectorstore.as_retriever(search_kwargs={"k": 4}) # Get top 4 results
                
                # 2. Define strict RAG Prompt
                system_prompt = (
                    "You are a helpful assistant. Use the following pieces of retrieved context to "
                    "answer the user's question. If you don't know the answer based on the context, "
                    "just say that you don't know. Keep the answer concise and strictly factual.\n\n"
                    "Context: {context}"
                )
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{input}"),
                ])
                
                # 3. Create Chain & Run
                question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)
                
                response = rag_chain.invoke({"input": prompt})
                answer = response["answer"]
                
                st.markdown(answer)
                # Save assistant response to history
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"Oops! Something went wrong: {str(e)}")
                st.info("Make sure the Ollama app is running in the background and the models are downloaded!")
