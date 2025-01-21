import streamlit as st
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
import os
import nltk
from nltk.tokenize import sent_tokenize
import tiktoken
from langchain_google_genai import ChatGoogleGenerativeAI
from supabase import create_client, Client
import uuid
from datetime import datetime

nltk.download('punkt', quiet=True)

# Supabase Setup
SUPABASE_URL = "https://isolwubdadbolryjzbnx.supabase.co"  # Replace with your url
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imlzb2x3dWJkYWRib2xyeWp6Ym54Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3Mzc0NjYzNzIsImV4cCI6MjA1MzA0MjM3Mn0.xCeaf2x3vmho7tkot2S7PU0hWszZNEJIHnHPgpvJ3x0"  # Replace with your key
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


st.title("MBBS Chatbot")


def get_tokenizer_for_model(model_name):
    """
    Determine the correct tokenizer function based on model name.
    """
    try:
        encoding = tiktoken.encoding_for_model(model_name)
        return lambda text: encoding.encode(text)
    except KeyError:
        return nltk.word_tokenize


def load_pdf_documents(pdf_folder):
    """Load all pdfs from the specified folder"""
    documents = []
    for filename in os.listdir(pdf_folder):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder, filename)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
    return documents


def split_text_with_sentences(text, tokenizer, chunk_size=500, chunk_overlap=50):
    """Split the text into chunks based on sentences with token and character limits"""
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        tokenized_sentence = tokenizer(sentence)
        if len(tokenizer(current_chunk + sentence)) <= chunk_size:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def preprocess_documents(documents, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Preprocess documents into chunks"""
    tokenizer = get_tokenizer_for_model(model_name)

    split_docs = []
    for doc in documents:
        text = doc.page_content
        chunks = split_text_with_sentences(text, tokenizer)
        for chunk in chunks:
            split_docs.append(chunk)
    return split_docs


def create_vector_store(split_documents, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Create the vectorstore for the given document splits."""
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vector_store = Chroma.from_texts(texts=split_documents, embedding=embeddings)
    return vector_store


def create_llm(gemini_api_key):
    """Create a Gemini LLM object."""
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=gemini_api_key)
    return llm


def create_retrieval_qa_chain(llm, vector_store):
    """Creates retrieval QA chain from the vector store and llm."""
    template = """
        Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        {context}

        Question: {question}
        Helpful Answer:
        """
    QA_PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever(),
                                          chain_type_kwargs={"prompt": QA_PROMPT})
    return qa_chain


def generate_response(chain, query):
    """Generates the response from the query."""
    response = chain.run(query)
    return response


# Authentication Functions
def create_new_user(email, password):
    """Creates a new user in Supabase"""
    try:
        new_user_data = {"id": str(uuid.uuid4()), "email": email, "password": password, "created_at": str(datetime.now())}
        data, count = supabase.table("Users").insert(new_user_data).execute()
        return data, None
    except Exception as e:
        return None, e


def get_user_by_email(email):
    """Retrieves a user by their email."""
    try:
        data, count = supabase.table("Users").select("*").eq("email", email).execute()
        return data[1][0], None
    except Exception as e:
        return None, e


def check_credentials(email, password):
    """Checks if the email and password match in database"""
    user, e = get_user_by_email(email)
    if e is None and user is not None:
        if password == user['password']:
            return user, None
        else:
            return None, "incorrect password"
    else:
        return None, e


def process_user_data(user_id):
    """Process user specific files if not already processed."""
    if user_id not in st.session_state:
        st.session_state[user_id] = {"vector_store": None, "qa_chain": None, "chat_history": [], "pdf_paths": []}
    if "pdf_folder" not in st.session_state[user_id]:
        st.session_state[user_id]["pdf_folder"] = "pdf_folder"
    if st.session_state[user_id]["vector_store"] is None:
        with st.spinner("Processing your textbooks..."):
            pdf_folder = st.session_state[user_id]["pdf_folder"]
            documents = load_pdf_documents(pdf_folder)
            split_documents = preprocess_documents(documents)
            st.session_state[user_id]["vector_store"] = create_vector_store(split_documents)
        st.success("Textbooks processed successfully!")

    if st.session_state[user_id]["qa_chain"] is None:
        with st.spinner("Loading model..."):
            gemini_api_key = "YAIzaSyCyKtp7Q20a0xRjOo_c9xkX5mZyF5y4xY4"  # Replace with your API key
            llm = create_llm(gemini_api_key)
            st.session_state[user_id]["qa_chain"] = create_retrieval_qa_chain(llm,
                                                                             st.session_state[user_id]["vector_store"])
        st.success("LLM Model Loaded successfully!")


# Supabase Storage
def upload_file_to_storage(file, user_id):
    """Uploads a file to supabase storage"""
    try:
        file_path = f"{user_id}/{file.name}"
        supabase.storage.from_("files").upload(file_path, file.getvalue(), file_options={"content-type": file.type})
        return file_path, None
    except Exception as e:
        return None, e


# UI Logic
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.subheader("Login or Register")
    login_email = st.text_input("Email", key="login_email")
    login_password = st.text_input("Password", type="password", key="login_password")
    login_button = st.button("Login")
    if login_button:
        user, e = check_credentials(login_email, login_password)
        if e is None and user is not None:
            st.session_state.logged_in = True
            st.session_state.user_id = user["id"]
            st.success("Login Successful")
        else:
            st.error(f"Login Failed. {e}")
    st.subheader("Register")
    register_email = st.text_input("Email", key="register_email")
    register_password = st.text_input("Password", type="password", key="register_password")
    register_button = st.button("Register")
    if register_button:
        new_user, e = create_new_user(register_email, register_password)
        if e is None and new_user is not None:
            st.success("User Registered")
        else:
            st.error(f"Registration Failed. {e}")

if st.session_state.logged_in:
    user_id = st.session_state.user_id

    process_user_data(user_id)
    uploaded_files = st.file_uploader("Upload your PDFs here", accept_multiple_files=True, type="pdf")
    if uploaded_files:
        with st.spinner("Uploading your files..."):
            for file in uploaded_files:
                file_path, e = upload_file_to_storage(file, user_id)
                if e is None:
                    st.session_state[user_id]["pdf_paths"].append(file_path)
                    supabase.table("Files").insert(
                        {"id": str(uuid.uuid4()), "user_id": user_id, "file_path": file_path,
                         "uploaded_at": str(datetime.now())}).execute()
                else:
                    st.error(f"Upload failed {e}")
            st.session_state[user_id]["pdf_folder"] = "pdf_folder"
            process_user_data(user_id)
        st.success("Uploaded PDFs successfully!")

    query = st.text_input("Enter your question:")
    if query:
        with st.spinner("Generating response..."):
            response = generate_response(st.session_state[user_id]["qa_chain"], query)
            st.session_state[user_id]["chat_history"].append({"query": query, "response": response})
            supabase.table("Chat").insert(
                {"id": str(uuid.uuid4()), "user_id": user_id, "query": query, "response": response,
                 "timestamp": str(datetime.now())}).execute()
        st.write("Bot:", response)

    # Display Chat History
    if "chat_history" in st.session_state[user_id]:
        st.subheader("Chat History:")
        for chat in st.session_state[user_id]["chat_history"]:
            st.write(f"**You:** {chat['query']}")
            st.write(f"**Bot:** {chat['response']}")
