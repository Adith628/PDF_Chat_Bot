import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv
# Gemini support
import google.generativeai as genai

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not openai_api_key:
    st.error("OPENAI_API_KEY is not set. Please set it in the environment variables.")
if not gemini_api_key:
    st.warning("GEMINI_API_KEY is not set. Gemini model will not work until you set it in the environment variables.")
else:
    os.environ["OPENAI_API_KEY"] = openai_api_key

    # Set up the UI
    st.title("PDF Chatbot")
    st.write("Upload a PDF file and ask a question")

    # Model selection
    model_choice = st.selectbox("Select Model", ["OpenAI", "Gemini"])

    # Create a file uploader
    uploaded_file = st.file_uploader("Select a PDF file", type=["pdf"])

    # Create a text input for the user's question
    question = st.text_input("Ask a question")

    # Create a button to submit the question
    submit_button = st.button("Submit")

    # Define a function to generate a response
    def generate_response(question, chain):
        result = chain({"question": question, "chat_history": []}, return_only_outputs=True)
        return result["answer"]

    # Define a function to display the response
    def display_response(response):
        st.write("Response:")
        st.write(response)

    # Set up the Langchain or Gemini model
    if uploaded_file is not None:
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_text(text)

        if model_choice == "OpenAI":
            embeddings = OpenAIEmbeddings()
            pdfsearch = Chroma.from_texts(documents, embeddings)
            chain = ConversationalRetrievalChain.from_llm(
                ChatOpenAI(temperature=1),
                retriever=pdfsearch.as_retriever(search_kwargs={"k": 1}),
                return_source_documents=True
            )
            # Run the app
            if submit_button and question and chain:
                response = generate_response(question, chain)
                display_response(response)
        elif model_choice == "Gemini":
            if not gemini_api_key:
                st.error("GEMINI_API_KEY is not set. Please set it in the environment variables.")
            else:
                genai.configure(api_key=gemini_api_key)
                model = genai.GenerativeModel('models/gemini-2.0-flash')
                # Simple retrieval: concatenate all text chunks
                context = "\n".join(documents)
                def gemini_generate_response(question, context):
                    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
                    response = model.generate_content(prompt)
                    return response.text if hasattr(response, 'text') else str(response)
                if submit_button and question:
                    response = gemini_generate_response(question, context)
                    display_response(response)
