import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY is not set. Please set it in the environment variables.")
else:
    os.environ["OPENAI_API_KEY"] = api_key

    # Set up the UI
    st.title("PDF Chatbot")
    st.write("Upload a PDF file and ask a question")

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

    # Set up the Langchain model
    if uploaded_file is not None:
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_text(text)
        
        embeddings = OpenAIEmbeddings()
        pdfsearch = Chroma.from_texts(documents, embeddings)
        
        chain = ConversationalRetrievalChain.from_llm(
            ChatOpenAI(temperature=1),
            retriever=pdfsearch.as_retriever(search_kwargs={"k": 1}),
            return_source_documents=True
        )
        
        # Run the app
        if submit_button and question:
            response = generate_response(question, chain)
            display_response(response)
