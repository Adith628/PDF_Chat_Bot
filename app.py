import streamlit as st
from PyPDF2 import PdfReader

# Sidebar
with st.sidebar:
    st.title("LLM Chat App")
    st.markdown("Welcome to the LLM Chat App!")
    st.write("This is a chat app that uses the LLM model to generate responses to your messages.")
    
    
def main():
    st.header("Chat with the LLM model")
    
    # upload file
    pdf = st.file_uploader("Upload a PDF file", type="pdf" )
    
    if pdf :
        pdf_reader = PdfReader(pdf)
        st.write(pdf_reader)

if __name__ == "__main__":
    main()
