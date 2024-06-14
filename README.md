# PDF Chatbot

This project is a PDF Chatbot application built using Streamlit, Langchain, and OpenAI's language model. The application allows users to upload a PDF file and ask questions about the content. The chatbot processes the PDF, splits the text into manageable chunks, embeds the text using OpenAI's embeddings, and then answers questions based on the PDF content.

## Features

- Upload a PDF file.
- Ask questions about the PDF content.
- Get accurate answers based on the PDF content using OpenAI's language model.

## Requirements

- Python 3.7+
- Streamlit
- Langchain
- OpenAI API Key
- PyPDF2
- python-dotenv

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/pdf-chatbot.git
cd pdf-chatbot
```
2. Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Create a .env file in the project directory and add your OpenAI API key:

```
OPENAI_API_KEY=your_openai_api_key_here
```
5. Run the Streamlit application:

```bash
streamlit run app.py
```
