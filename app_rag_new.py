import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_mistralai import ChatMistralAI
from langchain.prompts import PromptTemplate
import tempfile
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# Make sure to pass the API key to ChatMistralAI
llm = ChatMistralAI(
    model="mistral-large-latest", 
    temperature=0,
    mistral_api_key=MISTRAL_API_KEY
)

prompt = PromptTemplate.from_template(
    """
You are an expert in solving students questions on the material they have given you.
Based only on the summaries provided below, answer the given question for the farmers.
Give all details as much as possible.
If you don't know the answer, respond with "I don't know".
Do not generate new questions.
Do not make up your own answer from outside the source material.
Do not write anything after the answer.
Write the answer in the language provided by the user. Do not write in any other language.

Source material: {context}
language: {language}

Question: {question}
Answer:
"""
)

chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)

def split_docs(documents, chunk_size=512, chunk_overlap=30):
    """Split documents into text chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return text_splitter.split_documents(documents)

def get_similar_docs(index, query, k=5):
    return index.similarity_search(query, k=k)

st.title("Document-based Question Answering for students")

with st.sidebar:
    st.title("AI PDF QA App")
    uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

relevant_questions = [
    "What is the main topic of the document?",
    "Can you summarize the key points of the document?",
    "What are the main arguments presented in the document?",
    "What is biology?"
]

selected_question = st.sidebar.selectbox(
    "Choose a predefined question (or enter your own below):",
    relevant_questions
)

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        docs = split_docs(documents)
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        index = FAISS.from_documents(docs, embeddings)

        query = st.text_input(
            "Ask a question related to the document", 
            value=selected_question if selected_question else ""
        )

        languages = [
            "English", "French", "Spanish", "German", "Chinese", "Japanese", "Hindi", "Arabic", 
            "Russian", "Portuguese", "Italian", "Korean", "Turkish", "Dutch", "Greek",
            "Swedish", "Polish", "Thai", "Indonesian", "Czech", "Hebrew", "Danish",
            "Finnish", "Norwegian", "Hungarian", "Romanian", "Bulgarian", "Ukrainian"
        ]

        language = st.selectbox(
            "Select the language you want to translate into:", 
            languages
        )

        if query:
            with st.spinner("Searching for answer..."):
                relevant_docs = get_similar_docs(index, query)
                result = chain.run(
                    input_documents=relevant_docs, 
                    question=query, 
                    language=language
                )
                st.markdown("### Answer")
                st.write(result)
    finally:
        os.unlink(tmp_path)
else:
    st.info("Please upload a PDF file to get started.")
