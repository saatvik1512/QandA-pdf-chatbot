import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
import pdfplumber
import tempfile
import os

# Set your Gemini API key
os.environ["GOOGLE_API_KEY"] = "***********"

# Initialize models
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Sidebar for settings
st.sidebar.title("Settings")
dark_mode = st.sidebar.checkbox("Enable Dark Mode")
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type="pdf")

# Apply dark mode styling if enabled
if dark_mode:
    st.markdown(
        """
        <style>
        .main {
            background-color: #121212;
            color: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <style>
        .main {
            background-color: #ffffff;
            color: #000000;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Header section
st.image("https://via.placeholder.com/150", width=150)  # Replace with your logo URL
st.title("Document Q&A")
st.write("Upload a PDF document and ask questions about its content.")

# Main functionality
if uploaded_file:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    with st.spinner("Extracting text from the PDF..."):
        # Use pdfplumber to extract text
        with pdfplumber.open(temp_file_path) as pdf:
            text = "".join(page.extract_text() for page in pdf.pages)

    # Check if text was successfully extracted
    if text.strip():
        with st.spinner("Processing the document..."):
            # Split the text into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            split_documents = text_splitter.create_documents([text])

            # Create a FAISS vector store
            vector_store = FAISS.from_documents(split_documents, embedding_model)

            # Create a retriever and a QA chain
            retriever = vector_store.as_retriever()
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever
            )

        # User input for queries
        query = st.text_input("Ask a question about the document:")
        if query:
            with st.spinner("Fetching the answer..."):
                # Get the answer from the QA chain
                answer = qa_chain.run(query)
                st.markdown(
                    f"<div style='font-size: 20px; color: green;'>**Answer:** {answer}</div>",
                    unsafe_allow_html=True
                )
    else:
        st.error("Failed to extract text from the PDF. Please check the file and try again.")

# Footer
st.sidebar.write("---")
st.sidebar.write("Built with ❤️ using [LangChain](https://langchain.com) and [Streamlit](https://streamlit.io)")
