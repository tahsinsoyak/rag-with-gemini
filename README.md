
# RAG with Gemini

This project is a Retrieval-Augmented Generation (RAG) application built using the Gemini model. It allows users to upload documents, which are then embedded and stored in a vector store. Users can ask questions, and the application retrieves relevant document chunks to generate answers using a language model.

## Features

- Upload PDF and TXT documents.
- Embed documents into a vector store using Google Generative AI embeddings.
- Retrieve relevant document chunks for question answering.
- Generate answers using the Gemini model.
- Interactive chat interface built with Streamlit.

## Environment Setup

1. **Clone the repository from GitHub:**

   ```bash
   git clone https://github.com/tahsinsoyak/rag-with-gemini.git
   cd gemini-rag
   ```

2. **Create and activate a conda environment:**

   ```bash
   conda create -n gemini_rag python=3.10
   conda activate gemini_rag
   ```

3. **Upgrade pip and install required packages:**

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Usage

1. **Run the Streamlit application:**

   ```bash
   streamlit run app.py
   ```

2. **Upload Documents:**

   - Use the sidebar to upload PDF or TXT files.
   - Adjust chunk size and retriever `k` as needed.

3. **Ask Questions:**

   - Once documents are embedded, navigate to the "Ask a question" section.
   - Enter your question in the chat input to receive an answer.

## Project Structure

- `app.py`: Main application file containing the Streamlit interface and logic.
- `requirements.txt`: List of required Python packages.
- `README.md`: Project documentation.
- `demo.webm`: Demo video showcasing the application in action. (you can open by dragging and dropping in browser)
- `Rag_with_Gemini.ipynb`: Jupyter Notebook demonstrating the RAG process with Gemini.

## Built With

- [Streamlit](https://streamlit.io/)
- [LangChain](https://langchain.com/)
- [Google Generative AI](https://ai.google/tools/)
