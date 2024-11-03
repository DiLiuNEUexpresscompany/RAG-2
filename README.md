# RAG-2
# Conversational System with RAG Support

This project is a conversation system enabling users to upload PDF files and interact with their content. It utilizes the **Llama** model for real-time conversation, supporting two modes: Retrieval-Augmented Generation (RAG) and standard conversation. This README provides installation instructions and an overview of project functionality.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [License](#license)

---

### Project Overview

The goal is to implement a basic Retrieval-Augmented Generation (RAG) system, using **Llama** for text generation and embedding with Ollama, and compare it with standalone LLMs. The system processes uploaded PDFs for interaction in RAG mode and provides real-time response streaming.

### Features

- **Mode Switching**: Users can easily switch between RAG and non-RAG modes, as well as between different LLMs.

- **Response Time Management**: Users can view response times, making it transparent when responses might be delayed due to RAG processing.

- **User Feedback**: The interface now includes a feature for users to rate the quality of responses, enabling continuous improvement based on user input.

- **Two Modes**: RAG mode for content-aware responses and a regular mode for standalone LLM responses.

- **Document Upload**: Supports uploading PDF, DOC, DOCX, Markdown, and TXT for interaction in RAG mode.

- **User Interface**: Built with Streamlit, allowing model selection (Llama 3.1, Gemma2) and RAG activation.

- **Real-Time Interaction**: Responses are generated in real-time, with options to enable or disable RAG mode.

- **Fetch Web Content**: Users can input a URL to fetch and scrape web content, which is then added to the knowledge base for future interactions. This feature ensures that the chatbot has access to the latest information available online, further enhancing response accuracy and relevance.

- **Maintain Local Knowledge Base**: The local knowledge base is incrementally updated with new content, avoiding the need for a complete rebuild. JSON files (docstore.json, index_store.json, vector_store.json) are used to store and manage the knowledge base, providing efficient access to documents and ensuring that the knowledge base remains current and manageable.

- **Automatic Local Knowledge Base Updates**: The system includes a scheduler that automatically updates the local knowledge base at predefined intervals. This ensures that the knowledge base is consistently up-to-date, incorporating newly available domain-specific documents without manual intervention.



---

### Installation

1. **Clone the repository**:
```bash
git clone <repository_url>
cd RAG-2
```
### Install Dependencies
Ensure you have Python 3.8+ installed, then run:
```bash
pip install -r requirements.txt
```
### Run the Application
To start the application, run:
 ```bash
streamlit run app.py
 ```
## Usage

1. **Upload PDF**: In the Streamlit sidebar, upload the PDF document you wish to interact with.
2. **Choose Conversation Mode**: Toggle the `use_rag` checkbox to switch between standard and RAG modes. 
6. **Ask Questions**: Enter your questions in the input box. In RAG mode, responses will be based on the PDF content uploaded.