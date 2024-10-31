# RAG-1
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

- **Two Modes**: RAG mode for content-aware responses and a regular mode for standalone LLM responses.
- **Document Upload**: Supports uploading PDF documents for interaction in RAG mode.
- **User Interface**: Built with Streamlit, allowing model selection (Llama 3.1, Gemma2) and RAG activation.
- **Real-Time Interaction**: Responses are generated in real-time, with options to enable or disable RAG mode.

---

### Installation

1. **Clone the repository**:
```bash
git clone <repository_url>
cd RAG-1
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
3. **Ask Questions**: Enter your questions in the input box. In RAG mode, responses will be based on the PDF content uploaded.