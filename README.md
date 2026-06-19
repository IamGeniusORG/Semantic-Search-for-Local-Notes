<p align="center">
  <img src="Tests/terminal%20tests.png" alt="Semantic Search for Local Notes Banner" width="100%">
</p>

# Semantic Search for Local Notes

A lightweight and powerful AI-powered semantic search tool for your local notes. Unlike traditional keyword-based search, this tool understands the **meaning** of your questions and retrieves the most conceptually relevant paragraphs from your local Markdown (`.md`), Text (`.txt`), and PDF (`.pdf`) documents.

## Features
- **Semantic Understanding**: Finds answers based on concept and meaning, not just exact keyword matches.
- **Offline & Local**: Runs entirely locally on your machine using the `all-MiniLM-L6-v2` model from Hugging Face. No API keys or internet connection required after the initial setup.
- **Multiple Formats**: Automatically extracts and parses text from `.txt`, `.md`, and `.pdf` files.
- **Clean Formatting**: Strips PDF noise, fixes messy bullet points, and wraps text for a beautiful command-line reading experience.

## Getting Started

### Prerequisites
- Python 3.8+ installed on your system.

### Installation

1. **Clone the repository:**
   ```powershell
   git clone https://github.com/IamGeniusORG/Semantic-Search-for-Local-Notes.git
   cd "Semantic Search for Local Notes"
   ```

2. **Set up a virtual environment:**
   ```powershell
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   - **Windows:**
     ```powershell
     .\venv\Scripts\activate
     ```
   - **Mac/Linux:**
     ```bash
     source venv/bin/activate
     ```

4. **Install the dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

### Usage

1. A sample note (`notes/sample_note.md`) is included in the repository for demo purposes so you can test it immediately! You can drop any of your actual `.txt`, `.md`, or `.pdf` files inside this `notes` folder.
2. Run the search script:
   ```powershell
   python semantic_search.py
   ```
3. Type your question when prompted and let the AI find the exact concepts you're looking for! Type `exit` or `quit` to stop the tool.
