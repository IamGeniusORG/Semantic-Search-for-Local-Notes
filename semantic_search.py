import os
import argparse
import glob
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()
from langchain_text_splitters import RecursiveCharacterTextSplitter
import PyPDF2
import docx
import textwrap

NOTES_DIR = "./notes"
INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "metadata.pkl"

def extract_text_from_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    try:
        if ext in ['.txt', '.md']:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        elif ext == '.pdf':
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n\n"
        elif ext == '.docx':
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    
    # Clean PDF noise
    text = text.replace("http://www.knowledgegate.in/gate", "")
    return text

def build_index(embedder):
    if not os.path.exists(NOTES_DIR):
        os.makedirs(NOTES_DIR)
        
    file_patterns = [
        os.path.join(NOTES_DIR, "*.txt"), 
        os.path.join(NOTES_DIR, "*.md"),
        os.path.join(NOTES_DIR, "*.pdf"),
        os.path.join(NOTES_DIR, "*.docx")
    ]
    files = []
    for pattern in file_patterns:
        files.extend(glob.glob(pattern))
        
    if not files:
        print(f"No text, markdown, PDF, or DOCX files found in '{NOTES_DIR}'. Please add some.")
        return None, None

    # Smarter Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    all_chunks = []
    metadata = []
    
    print("Reading and chunking files...")
    for file_path in files:
        text = extract_text_from_file(file_path)
        if text.strip():
            chunks = text_splitter.split_text(text)
            for chunk in chunks:
                all_chunks.append(chunk)
                metadata.append({"file": os.path.basename(file_path), "text": chunk})
                
    if not all_chunks:
        return None, None

    print("Generating embeddings... (This is fast!)")
    embeddings = embedder.encode(all_chunks, show_progress_bar=False)
    
    # FAISS setup
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    
    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, 'wb') as f:
        pickle.dump(metadata, f)
        
    print(f"Successfully indexed {len(files)} files into {len(all_chunks)} chunks!")
    return index, metadata

def load_index(embedder, rebuild=False):
    if rebuild or not os.path.exists(INDEX_FILE) or not os.path.exists(METADATA_FILE):
        print("Building new index database...")
        return build_index(embedder)
    
    print("Loading cached vector database...")
    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, 'rb') as f:
        metadata = pickle.load(f)
    return index, metadata

def main():
    parser = argparse.ArgumentParser(description="Semantic Search for Local Notes")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild the FAISS index")
    parser.add_argument("--query", type=str, help="Search query")
    args = parser.parse_args()

    print("Loading the sentence-transformers model (all-MiniLM-L6-v2)...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    print("Loading AI generative model (SmolLM-135M) for intelligent answers...")
    llm_pipe = pipeline("text-generation", model="HuggingFaceTB/SmolLM-135M-Instruct")

    index, metadata = load_index(embedder, rebuild=args.rebuild)
    if not index or not metadata:
        return

    print("\n" + "="*50)
    print(" SEMANTIC SEARCH READY ".center(50, '='))
    print("="*50)
    print("Type 'exit' or 'quit' to stop, or run with --rebuild to index new files.")
    
    while True:
        query = args.query
        if query:
            args.query = None
        else:
            try:
                query = input("\nEnter your search prompt: ")
            except (EOFError, KeyboardInterrupt):
                print("\nExiting search tool. Goodbye!")
                break

        if not query or not query.strip():
            continue
            
        if query.strip().lower() in ['exit', 'quit']:
            print("Exiting search tool. Goodbye!")
            break

        print(f"\nSearching for: '{query}'...")

        query_vector = embedder.encode([query]).astype('float32')
        k = 3
        distances, indices = index.search(query_vector, k)

        best_chunks = [metadata[i] for i in indices[0]]
        combined_context = " ".join([chunk['text'] for chunk in best_chunks])

        # RAG Generation
        messages = [
            {"role": "system", "content": "You are a direct and strictly factual AI. Output ONLY the answer using the provided context. Do NOT use conversational filler, do NOT say 'Here is the answer', and do NOT make small talk. Just write the answer directly and concisely."},
            {"role": "user", "content": f"Context:\n{combined_context}\n\nQuestion: {query}"}
        ]
        # Use the model's official chat template
        prompt = llm_pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        print("\n" + "*"*50)
        print(" ✨ AI GENERATED ANSWER ✨ ".center(50, '*'))
        try:
            result = llm_pipe(
                prompt, 
                max_new_tokens=150, 
                do_sample=True, 
                temperature=0.3,
                repetition_penalty=1.15, 
                return_full_text=False
            )
            generated_text = result[0]['generated_text'].strip()
            for line in generated_text.split('\n'):
                print(textwrap.fill(line, width=65) if line.strip() else "")
        except Exception as e:
            print("Could not generate answer.")
        print("*"*50)

        print("\n" + "="*50)
        print(" TOP MATCHING CONCEPTS ".center(50, '='))
        print("="*50)
        
        # FAISS returns L2 distances, lower is better. 
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            para_info = metadata[idx]
            print(f"\n[{rank + 1}] Distance: {dist:.4f} | Source: {para_info['file']}")
            print("-" * 50)
            for line in para_info['text'].split('\n'):
                print(textwrap.fill(line, width=65) if line.strip() else "")
            print("-" * 50)

if __name__ == "__main__":
    main()
