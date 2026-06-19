import os
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
import glob
import re
import textwrap

def load_documents(directory):
    """Reads all text, markdown, and PDF documents from the target directory."""
    documents = []
    # Find all .txt, .md, and .pdf files
    file_patterns = [
        os.path.join(directory, "*.txt"), 
        os.path.join(directory, "*.md"),
        os.path.join(directory, "*.pdf")
    ]
    files = []
    for pattern in file_patterns:
        files.extend(glob.glob(pattern))
    
    for file_path in files:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf':
            try:
                import PyPDF2
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    content = []
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            content.append(page_text)
                    documents.append({"file": os.path.basename(file_path), "content": "\n\n".join(content)})
            except Exception as e:
                print(f"Error reading PDF {file_path}: {e}")
        else:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                # Store filename and content
                documents.append({"file": os.path.basename(file_path), "content": content})
            
    return documents

def split_into_paragraphs(text):
    """Splits text into paragraphs using basic punctuation rules and cleans up artifacts."""
    # Split by double newlines for typical markdown/text paragraphs
    paragraphs = re.split(r'\n\s*\n', text)
    
    # Clean up whitespace and filter out empty paragraphs
    cleaned_paragraphs = []
    for p in paragraphs:
        # Clean common PDF noise (like specific URLs, weird bullets)
        cleaned = p.replace("http://www.knowledgegate.in/gate", "")
        # Replace unicode bullets or weird characters
        cleaned = re.sub(r'[•●]', '-', cleaned)
        # Collapse multiple spaces
        cleaned = re.sub(r' +', ' ', cleaned)
        # Strip trailing/leading whitespace
        cleaned = cleaned.strip()
        
        if cleaned:
            cleaned_paragraphs.append(cleaned)
            
    return cleaned_paragraphs

def cosine_similarity(vec1, vec2):
    """Computes cosine similarity between two vectors using numpy."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

def main():
    parser = argparse.ArgumentParser(description="Semantic Search for Local Notes")
    parser.add_argument("--notes_dir", type=str, default="./notes", help="Directory containing text/markdown notes")
    parser.add_argument("--query", type=str, help="Search query (if not provided, will prompt interactively)")
    args = parser.parse_args()

    # 1. Read documents
    if not os.path.exists(args.notes_dir):
        print(f"Notes directory '{args.notes_dir}' does not exist. Creating it...")
        os.makedirs(args.notes_dir)
        print(f"Please add some .txt, .md, or .pdf files to '{args.notes_dir}' and run again.")
        return

    documents = load_documents(args.notes_dir)
    if not documents:
        print(f"No text, markdown, or PDF files found in '{args.notes_dir}'. Please add some and run again.")
        return

    # 2. Split into paragraphs
    all_paragraphs = []
    for doc in documents:
        paragraphs = split_into_paragraphs(doc["content"])
        for p in paragraphs:
            all_paragraphs.append({
                "file": doc["file"],
                "text": p
            })

    if not all_paragraphs:
        print("No content found in the documents.")
        return

    print(f"Loaded {len(documents)} document(s) containing {len(all_paragraphs)} paragraph(s).")
    
    # 3. Load model
    print("Loading the sentence-transformers model (all-MiniLM-L6-v2)...")
    # This might take a moment to download on the first run
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # 4. Compute embeddings for all paragraphs
    print("Encoding paragraphs...")
    paragraph_texts = [p["text"] for p in all_paragraphs]
    paragraph_embeddings = model.encode(paragraph_texts)

    # 5. Interactive search loop
    print("\nType 'exit' or 'quit' to stop the search tool.")
    
    while True:
        query = args.query
        if query:
            args.query = None  # Clear it so it asks for input on the next loop
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

        print(f"\nSearching for: '{query}'")

        # 6. Encode the query
        query_embedding = model.encode([query])[0]

        # 7. Compute similarities
        similarities = []
        for i, p_emb in enumerate(paragraph_embeddings):
            sim = cosine_similarity(query_embedding, p_emb)
            similarities.append((i, sim))

        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)

        # 8. Print the top 3 matching concepts
        print("\n" + "="*50)
        print(" TOP 3 MATCHING CONCEPTS ".center(50, '='))
        print("="*50)
        top_k = min(3, len(similarities))
        for idx in range(top_k):
            orig_idx, sim_score = similarities[idx]
            para_info = all_paragraphs[orig_idx]
            print(f"\n[{idx + 1}] Score: {sim_score:.4f} | Source: {para_info['file']}")
            print("-" * 50)
            
            # Wrap the text to 80 characters for better readability
            wrapped_text = textwrap.fill(para_info['text'], width=80)
            print(wrapped_text)
            print("-" * 50)

if __name__ == "__main__":
    main()
