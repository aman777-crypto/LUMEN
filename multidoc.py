import time
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# load all PDFs from the docs folder
docs_folder = "docs"
all_chunks = []

for filename in os.listdir(docs_folder):
    if filename.endswith(".pdf"):
        path = os.path.join(docs_folder, filename)
        print(f"Loading {filename}...")
        loader = PyPDFLoader(path)
        documents = loader.load()
        print(f"  {len(documents)} pages")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=30
        )
        chunks = splitter.split_documents(documents)
        print(f"  {len(chunks)} chunks")
        all_chunks.extend(chunks)

print(f"\nTotal chunks across all PDFs: {len(all_chunks)}")

# store in chroma in batches
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
BATCH_SIZE = 50
db = None

for i in range(0, len(all_chunks), BATCH_SIZE):
    batch = all_chunks[i:i + BATCH_SIZE]
    print(f"Embedding batch {i//BATCH_SIZE + 1}: chunks {i} to {i+len(batch)-1}...")

    if db is None:
        db = Chroma.from_documents(
            documents=batch,
            embedding=embeddings,
            persist_directory="chroma_db"
        )
    else:
        db.add_documents(batch)

    if i + BATCH_SIZE < len(all_chunks):
        print("  sleeping 65s...")
        time.sleep(65)

print(f"\nTotal chunks in Chroma: {db._collection.count()}")

# test retrieval across different topics
questions = [
    "What is process scheduling?",
    "How does binary search work?",
    "What is a sorting algorithm?",
    "What is a critical section?"
]

for question in questions:
    print(f"\nQ: {question}")
    results = db.similarity_search(question, k=2)
    for doc in results:
        print(f"  source: {doc.metadata['source']} | page: {doc.metadata['page']}")
        print(f"  {doc.page_content[:100]}...")