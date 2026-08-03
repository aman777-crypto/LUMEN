import time
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# load and split
loader = PyPDFLoader("Graph_Algorithms_Notes.pdf")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
chunks = splitter.split_documents(documents)
print(f"Total chunks to embed: {len(chunks)}")

# embed and store in batches
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
BATCH_SIZE = 50
db = None

for i in range(0, len(chunks), BATCH_SIZE):
    batch = chunks[i:i + BATCH_SIZE]
    print(f"Embedding batch {i//BATCH_SIZE + 1}: chunks {i} to {i+len(batch)-1}...")

    if db is None:
        db = Chroma.from_documents(
            documents=batch,
            embedding=embeddings,
            persist_directory="chroma_db"
        )
    else:
        db.add_documents(batch)

    print(f"  done. sleeping 65s before next batch...")
    if i + BATCH_SIZE < len(chunks):
        time.sleep(65)

print(f"\nAll done. Total chunks in Chroma: {db._collection.count()}")

# now search
question = "How does Dijkstra's algorithm work?"
results = db.similarity_search(question, k=3)
print(f"\nTop 3 results for: '{question}'\n")
for i, doc in enumerate(results):
    print(f"--- result {i} ---")
    print(f"page: {doc.metadata['page']}")
    print(doc.page_content[:200])
    print()