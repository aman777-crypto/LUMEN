from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

print(f"Total chunks in Chroma: {db._collection.count()}")

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