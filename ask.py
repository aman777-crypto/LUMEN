from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# must be the SAME embedding model used in app.py
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

# retrieve
question = "How does Dijkstra's algorithm work?"
results = db.similarity_search(question, k=3)

context = "\n\n".join([doc.page_content for doc in results])

prompt = ChatPromptTemplate.from_template("""
You are a helpful study assistant. Answer the question using only the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}
""")

llm = ChatGroq(model="openai/gpt-oss-120b")
chain = prompt | llm

response = chain.invoke({"context": context, "question": question})
print("Answer:")
print(response.content)
print("\nSources:")
for doc in results:
    print(f"  page {doc.metadata.get('page', '?')} — {doc.page_content[:60]}...")