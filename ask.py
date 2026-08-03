from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# load existing chroma db — no re-embedding needed
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

# retrieve
question = "How does Dijkstra's algorithm work?"
results = db.similarity_search(question, k=3)

# build context from retrieved chunks
context = "\n\n".join([doc.page_content for doc in results])

# build prompt
prompt = ChatPromptTemplate.from_template("""
You are a helpful study assistant. Answer the question using only the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}
""")

# call LLM
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")
chain = prompt | llm

response = chain.invoke({"context": context, "question": question})
print("Answer:")
print(response.content[0]['text'])
print("\nSources:")
for doc in results:
    print(f"  page {doc.metadata['page']} — {doc.page_content[:60]}...")