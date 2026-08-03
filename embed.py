from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# load and split (same as before)
loader = PyPDFLoader("Graph_Algorithms_Notes.pdf")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
chunks = splitter.split_documents(documents)

# embed just ONE chunk manually
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
vector = embeddings.embed_query(chunks[5].page_content)

print(f"chunk text: {chunks[5].page_content[:100]}")
print(f"vector length: {len(vector)}")
print(f"first 5 values: {vector[:5]}")