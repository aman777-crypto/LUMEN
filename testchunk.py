from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyPDFLoader("Graph_Algorithms_Notes.pdf")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30
)
chunks = splitter.split_documents(documents)

print(f"Split {len(documents)} pages into {len(chunks)} chunks")

for i, chunk in enumerate(chunks[:110]):
    print(f"\n--- chunk {i} ---")
    print(f"metadata: {chunk.metadata}")
    print(f"length: {len(chunk.page_content)} chars")
    print(chunk.page_content[:150])