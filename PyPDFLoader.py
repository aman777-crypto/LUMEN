from langchain_community.document_loaders import PyPDFLoader

loader =PyPDFLoader("Graph_Algorithms_Notes.pdf")
documents = loader.load()

print(f"loaded{len(documents)} pages")
print(documents[0].page_content[:300])