from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# load existing chroma db
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

# prompt now includes chat history
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful study assistant. 
Answer the question using only the context below.
If the answer is not in the context, say 'I don't know.'

Context: {context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

chain = prompt | llm

# conversation memory — starts empty
chat_history = []

print("Study Assistant ready. Type 'quit' to exit.\n")

while True:
    question = input("You: ").strip()
    if question.lower() == "quit":
        break

    # retrieve relevant chunks
    results = db.similarity_search(question, k=6)

    if chat_history:

        rewrite_prompt = f"""
    Given this conversation history:
    {chat_history[-2].content}: {chat_history[-1].content}

    Rewrite this follow-up question as a standalone search query:
    "{question}"

    Return only the rewritten query, nothing else.
    """
        
        rewritten = llm.invoke(rewrite_prompt)
        search_query = rewritten.content[0]['text'] if isinstance(rewritten.content, list) else rewritten.content
        print(f"Search query: {search_query}")
    else:

        search_query = question

    results = db.similarity_search(search_query, k=6)

    context = "\n\n".join([doc.page_content for doc in results])
    sources = set([doc.metadata['source'] for doc in results])

    # call LLM with history
    response = chain.invoke({
        "context": context,
        "chat_history": chat_history,
        "question": question
    })

    answer = response.content[0]['text'] if isinstance(response.content, list) else response.content

    print(f"\nAssistant: {answer}")
    print(f"Sources: {', '.join(sources)}\n")

    # update memory
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=answer))
