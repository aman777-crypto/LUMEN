# ALL imports at the top
from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
import os
import time

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)

UPLOAD_FOLDER = 'docs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    if not file:
        return jsonify({'message': 'No file received'})

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)


    db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    existing = db.get(where={"source": filepath})
    if existing and len(existing['ids']) > 0:
        return jsonify({'message': f'{file.filename} already indexed — {len(existing["ids"])} chunks exist. Skipping.'})

    loader = PyPDFLoader(filepath)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    
    BATCH_SIZE = 50
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        db.add_documents(batch)
        if i + BATCH_SIZE < len(chunks):
            time.sleep(65)

    return jsonify({'message': f'Indexed {len(chunks)} chunks from {file.filename}'})

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get('question')
    if not question:
        return jsonify({'answer': 'No question received'})

    # load chat history from session
    history_data = session.get('chat_history', [])
    chat_history = []
    for msg in history_data:
        if msg['role'] == 'human':
            chat_history.append(HumanMessage(content=msg['content']))
        else:
            chat_history.append(AIMessage(content=msg['content']))

    # query rewriting
    db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    if chat_history:
        rewrite_prompt = f"""
Given this conversation:
User: {chat_history[-2].content if len(chat_history) >= 2 else ''}
Assistant: {chat_history[-1].content if chat_history else ''}
Rewrite this as a standalone search query: "{question}"
Return only the rewritten query.
"""
        rewritten = llm.invoke(rewrite_prompt)
        search_query = rewritten.content[0]['text'] if isinstance(rewritten.content, list) else rewritten.content
    else:
        search_query = question

    # retrieve and answer
    results = db.similarity_search(search_query, k=6)
    context = "\n\n".join([doc.page_content for doc in results])
    sources = list(set([f"{doc.metadata['source']} p.{doc.metadata['page']}" for doc in results]))

    prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful study assistant for a computer science student.
    You are given context chunks retrieved from study notes and textbooks.

        Instructions:
            - Answer clearly and concisely using only the provided context
            - Use bullet points or numbered lists for multi-step explanations
            - If the context contains code or algorithms, include them in your answer
            - If the answer spans multiple topics, organize with clear headings
            - If the context doesn't contain enough information, say exactly what is missing
            - Always answer in simple language a student can understand

    Context:
    {context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    chain = prompt | llm
    response = chain.invoke({"context": context, "chat_history": chat_history, "question": question})
    answer = response.content[0]['text'] if isinstance(response.content, list) else response.content

    # save updated history to session
    history_data.append({'role': 'human', 'content': question})
    history_data.append({'role': 'assistant', 'content': answer})
    session['chat_history'] = history_data

    return jsonify({'answer': answer, 'sources': sources})

#clear
@app.route('/clear', methods=['POST'])
def clear():
    session.pop('chat_history', None)
    return jsonify({'message': 'Chat history cleared'})

#status
@app.route('/status')
def status():
    db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    count = db._collection.count()
    return jsonify({'chunks': count})

if __name__ == '__main__':
    app.run(debug=True)