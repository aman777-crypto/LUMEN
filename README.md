LANGCAHIN_RAG

A simple **RAG (Retrieval-Augmented Generation) based document chatbot** that lets users ask questions from their PDF/document files.

## 💡 Idea

The main idea is simple:

> **Upload your documents → Ask a question → Get an answer from your documents**

Instead of searching through a large PDF manually, the user can simply ask a question and the system finds the relevant information and generates an answer.

## 🔄 How It Works

```text
        📄 PDF / Documents
               ↓
        🔍 RAG Retrieval
               ↓
        🧠 LangChain
               ↓
          💬 Answer
               ↓
        📌 Source of Answer
```

## ✨ Main Features

* 📄 Upload and work with documents
* 💬 Ask questions about the documents
* 🔍 Find relevant information using RAG
* 🧠 Generate answers using an LLM
* 📌 Show the source of the information
* 🕘 Keep track of previous questions

## 🛠️ Technologies Used

* **Python**
* **LangChain**
* **ChromaDB**
* **Streamlit**
* **LLMs / Embeddings**

## ▶️ Run the Project

Clone the repository:

```bash
git clone https://github.com/aman777-crypto/LANGCAHIN_RAG.git
```

Go into the project:

```bash
cd LANGCAHIN_RAG
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run ui.py
```

## 🎯 Example

A user uploads a college PDF and asks:

> **"What is the syllabus of Unit 3?"**

The system searches the document, finds the relevant information and generates an answer.

## 🚀 Future Improvements

* Better chat history
* Multiple document support
* Better source/citation display
* Document upload directly from the UI
* More LLM choices
* Improved UI

---

### 👨‍💻 Author

AMAN

GitHub: [aman777-crypto](https://github.com/aman777-crypto)
website link : https://lumen-oxpx.onrender.com

