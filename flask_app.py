import os
import tempfile
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import PyPDF2

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Initialize components globally
documents = []
vectorstore = None
bm25_retriever = None
embeddings = GoogleGenerativeAIEmbeddings(model="embeddings-model")
llm = ChatGoogleGenerativeAI(model="model-name", temperature=0.7)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

def extract_text_from_pdf(file_path):
    with open(file_path, "rb") as f:
        pdf_reader = PyPDF2.PdfReader(f)
        return ''.join(page.extract_text() or "" for page in pdf_reader.pages)

@app.route('/')
def index():
    return render_template('index.html')

import mysql.connector
import re

def sanitize_db_name(name):
    return re.sub(r'\W+', '_', name).lower()

@app.route('/upload', methods=['POST'])
def upload_file():
    global documents, vectorstore, bm25_retriever

    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    filename = secure_filename(file.filename)
    paper_title = os.path.splitext(filename)[0]
    db_name = sanitize_db_name("research_paper")
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        # 🧾 Extract content
        if filename.lower().endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
        else:
            text = file.read().decode('utf-8')

        # 🧠 Embed text into LangChain docs
        chunks = text_splitter.split_text(text)
        doc_objects = [Document(page_content=chunk) for chunk in chunks]
        documents.extend(doc_objects)

        # 🔍 Build vector store
        if vectorstore is None:
            vectorstore = FAISS.from_documents(doc_objects, embeddings)
        else:
            vectorstore.add_documents(doc_objects)

        texts = [doc.page_content for doc in documents]
        bm25_retriever = BM25Retriever.from_texts(texts, metadatas=[{} for _ in texts])

        # 🧠 Basic abstract extraction (first 500 chars)
        abstract = text.strip().replace('\n', ' ')[:500]

        # 🗄️ MySQL: Create DB + Table + Insert record
        conn = mysql.connector.connect(host="localhost", user="root", password="Ammu@3851")
        cursor = conn.cursor()

        cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}`")
        cursor.execute(f"USE `{db_name}`")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS research_paper (
                paper_id INT AUTO_INCREMENT PRIMARY KEY,
                title VARCHAR(255) NOT NULL,
                abstract TEXT NOT NULL
            )
        """)
        cursor.execute("INSERT INTO research_paper (title, abstract) VALUES (%s, %s)", (paper_title, abstract))
        conn.commit()

        paper_id = cursor.lastrowid
        cursor.close()
        conn.close()

        return jsonify({
            'message': f'Successfully processed {filename}',
            'paper_id': paper_id,
            'database': db_name
        })

    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500


@app.route('/chat', methods=['POST'])
def chat():
    global documents, vectorstore, bm25_retriever

    data = request.get_json()
    query = data.get('query')

    if not documents:
        return jsonify({'response': "Please upload at least one research paper first."})

    try:
        faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        ensemble_retriever = EnsembleRetriever(
            retrievers=[faiss_retriever, bm25_retriever],
            weights=[0.7, 0.3]
        )

        relevant_docs = ensemble_retriever.get_relevant_documents(query)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        prompt = f"""You are a helpful research assistant. Answer the question based on the provided research paper context.

Context:
{context}

Question: {query}

Answer:"""

        response = llm.invoke(prompt)
        return jsonify({'response': response.content})

    except Exception as e:
        return jsonify({'error': f'Error generating response: {str(e)}'}), 500

if __name__ == '__main__':
    if not os.getenv("GOOGLE_API_KEY"):
        raise EnvironmentError("Please set the GOOGLE_API_KEY environment variable.")
    app.run(debug=True)
