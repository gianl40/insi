from flask import Flask, request, jsonify
import os
import asyncio
import pandas as pd
import json
from icalendar import Calendar
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
import requests
import PyPDF2
import textract

app = Flask(__name__)

# Configuratie
DATA_DIR = "data"
ALLOWED_TEXT_EXTENSIONS = [".txt", ".pdf"]
ALLOWED_SPREADSHEET_EXTENSIONS = [".csv", ".xlsx"]
ALLOWED_EVENT_EXTENSIONS = [".ics"]
XAI_API_KEY = os.getenv("XAI_API_KEY")

# Maak data directory
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Documentverwerking
def process_text_file(file_path):
    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    elif file_path.endswith(".pdf"):
        try:
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
                return text
        except:
            return textract.process(file_path).decode("utf-8")
    return ""

def process_spreadsheet(file_path):
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    else:
        return ""
    summary = f"Spreadsheet met {len(df)} rijen en kolommen: {list(df.columns)}\n"
    for _, row in df.iterrows():
        summary += f"Rij: {row.to_dict()}\n"
    return summary

def process_event(file_path):
    if file_path.endswith(".ics"):
        with open(file_path, "r") as f:
            cal = Calendar.from_ical(f.read())
            events = []
            for component in cal.walk():
                if component.name == "VEVENT":
                    event = {
                        "summary": str(component.get("summary", "")),
                        "start": str(component.get("dtstart").dt) if component.get("dtstart") else "",
                        "end": str(component.get("dtend").dt) if component.get("dtend") else "",
                        "description": str(component.get("description", ""))
                    }
                    events.append(event)
            return "\n".join([f"Evenement: {e['summary']} op {e['start']} - {e['end']}: {e['description']}" for e in events])
    return ""

def load_documents():
    documents = []
    for file_name in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, file_name)
        if any(file_name.endswith(ext) for ext in ALLOWED_TEXT_EXTENSIONS):
            content = process_text_file(file_path)
            documents.append(Document(page_content=content, metadata={"source": file_name}))
        elif any(file_name.endswith(ext) for ext in ALLOWED_SPREADSHEET_EXTENSIONS):
            content = process_spreadsheet(file_path)
            documents.append(Document(page_content=content, metadata={"source": file_name}))
        elif any(file_name.endswith(ext) for ext in ALLOWED_EVENT_EXTENSIONS):
            content = process_event(file_path)
            documents.append(Document(page_content=content, metadata={"source": file_name}))
    return documents

# xAI API-integratie
class XAILLM:
    def __init__(self, api_key):
        self.api_key = api_key
        self.api_url = "https://api.x.ai/v1/completions"

    def __call__(self, prompt, **kwargs):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", 100),
            "temperature": kwargs.get("temperature", 0.7)
        }
        response = requests.post(self.api_url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()["choices"][0]["text"]
        else:
            raise Exception(f"API error: {response.status_code} - {response.text}")

# RAG-agent setup
def setup_rag_agent():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    documents = load_documents()
    if not documents:
        return None, "Geen documenten gevonden om te verwerken."
    vectorstore = FAISS.from_documents(documents, embeddings)
    if not XAI_API_KEY:
        return None, "xAI API-sleutel niet ingesteld."
    llm = XAILLM(api_key=XAI_API_KEY)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    return qa_chain, None

# Query functie
async def query_rag_agent(query, qa_chain):
    result = qa_chain({"query": query})
    return {
        "answer": result["result"],
        "sources": [doc.metadata["source"] for doc in result["source_documents"]]
    }

# Flask API endpoints
qa_chain, error = setup_rag_agent()
if error:
    raise Exception(error)

@app.route('/upload', methods=['POST'])
async def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "Geen bestand geüpload"}), 400
    file = request.files['file']
    file_path = os.path.join(DATA_DIR, file.filename)
    file.save(file_path)
    return jsonify({"message": f"Bestand {file.filename} geüpload"}), 200

@app.route('/query', methods=['POST'])
async def query():
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Geen query opgegeven"}), 400
    result = await query_rag_agent(data['query'], qa_chain)
    return jsonify(result), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))