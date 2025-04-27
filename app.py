from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pandas as pd
from icalendar import Calendar
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import PyPDF2
import textract

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Sta alle origines toe voor alle endpoints

# Configuratie
DATA_DIR = "data"
ALLOWED_TEXT_EXTENSIONS = [".txt", ".pdf"]
ALLOWED_SPREADSHEET_EXTENSIONS = [".csv", ".xlsx"]
ALLOWED_EVENT_EXTENSIONS = [".ics"]
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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

# RAG-agent setup
def setup_rag_agent():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    documents = load_documents()
    if not documents:
        return None, "Geen documenten gevonden, upload eerst bestanden via /upload."
    vectorstore = FAISS.from_documents(documents, embeddings)
    if not OPENAI_API_KEY:
        return None, "OpenAI API-sleutel niet ingesteld."
    llm = OpenAI(api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    return qa_chain, None

# Query functie
def query_rag_agent(query, qa_chain):
    if qa_chain is None:
        return {"error": "Geen documenten beschikbaar, upload eerst bestanden via /upload."}
    result = qa_chain({"query": query})
    return {
        "answer": result["result"],
        "sources": [doc.metadata["source"] for doc in result["source_documents"]]
    }

# Flask API endpoints
qa_chain, error = setup_rag_agent()
if error:
    app.logger.warning(error)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "Geen bestand geüpload"}), 400
    file = request.files['file']
    file_path = os.path.join(DATA_DIR, file.filename)
    file.save(file_path)
    # Herlaad RAG-agent na upload
    global qa_chain
    qa_chain, error = setup_rag_agent()
    if error:
        return jsonify({"message": f"Bestand {file.filename} geüpload, maar: {error}"}), 200
    return jsonify({"message": f"Bestand {file.filename} geüpload"}), 200

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Geen query opgegeven"}), 400
    result = query_rag_agent(data['query'], qa_chain)
    return jsonify(result), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
