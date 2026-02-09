import os
from flask import Flask, render_template, request, jsonify
from PyPDF2 import PdfReader
from langchain_community.llms import Ollama

app = Flask(__name__)

# --- CONFIGURATION ---
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize Ollama (Make sure you have Ollama running: 'ollama serve')
# You can change 'llama3' to 'mistral' or 'gemma' depending on what you have installed.
llm = Ollama(model="llama3") 

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload_resume', methods=['POST'])
def upload_resume():
    if 'resume' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['resume']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Extract text for context
        resume_text = extract_text_from_pdf(filepath)
        
        # Store resume text in a global variable or session for simplicity in this MVP
        # In production, use a database.
        app.config['CURRENT_RESUME'] = resume_text
        
        return jsonify({"message": "Resume uploaded successfully!", "text_preview": resume_text[:200] + "..."})

@app.route('/generate_question', methods=['POST'])
def generate_question():
    data = request.json
    role = data.get('role', 'Software Engineer')
    resume_text = app.config.get('CURRENT_RESUME', '')

    prompt = f"""
    You are an expert technical interviewer. 
    Based on the following resume snippet and the target role '{role}', generate ONE technical interview question.
    
    Resume Snippet: {resume_text[:2000]}
    
    Output ONLY the question. Do not add introductory text.
    """
    
    try:
        # AI Call
        question = llm.invoke(prompt)
        return jsonify({"question": question.strip()})
    except Exception as e:
        return jsonify({"question": "Could not generate question. Ensure Ollama is running.", "error": str(e)})

@app.route('/evaluate_answer', methods=['POST'])
def evaluate_answer():
    data = request.json
    question = data.get('question')
    user_answer = data.get('answer')
    
    prompt = f"""
    You are an expert interviewer. Evaluate the candidate's answer.
    
    Question: {question}
    Candidate's Answer: {user_answer}
    
    Provide a JSON-style response with:
    1. Score (out of 10)
    2. Feedback (What was good, what was missing)
    3. Suggested Improvement
    
    Keep the response concise.
    """
    
    try:
        feedback = llm.invoke(prompt)
        return jsonify({"feedback": feedback})
    except Exception as e:
        return jsonify({"feedback": "Error evaluating response."})

if __name__ == '__main__':
    app.run(debug=True)