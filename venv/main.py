from fastapi import FastAPI, UploadFile, Form, HTTPException, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import fitz  # PyMuPDF
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Enable CORS (for frontend compatibility)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ATSResponse(BaseModel):
    score: float
    analysis: str

# Extract text from PDF
def extract_text_from_pdf(file):
    try:
        pdf_document = fitz.open(stream=file, filetype="pdf")
        text = ""
        for page in pdf_document:
            text += page.get_text()
        pdf_document.close()
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting text: {e}")

# Function to calculate similarity score using cosine similarity
def calculate_similarity(resume_text, job_desc):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_desc])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    score = similarity[0][0] * 100  # Convert to percentage
    return round(score, 2)

# Function to get ATS score from a free API (example using Twinword)
def get_ats_score(resume_text, job_desc):
    try:
        url = "https://api.twinword.com/api/ats/match/"
        headers = {
            "Content-Type": "application/json",
            "X-Twaip-Key": "YOUR_TWINWORD_API_KEY"  # Replace with your key
        }
        payload = {
            "resume": resume_text,
            "job_description": job_desc
        }
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return data.get("match_score", 0)
        else:
            return calculate_similarity(resume_text, job_desc)
    except Exception as e:
        print(f"Failed to fetch ATS score: {e}")
        return calculate_similarity(resume_text, job_desc)

# Endpoint to analyze resume and job description
@app.post("/analyze", response_model=ATSResponse)
async def analyze_resume(
    resume: UploadFile = File(...),  # ✅ Fix here
    job_description: str = Form(...)
):
    if not resume.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Please upload a PDF file")

    resume_text = extract_text_from_pdf(await resume.read())
    if not resume_text:
        raise HTTPException(status_code=400, detail="Failed to extract text from resume")

    # Get ATS score
    score = get_ats_score(resume_text, job_description)

    analysis = f"Your resume matches the job description by {score}%."

    return ATSResponse(score=score, analysis=analysis)

# Health check
@app.get("/health")
def health():
    return {"status": "ok"}

# Run server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)