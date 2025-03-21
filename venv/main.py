from fastapi import FastAPI, UploadFile, Form, HTTPException, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import fitz  # PyMuPDF
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from textblob import TextBlob
import textstat

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ATSResponse(BaseModel):
    ats_score: float
    keyword_match_rate: float
    format_score: float
    content_quality: float
    education_metrics: float
    qualification_score: float
    career_progress: float
    analysis: str
    suggestions: List[str]  # New field for improvement suggestions

def extract_text_from_pdf(file):
    try:
        pdf_document = fitz.open(stream=file, filetype="pdf")
        text = ""
        metadata = {"pages": pdf_document.page_count}
        for page in pdf_document:
            text += page.get_text()
        pdf_document.close()
        return text.strip(), metadata
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting text: {e}")

def calculate_keyword_match(resume_text, job_desc):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_desc])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return round(similarity[0][0] * 100, 2)

def analyze_format(resume_text, metadata):
    score = 100.0
    word_count = len(resume_text.split())
    if word_count < 150 or word_count > 1000:
        score -= 20
    if metadata["pages"] > 2:
        score -= 10
    if resume_text.count('•') < 3:
        score -= 15
    return max(0, min(100, round(score, 2)))

def analyze_content_quality(resume_text):
    readability = textstat.flesch_reading_ease(resume_text)
    readability_score = min(100, max(0, readability))
    blob = TextBlob(resume_text)
    sentiment = blob.sentiment.polarity
    return round((readability_score * 0.7 + (sentiment + 1) * 15), 2)

def analyze_education(resume_text):
    education_keywords = ['bachelor', 'master', 'phd', 'degree', 'university', 'college']
    score = 0
    lines = resume_text.lower().split('\n')
    for line in lines:
        if any(keyword in line for keyword in education_keywords):
            score += 30
            if re.search(r'\d{4}\s*-\s*\d{4}|\d{4}\s*grad', line):
                score += 20
    return min(100, score)

def analyze_qualifications(resume_text, job_desc):
    skill_keywords = re.findall(r'\b\w+\b', job_desc.lower())
    resume_lower = resume_text.lower()
    matches = sum(1 for keyword in skill_keywords if keyword in resume_lower)
    return round(min(100, (matches / max(1, len(skill_keywords))) * 100), 2)

def analyze_career_progress(resume_text):
    job_titles = re.findall(r'(manager|senior|lead|director|specialist|engineer|developer)', 
                          resume_text.lower())
    years = len(re.findall(r'\b(19|20)\d{2}\b', resume_text))
    score = min(100, len(set(job_titles)) * 20 + min(years * 5, 40))
    return round(score, 2)

def generate_suggestions(resume_text, job_desc, metrics):
    suggestions = []
    word_count = len(resume_text.split())
    
    if metrics["keyword_match_rate"] < 70:
        suggestions.append("Increase keyword relevance by incorporating more specific terms from the job description.")
    if metrics["format_score"] < 70:
        if word_count < 150:
            suggestions.append("Expand your resume content; it’s too short (less than 150 words).")
        elif word_count > 1000:
            suggestions.append("Shorten your resume; it’s too long (over 1000 words).")
        if resume_text.count('•') < 3:
            suggestions.append("Add more bullet points to improve readability and highlight achievements.")
        if metrics["pages"] > 2:
            suggestions.append("Condense your resume to 1-2 pages for better ATS compatibility.")
    if metrics["content_quality"] < 70:
        if textstat.flesch_reading_ease(resume_text) < 60:
            suggestions.append("Simplify your language to improve readability (aim for shorter sentences and common words).")
        if TextBlob(resume_text).sentiment.polarity < 0:
            suggestions.append("Use more positive language to enhance the tone of your resume.")
    if metrics["education_metrics"] < 50:
        suggestions.append("Add or clarify your education details (e.g., degree, institution, graduation year).")
    if metrics["qualification_score"] < 70:
        suggestions.append("Highlight more relevant skills or experiences that match the job description.")
    if metrics["career_progress"] < 50:
        suggestions.append("Detail your career progression more clearly (e.g., job titles, dates, responsibilities).")
    
    return suggestions if suggestions else ["Your resume is well-optimized, but consider minor tweaks for perfection."]

@app.post("/analyze", response_model=ATSResponse)
async def analyze_resume(
    resume: UploadFile = File(...),
    job_description: str = Form(...)
):
    if not resume.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Please upload a PDF file")

    resume_text, metadata = extract_text_from_pdf(await resume.read())
    if not resume_text:
        raise HTTPException(status_code=400, detail="Failed to extract text from resume")

    keyword_match_rate = calculate_keyword_match(resume_text, job_description)
    format_score = analyze_format(resume_text, metadata)
    content_quality = analyze_content_quality(resume_text)
    education_metrics = analyze_education(resume_text)
    qualification_score = analyze_qualifications(resume_text, job_description)
    career_progress = analyze_career_progress(resume_text)
    
    ats_score = round((
        keyword_match_rate * 0.3 +
        format_score * 0.2 +
        content_quality * 0.15 +
        education_metrics * 0.15 +
        qualification_score * 0.15 +
        career_progress * 0.05
    ), 2)

    analysis = (
        f"Overall ATS Score: {ats_score}%\n"
        f"Keyword Match: {keyword_match_rate}%\n"
        f"Format Score: {format_score}%\n"
        f"Content Quality: {content_quality}%\n"
        f"Education Metrics: {education_metrics}%\n"
        f"Qualification Score: {qualification_score}%\n"
        f"Career Progress: {career_progress}%"
    )

    metrics = {
        "keyword_match_rate": keyword_match_rate,
        "format_score": format_score,
        "content_quality": content_quality,
        "education_metrics": education_metrics,
        "qualification_score": qualification_score,
        "career_progress": career_progress,
        "pages": metadata["pages"]
    }
    suggestions = generate_suggestions(resume_text, job_description, metrics)

    return ATSResponse(
        ats_score=ats_score,
        keyword_match_rate=keyword_match_rate,
        format_score=format_score,
        content_quality=content_quality,
        education_metrics=education_metrics,
        qualification_score=qualification_score,
        career_progress=career_progress,
        analysis=analysis,
        suggestions=suggestions
    )

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)