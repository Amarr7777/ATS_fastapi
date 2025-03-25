from fastapi import FastAPI, UploadFile, Form, HTTPException, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import fitz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from textblob import TextBlob
import textstat
import nltk
from nltk.corpus import stopwords
import logging

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
STOPWORDS = set(stopwords.words('english'))

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
    action_verb_strength: float
    achievement_density: float
    section_completeness: Dict[str, float]
    specificity_score: float
    readability_complexity: float
    analysis: str
    suggestions: List[str]
    keywords: List[str]

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
    documents = [resume_text, job_desc]
    vectorizer = TfidfVectorizer(stop_words='english', max_features=100, token_pattern=r'\b[a-zA-Z]{2,}\b')
    tfidf_matrix = vectorizer.fit_transform(documents)
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()
    resume_scores = tfidf_scores[0]
    job_scores = tfidf_scores[1]
    
    keyword_scores = [(word, score * (1 + job_scores[i])) 
                      for i, (word, score) in enumerate(zip(feature_names, resume_scores))
                      if not re.match(r'^\d+$', word)]
    
    keyword_scores.sort(key=lambda x: x[1], reverse=True)
    keywords = [word for word, score in keyword_scores[:15]]
    
    logger.debug(f"Top Keywords: {keywords}")
    return round(similarity * 100, 2), keywords

def analyze_format(resume_text, metadata):
    score = 100.0
    word_count = len(resume_text.split())
    if word_count < 300:
        score -= (300 - word_count) / 15
    elif word_count > 800:
        score -= (word_count - 800) / 20
    if metadata["pages"] > 2:
        score -= 15
    bullet_count = resume_text.count('•')
    expected_bullets = max(5, word_count // 100)
    if bullet_count < expected_bullets:
        score -= min(20, (expected_bullets - bullet_count) * 2)
    return max(0, min(100, round(score, 2)))

def analyze_content_quality(resume_text):
    flesch = textstat.flesch_reading_ease(resume_text)
    sentences = [s.strip() for s in resume_text.split('.') if s.strip()]
    avg_sentence_length = sum(len(s.split()) for s in sentences) / max(1, len(sentences))
    sentiment = TextBlob(resume_text).sentiment.polarity
    
    readability_score = min(100, max(0, flesch)) * 0.8
    sentence_score = max(0, 100 - abs(avg_sentence_length - 12) * 5) * 0.1
    sentiment_score = (sentiment + 1) * 5
    
    return round(readability_score + sentence_score + sentiment_score, 2)

def analyze_education(resume_text):
    resume_lower = resume_text.lower()
    keywords = ['bachelor', 'master', 'phd', 'degree', 'university', 'college', 'school']
    dates = re.search(r'\d{4}\s*-\s*\d{4}|\d{4}\s*grad|\d{4}', resume_lower)
    lines = [l for l in resume_text.split('\n') if any(k in l.lower() for k in keywords)]
    
    score = 0
    if lines:
        score += 30
        score += min(40, len(lines) * 10)
        if dates:
            score += 30
    return min(100, round(score, 2))

def analyze_qualifications(resume_text, job_desc):
    job_blob = TextBlob(job_desc)
    job_terms = set([w.lower() for w, t in job_blob.tags if t in ('NN', 'NNS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ')])
    resume_lower = resume_text.lower()
    matches = sum(1 for term in job_terms if term in resume_lower)
    return round(min(100, (matches / max(1, len(job_terms))) * 100), 2)

def analyze_career_progress(resume_text):
    resume_lower = resume_text.lower()
    titles = re.findall(r'\b(manager|senior|lead|director|specialist|engineer|developer|analyst|teacher|coordinator|executive|assistant|associate)\b', resume_lower)
    date_pairs = re.findall(r'(\d{4})\s*-\s*(\d{4}|\bpresent\b)', resume_lower)
    years = sum(abs(int(end if end != 'present' else 2025) - int(start)) for start, end in date_pairs)
    
    score = min(50, len(set(titles)) * 10) + min(50, years * 2)
    return round(score, 2)

def analyze_action_verbs(resume_text):
    job_verbs = ['lead', 'manage', 'develop', 'create', 'improve', 'execute', 'design', 'deliver', 'build', 'teach', 'coordinate', 'analyze']
    weak_verbs = ['do', 'work', 'be', 'have', 'make', 'assist', 'help']
    words = re.findall(r'\b\w+\b', resume_text.lower())
    
    strong_count = sum(1 for w in words if any(v in w for v in job_verbs))
    weak_count = sum(1 for w in words if any(v in w for v in weak_verbs))
    total = strong_count + weak_count
    return round((strong_count / total) * 100, 2) if total > 0 else 50

def analyze_achievement_density(resume_text):
    achievements = re.findall(
        r'\b(increased|decreased|improved|reduced|saved|generated|boosted|achieved|delivered|enhanced|grew|optimized|doubled|tripled)\s+.*?(\d+%|\$\d+|\d+\s*(?:percent|dollars|users|clients|savings|revenue|points|times))',
        resume_text.lower()
    ) or re.findall(r'\b\d+\s*(?:users|clients|projects|systems|applications|students|patients|sales)', resume_text.lower())
    word_count = len(resume_text.split())
    density = len(achievements) / max(1, word_count / 200)
    return round(min(100, density * 20), 2)

def analyze_section_completeness(resume_text):
    sections = {"Experience": 0, "Education": 0, "Skills": 0}
    lines = resume_text.split('\n')
    
    exp_keywords = ['experience', 'work', 'employment', 'history', 'job', 'position', 'role', 'internship', 'project']
    exp_lines = [l for l in lines if any(k in l.lower() for k in exp_keywords) or re.search(r'\d{4}\s*-\s*\d{4}', l)]
    sections["Experience"] = min(100, len(exp_lines) * 10) if exp_lines and sum(len(l.split()) for l in exp_lines) > 50 else 30
    
    edu_keywords = ['education', 'degree', 'university', 'college', 'school', 'training', 'certification']
    edu_lines = [l for l in lines if any(k in l.lower() for k in edu_keywords) or re.search(r'\d{4}\s*grad|\d{4}', l)]
    sections["Education"] = min(100, len(edu_lines) * 15) if edu_lines and len(edu_lines) > 1 else 30
    
    skill_keywords = ['skills', 'abilities', 'competencies', 'proficiency', 'expertise', 'tools', 'technologies']
    skill_lines = [l for l in lines if any(k in l.lower() for k in skill_keywords) or ',' in l]
    sections["Skills"] = min(100, len(skill_lines) * 20) if skill_lines and len(skill_lines) > 2 else 30
    
    return sections

def analyze_specificity(resume_text, job_desc):
    logger.debug("Entering analyze_specificity")
    logger.debug(f"Raw Resume Text (first 100 chars): {resume_text[:100]}")
    logger.debug(f"Raw Job Desc (first 100 chars): {job_desc[:100]}")
    
    job_blob = TextBlob(job_desc)
    job_words = set([w.lower() for w, t in job_blob.tags if t.startswith(('NN', 'JJ', 'VB'))])
    generic_terms = {'good', 'great', 'excellent', 'team', 'player', 'hard', 'worker', 'strong', 'best', 'nice'}
    
    resume_blob = TextBlob(resume_text)
    resume_words = set([w.lower() for w, t in resume_blob.tags if t.startswith(('NN', 'JJ', 'VB'))])
    
    logger.debug(f"Job Words (all): {sorted(list(job_words))}")
    logger.debug(f"Resume Words (all): {sorted(list(resume_words))}")
    
    if not resume_words or not job_words:
        logger.warning("No significant terms detected in resume or job description")
        return 60
    
    specific_overlap = len(resume_words & job_words - generic_terms)
    total_resume_terms = len(resume_words - generic_terms)
    generic_count = sum(resume_text.lower().count(term) for term in generic_terms)
    word_count = len(resume_text.split())
    generic_penalty = min(10, (generic_count / max(1, word_count)) * 50)
    
    score = (specific_overlap / max(1, total_resume_terms)) * 100 - generic_penalty
    final_score = round(max(60, min(100, score)), 2)
    logger.debug(f"Specific Overlap: {specific_overlap}, Total Terms: {total_resume_terms}, Penalty: {generic_penalty}, Final Score: {final_score}")
    return final_score

def analyze_readability_complexity(resume_text):
    logger.debug("Entering analyze_readability_complexity")
    logger.debug(f"Raw Resume Text (first 100 chars): {resume_text[:100]}")
    
    flesch = textstat.flesch_reading_ease(resume_text)
    gunning_fog = textstat.gunning_fog(resume_text)
    sentences = [s.strip() for s in resume_text.split('.') if s.strip()]
    avg_sentence_length = sum(len(s.split()) for s in sentences) / max(1, len(sentences)) if sentences else 10
    words = re.findall(r'\b\w+\b', resume_text.lower())
    word_freq = nltk.FreqDist(words)
    repetition_density = len([w for w, f in word_freq.items() if f > 3]) / max(1, len(words)) * 100 if words else 0
    
    readability_base = max(30, flesch + 40) * 0.8
    complexity_penalty = min(5, max(0, (gunning_fog - 8) * 1))
    sentence_penalty = min(5, max(0, abs(avg_sentence_length - 12) * 0.5))
    repetition_penalty = min(5, repetition_density * 0.5)
    
    score = readability_base - complexity_penalty - sentence_penalty - repetition_penalty
    final_score = round(max(50, min(100, score)), 2)
    logger.debug(f"Flesch: {flesch}, Gunning: {gunning_fog}, Avg Sentence: {avg_sentence_length}, Repetition: {repetition_density}, Final Score: {final_score}")
    return final_score

def generate_suggestions(resume_text, job_desc, metrics):
    suggestions = []
    word_count = len(resume_text.split())
    
    if metrics["keyword_match_rate"] < 70:
        suggestions.append("Add more key terms from the job description.")
    if metrics["format_score"] < 70:
        if word_count < 300:
            suggestions.append("Expand content (aim for 300-800 words).")
        elif word_count > 800:
            suggestions.append("Shorten resume (aim for 300-800 words).")
        if resume_text.count('•') < word_count // 100:
            suggestions.append("Increase bullet points for readability.")
        if metrics["pages"] > 2:
            suggestions.append("Reduce to 1-2 pages.")
    if metrics["content_quality"] < 70:
        if textstat.flesch_reading_ease(resume_text) < 60:
            suggestions.append("Simplify language (aim for Flesch 60-70).")
    if metrics["education_metrics"] < 50:
        suggestions.append("Detail education (e.g., degree, dates).")
    if metrics["qualification_score"] < 70:
        suggestions.append("Highlight job-relevant skills/experience.")
    if metrics["career_progress"] < 50:
        suggestions.append("Show progression with titles and dates.")
    if metrics["action_verb_strength"] < 70:
        suggestions.append("Use stronger verbs (e.g., 'led', 'designed').")
    if metrics["achievement_density"] < 50:
        suggestions.append("Add quantifiable results (e.g., 'grew sales 20%').")
    if any(score < 70 for score in metrics["section_completeness"].values()):
        incomplete = [k for k, v in metrics["section_completeness"].items() if v < 70]
        suggestions.append(f"Complete sections: {', '.join(incomplete)}.")
    if metrics["specificity_score"] < 70:
        suggestions.append("Tailor content with job-specific terms.")
    if metrics["readability_complexity"] < 70:
        suggestions.append("Shorten sentences and vary word choice.")
    
    return suggestions if suggestions else ["Resume is solid; refine for perfection."]

@app.post("/analyze", response_model=ATSResponse)
async def analyze_resume(
    resume: UploadFile = File(...),
    job_description: str = Form(...)
):
    logger.debug("Starting analyze_resume")
    if not resume.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Please upload a PDF file")

    resume_text, metadata = extract_text_from_pdf(await resume.read())
    logger.debug(f"Resume Text (first 100 chars): {resume_text[:100]}")
    logger.debug(f"Job Description (first 100 chars): {job_description[:100]}")
    
    if not resume_text:
        raise HTTPException(status_code=400, detail="Failed to extract text from resume")

    keyword_match_rate, keywords = calculate_keyword_match(resume_text, job_description)
    format_score = analyze_format(resume_text, metadata)
    content_quality = analyze_content_quality(resume_text)
    education_metrics = analyze_education(resume_text)
    qualification_score = analyze_qualifications(resume_text, job_description)
    career_progress = analyze_career_progress(resume_text)
    action_verb_strength = analyze_action_verbs(resume_text)
    achievement_density = analyze_achievement_density(resume_text)
    section_completeness = analyze_section_completeness(resume_text)
    specificity_score = analyze_specificity(resume_text, job_description)
    readability_complexity = analyze_readability_complexity(resume_text)
    
    logger.debug(f"Specificity Score: {specificity_score}")
    logger.debug(f"Readability Complexity: {readability_complexity}")
    
    ats_score = round((
        keyword_match_rate * 0.20 +
        format_score * 0.15 +
        content_quality * 0.15 +
        education_metrics * 0.10 +
        qualification_score * 0.15 +
        career_progress * 0.05 +
        action_verb_strength * 0.10 +
        achievement_density * 0.05 +
        specificity_score * 0.05 +
        readability_complexity * 0.05
    ), 2)

    analysis = (
        f"Overall ATS Score: {ats_score}%\n"
        f"Keyword Match: {keyword_match_rate}%\n"
        f"Format Score: {format_score}%\n"
        f"Content Quality: {content_quality}%\n"
        f"Education Metrics: {education_metrics}%\n"
        f"Qualification Score: {qualification_score}%\n"
        f"Career Progress: {career_progress}%\n"
        f"Action Verb Strength: {action_verb_strength}%\n"
        f"Achievement Density: {achievement_density}%\n"
        f"Specificity Score: {specificity_score}%\n"
        f"Readability Complexity: {readability_complexity}%\n"
        f"Section Completeness: {', '.join([f'{k}: {v}%' for k, v in section_completeness.items()])}"
    )

    metrics = {
        "keyword_match_rate": keyword_match_rate,
        "format_score": format_score,
        "content_quality": content_quality,
        "education_metrics": education_metrics,
        "qualification_score": qualification_score,
        "career_progress": career_progress,
        "action_verb_strength": action_verb_strength,
        "achievement_density": achievement_density,
        "section_completeness": section_completeness,
        "specificity_score": specificity_score,
        "readability_complexity": readability_complexity,
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
        action_verb_strength=action_verb_strength,
        achievement_density=achievement_density,
        section_completeness=section_completeness,
        specificity_score=specificity_score,
        readability_complexity=readability_complexity,
        analysis=analysis,
        suggestions=suggestions,
        keywords=keywords
    )

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)