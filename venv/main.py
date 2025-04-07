from fastapi import FastAPI, UploadFile, Form, HTTPException, File, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import fitz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from textblob import TextBlob
import textstat
import nltk
from nltk.corpus import stopwords
import logging
import json
import requests
from starlette_session import SessionMiddleware
from datetime import timedelta
import uuid
import time


SESSION_DATA = {} 

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
STOPWORDS = set(stopwords.words('english'))

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# Define your Pydantic model
class AnswerSubmission(BaseModel):
    answer: Optional[str] = ""
    audio: Optional[str] = None
    transcription: Optional[str] = ""
    question: Optional[dict] = None
    frames: Optional[List[str]] = []

# Add session middleware
app.add_middleware(
    SessionMiddleware,
    secret_key="a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",
    cookie_name="session",
    max_age=86400
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Groq API Key
GROQ_API_KEY = "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"

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
    resume_text: str

class AnswerSubmission(BaseModel):
    audio: Optional[str] = None
    transcription: Optional[str] = None
    answer: Optional[str] = None
    frames: Optional[List[str]] = None
    question: Optional[str] = None

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
def generate_interview_questions(resume_text: str, jd_text: str = "") -> List[Dict]:
    """Generate personalized questions based on resume text using Groq LLM"""
    # Truncate resume and JD text to reduce token usage
    max_resume_length = 2000
    max_jd_length = 1000
    
    if len(resume_text) > max_resume_length:
        resume_text = resume_text[:max_resume_length] + "... [truncated]"
    
    if jd_text and len(jd_text) > max_jd_length:
        jd_text = jd_text[:max_jd_length] + "... [truncated]"
    
    prompt = ""
    if jd_text:
        prompt = f"""
        Generate 5 interview questions based on this resume and job description:
        
        RESUME EXCERPT:
        {resume_text}
        
        JOB DESCRIPTION EXCERPT:
        {jd_text}
        
        Rules:
        1. If resume has projects/internships, create questions based on them
        2. If no projects/internships, focus on technical skills and job description
        3. First 3 questions: moderate difficulty
        4. Last 2 questions: hard difficulty
        5. At least 2 questions should be technical based on resume skills
        
        Format as JSON array with fields: question, difficulty, category, expected_answer
        """
    else:
        prompt = f"""
        Based on the following resume, create 5 interview questions:
        - 2 project-based questions
        - 2 internship-based questions
        - 1 technical skill question
        
        Among these, 2 questions should be challenging. Format the response as a JSON array with 
        objects containing 'question', 'type' (project/internship/technical), 'difficulty' (easy/hard), and 'expected_answer' field.
        
        Resume: {resume_text}
        """
    
    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "messages": [
                {"role": "system", "content": "You are an expert technical interviewer. You always respond with well-formatted JSON arrays containing question objects."},
                {"role": "user", "content": prompt}
            ],
            "model": "llama3-70b-8192",
            "temperature": 0.7,
            "max_tokens": 2000,
            "top_p": 1,
            "stream": False,
            "stop": None
        }
        
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            try:
                questions = json.loads(content)
                if len(questions) < 5:
                    logger.warning(f"Only got {len(questions)} questions, adding generic ones")
                    generic_questions = generate_fallback_questions(resume_text)
                    questions.extend(generic_questions[len(questions):5])
                return questions
            except json.JSONDecodeError:
                logger.error("Failed to parse JSON directly, trying to extract JSON")
                json_match = re.search(r'\[\s*\{.*\}\s*\]', content, re.DOTALL)
                if json_match:
                    questions = json.loads(json_match.group(0))
                    if len(questions) < 5:
                        generic_questions = generate_fallback_questions(resume_text)
                        questions.extend(generic_questions[len(questions):5])
                    return questions
                else:
                    logger.error("No JSON structure found in response")
                    return generate_fallback_questions(resume_text)
        else:
            logger.error(f"Error generating questions: {response.status_code}, {response.text}")
            return generate_fallback_questions(resume_text)
    except Exception as e:
        logger.error(f"Error generating questions: {e}")
        return generate_fallback_questions(resume_text)

def generate_fallback_questions(resume_text: str) -> List[Dict]:
    """Generate fallback questions when the API call fails"""
    skills = []
    resume_lower = resume_text.lower()
    common_skills = ["python", "javascript", "java", "c++", "react", "node.js", "sql", 
                    "machine learning", "data analysis", "cloud", "aws", "azure", 
                    "docker", "kubernetes", "agile", "scrum", "project management"]
    
    for skill in common_skills:
        if skill in resume_lower:
            skills.append(skill)
    
    if not skills:
        skills = ["programming", "problem solving", "teamwork"]
    
    questions = [
        {
            "question": f"Can you explain your experience with {skills[0] if skills else 'your technical skills'}?",
            "difficulty": "Moderate",
            "category": "Technical",
            "expected_answer": f"Looking for detailed explanation of {skills[0] if skills else 'technical skills'} with examples"
        },
        {
            "question": "Describe a challenging project you worked on and how you overcame obstacles.",
            "difficulty": "Moderate",
            "category": "Project-based",
            "expected_answer": "Should mention specific project, challenges faced, and solutions implemented"
        },
        {
            "question": f"How do you stay updated with the latest developments in {skills[1] if len(skills) > 1 else 'your field'}?",
            "difficulty": "Moderate",
            "category": "General",
            "expected_answer": "Looking for specific learning resources, communities, or practices"
        },
        {
            "question": f"Explain a complex concept in {skills[0] if skills else 'your field'} as if you were teaching it to a beginner.",
            "difficulty": "Hard",
            "category": "Technical",
            "expected_answer": "Should break down complex topic clearly and accurately"
        },
        {
            "question": "If you were given a task with unclear requirements, how would you approach it?",
            "difficulty": "Hard",
            "category": "General",
            "expected_answer": "Looking for problem-solving approach, communication skills, and initiative"
        }
    ]
    
    return questions

def evaluate_answer(question, resume_text, answer):
    """Evaluate user's answer using LLM"""
    try:
        # Check if question is a string or a dict
        question_text = question
        question_info = {"difficulty": "Unknown", "category": "Unknown", "expected_answer": "Unknown"}
        
        if isinstance(question, dict):
            question_text = question.get('question', '')
            question_info = {
                "difficulty": question.get('difficulty', 'Unknown'),
                "category": question.get('category', 'Unknown') if 'category' in question else question.get('type', 'Unknown'),
                "expected_answer": question.get('expected_answer', 'Unknown')
            }
        
        # Debug logs
        print("Question: ", question_text[:100])
        print("Question type: ", type(question))
        print("Answer length: ", len(answer))
        
        prompt = f"""
        You are a technical interviewer. Evaluate this answer based on accuracy, completeness, and clarity.
        
        Question: {question_text}
        Question Difficulty: {question_info['difficulty']}
        Question Category: {question_info['category']}
        Expected Answer Points: {question_info['expected_answer']}
        Resume Information: {resume_text[:500] if resume_text else ''}... [truncated]
        Answer: {answer}
        
        Evaluate the answer on a scale of 0-100 based on:
        - Technical accuracy (70%)
        - Clarity and coherence (20%)
        - Grammar, fluency, and confidence (10%)
        
        Provide a score out of 100 and detailed feedback including strengths and areas for improvement.
        Format your response as a JSON object with 'score' (number) and 'feedback' (string) keys.
        
        EXAMPLE RESPONSE FORMAT:
        {{"score": 85, "feedback": "Your answer was comprehensive and technically accurate..."}}
        """
        
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "messages": [
                {"role": "system", "content": "You are an expert technical interviewer who evaluates answers objectively. You always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            "model": "llama3-70b-8192",
            "temperature": 0.3,
            "max_tokens": 1000
        }
        
        print("Sending evaluation request to API...")
        
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        print(f"API response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            print(f"Raw API content: {content[:100]}...")
            
            try:
                # Try to parse the entire content as JSON
                evaluation = json.loads(content)
                print("Successfully parsed JSON response")
                
                # Validate the evaluation object has the required fields
                if 'score' not in evaluation or 'feedback' not in evaluation:
                    print("Missing required fields in evaluation")
                    # Create proper structure if missing
                    score = evaluation.get('score', 50)
                    feedback = evaluation.get('feedback', 'No detailed feedback available')
                    evaluation = {'score': int(score), 'feedback': feedback}
                
                # Ensure score is an integer
                if not isinstance(evaluation['score'], int):
                    try:
                        evaluation['score'] = int(float(evaluation['score']))
                    except (ValueError, TypeError):
                        evaluation['score'] = 50
                        
                print(f"Final evaluation: score={evaluation['score']}, feedback={evaluation['feedback'][:50]}...")
                return evaluation
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                # If that fails, try to extract JSON from the text
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    try:
                        json_str = json_match.group(0)
                        print(f"Extracted JSON string: {json_str[:100]}...")
                        evaluation = json.loads(json_str)
                        
                        # Validate the evaluation object has the required fields
                        if 'score' not in evaluation or 'feedback' not in evaluation:
                            score = evaluation.get('score', 50)
                            feedback = evaluation.get('feedback', 'No detailed feedback available')
                            evaluation = {'score': int(score), 'feedback': feedback}
                        
                        # Ensure score is an integer
                        if not isinstance(evaluation['score'], int):
                            try:
                                evaluation['score'] = int(float(evaluation['score']))
                            except (ValueError, TypeError):
                                evaluation['score'] = 50
                                
                        print(f"Extracted evaluation: score={evaluation['score']}, feedback={evaluation['feedback'][:50]}...")
                        return evaluation
                    except Exception as e:
                        print(f"Error parsing extracted JSON: {e}")
                        # Parse failure, use fallback with text extraction
                        
                        # Try to extract score using regex
                        score_match = re.search(r'score["\s:]+(\d+)', content, re.IGNORECASE)
                        score = int(score_match.group(1)) if score_match else 50
                        
                        # Use the content as feedback
                        return {
                            "score": score,
                            "feedback": content.strip()
                        }
                else:
                    print("No JSON pattern found in response")
                    # Fallback
                    return {
                        "score": 50,
                        "feedback": content.strip() if content else "Could not parse evaluation. The answer appears to be of average quality."
                    }
        else:
            print(f"Error evaluating answer: {response.status_code}, {response.text}")
            return generate_fallback_evaluation(question, answer)
    except Exception as e:
        print(f"Error evaluating answer: {str(e)}")
        import traceback
        traceback.print_exc()
        return generate_fallback_evaluation(question, answer)

def generate_fallback_evaluation(question, answer):
    """Generate a fallback evaluation when the API call fails"""
    
    # Simple keyword-based scoring
    score = 50  # Default average score
    
    # Check if answer is empty or very short
    if not answer or len(answer.split()) < 5:
        return {
            "score": 20,
            "feedback": "Your answer was too brief. Please provide more detailed responses to demonstrate your knowledge."
        }
    
    # Check for keywords from the expected answer
    expected_keywords = []
    if isinstance(question, dict) and 'expected_answer' in question:
        expected_answer = question['expected_answer'].lower()
        # Extract potential keywords (words longer than 4 characters)
        expected_keywords = [word for word in expected_answer.split() if len(word) > 4]
    
    # Count matching keywords
    keyword_matches = 0
    answer_lower = answer.lower()
    for keyword in expected_keywords:
        if keyword in answer_lower:
            keyword_matches += 1
    
    # Adjust score based on keyword matches
    if expected_keywords:
        keyword_score = min(70, (keyword_matches / len(expected_keywords)) * 100)
        score = (score + keyword_score) / 2
    
    # Adjust score based on answer length (up to a point)
    word_count = len(answer.split())
    if word_count < 20:
        score = max(20, score - 20)  # Penalize very short answers
    elif word_count > 50:
        score = min(90, score + 10)  # Reward substantial answers, but cap at 90
    
    # Generate feedback based on score
    if score >= 80:
        feedback = "Excellent answer! You demonstrated good understanding of the topic and provided a clear explanation."
    elif score >= 60:
        feedback = "Good answer. You covered some key points, but there's room to expand on certain aspects."
    elif score >= 40:
        feedback = "Average answer. You touched on the topic, but could provide more depth and technical details."
    else:
        feedback = "Your answer needs improvement. Try to be more specific and demonstrate your technical knowledge."
    
    return {
        "score": round(score),
        "feedback": feedback
    }

@app.post("/analyze", response_model=ATSResponse)
async def analyze_resume(
    request: Request,
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

    # Generate personalized interview questions
    questions = generate_interview_questions(resume_text, job_description)

    # Store in session
    # request.session["resume_text"] = resume_text
    # request.session["questions"] = questions
    # request.session["current_question"] = 0
    # request.session["results"] = []
    # Generate a unique session ID if not exists
    session_id = request.session.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        request.session["session_id"] = session_id
    
    # Store data in server-side cache
    SESSION_DATA[session_id] = {
        "resume_text": resume_text,
        "questions": questions,
        "current_question": 0,
        "results": [],
        "timestamp": time.time()  # For cleanup later
    }

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
        keywords=keywords,
        resume_text=resume_text,
    )
@app.post("/interview")
async def interview(request: Request):
    session_id = request.session.get("session_id")
    if not session_id or session_id not in SESSION_DATA:
        return JSONResponse(
            status_code=400,
            content={"error": "No resume uploaded. Please start over."}
        )
    
    session_data = SESSION_DATA[session_id]
    questions = session_data.get('questions')
    current_question_index = session_data.get('current_question', 0)
    
    logger.debug(f"Questions type: {type(questions)}")
    logger.debug(f"Questions data: {questions}")
    logger.debug(f"Current question index: {current_question_index}")
    
    if isinstance(questions, list):
        if current_question_index >= len(questions):
            return {
                "completed": True,
                "scores": request.session.get('results', []),
                "total_score": request.session.get('total_score', 0)
            }
        
        return {
            "questions": questions,
            "question_number": current_question_index + 1,
            "total_questions": len(questions)
        }
    elif isinstance(questions, str):
        try:
            parsed_questions = json.loads(questions)
            if isinstance(parsed_questions, list) and len(parsed_questions) > 0:
                request.session['questions'] = parsed_questions
                return {
                    "questions": parsed_questions,
                    "question_number": current_question_index + 1,
                    "total_questions": len(parsed_questions)
                }
        except:
            pass
    
    return {"questions": questions}

@app.post("/submit_answer")
async def submit_answer(
    data: AnswerSubmission,
    request: Request
):
    try:
        session_id = request.session.get("session_id")
        # Check if session has questions
        if not session_id or session_id not in SESSION_DATA:
            raise HTTPException(status_code=400, detail="No interview in progress")
        
        logger.debug(f"Received answer submission data: {str(data)[:200]}...")
        
        # Process answer
        answer = data.transcription if data.audio else data.answer
        
        # Cheating detection is disabled
        cheating_detected = False
        cheating_reason = ""
        
        # Get question data
        session_data = SESSION_DATA[session_id]
        questions = session_data.get('questions')
        current_question_index = session_data.get('current_question', 0)
        
        # Ensure questions is a list
        if isinstance(questions, str):
            try:
                questions = json.loads(questions)
                request.session['questions'] = questions
            except Exception as e:
                logger.error(f"Failed to parse questions: {e}")
                questions = [{"question": questions, "type": "general", "difficulty": "medium"}]
                request.session['questions'] = questions
        
        if not isinstance(questions, list):
            questions = [questions] if questions else [{"question": "General question", "type": "general", "difficulty": "medium"}]
            request.session['questions'] = questions
            
        # Get current question
        if current_question_index < len(questions):
            current_question = questions[current_question_index]
        else:
            current_question = data.question if data.question else {"question": "General question", "type": "general", "difficulty": "medium"}
        
        # Make sure evaluate_answer is async-compatible
        # If it's not, use run_in_threadpool
        from starlette.concurrency import run_in_threadpool
        
        # Check if evaluate_answer is async
        if not hasattr(evaluate_answer, "__await__"):
            evaluation = await run_in_threadpool(
                evaluate_answer,
                current_question,
                request.session.get('resume_text', ''),
                answer
            )
        else:
            evaluation = await evaluate_answer(
                current_question,
                request.session.get('resume_text', ''),
                answer
            )
        
        # Ensure evaluation format
        if isinstance(evaluation, str):
            try:
                evaluation = json.loads(evaluation)
            except:
                evaluation = {"score": 50, "feedback": evaluation}
        
        if not isinstance(evaluation, dict):
            evaluation = {"score": 50, "feedback": str(evaluation)}
        
        evaluation.setdefault("score", 50)
        evaluation.setdefault("feedback", "No feedback provided")
        
        try:
            evaluation["score"] = int(float(evaluation["score"]))
        except (ValueError, TypeError):
            evaluation["score"] = 50
        
        # Store result
        result_entry = {
            "question": current_question.get('question') if isinstance(current_question, dict) else str(current_question),
            "answer": answer,
            "score": evaluation["score"],
            "feedback": evaluation["feedback"]
        }
        
        # results = request.session.get('results', [])
        # results.append(result_entry)
        # request.session['results'] = results
        session_data["results"] = session_data.get("results", []) + [result_entry]
        session_data["current_question"] = current_question_index + 1
        SESSION_DATA[session_id] = session_data
        
        is_completed = session_data["current_question"] >= len(questions)
        total_score = sum(item["score"] for item in session_data["results"]) / len(session_data["results"]) if session_data["results"] else 0
        
        # Next question
        request.session['current_question'] = current_question_index + 1
        is_completed = request.session['current_question'] >= len(questions)
        
        response_data = {
        "success": True,
        "evaluation": evaluation,
        }
    
        if is_completed:
            response_data.update({
            "message": "Interview completed",
            "redirect": "/results",
            "total_score": total_score
            })
        else:
            response_data["next_question"] = session_data["current_question"] + 1
    
        return response_data
        
    except Exception as e:
        logger.error(f"Error processing answer: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            "success": True,
            "evaluation": {
                "score": 50,
                "feedback": "Error occurred, but answer recorded"
            }
        }

@app.get("/results")
async def results(request: Request):
    session_id = request.session.get("session_id")
        # Check if session has questions
    if not session_id or session_id not in SESSION_DATA:
        raise HTTPException(status_code=400, detail="No interview results found")
    
    session_data = SESSION_DATA[session_id]
    results = session_data.get("results", [])
    total_score = sum(item["score"] for item in results) / len(results) if results else 0
    
    return {
        "scores": results,
        "total_score": total_score
    }


@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)


