import PyPDF2
import re
import string
import pandas as pd
import colorama
from colorama import Fore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from difflib import SequenceMatcher
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import Callable, Optional, Dict, List, Any
from hashlib import md5

tokenizer = None
embedding_model = None
_embedding_loaded = False
RESUME_TEXT_CACHE: Dict[str, Dict[str, Any]] = {}
RESUME_EMBEDDING_CACHE: Dict[str, torch.Tensor] = {}
JD_EMBEDDING_CACHE: Dict[str, torch.Tensor] = {}

TOP_K_FOR_EMBEDDING = 10

SYNONYM_MAP = {
    "developer": ["engineer", "programmer"],
    "rest": ["api", "restful"],
    "database": ["db", "sql"],
    "ml": ["machine learning"],
}

SYNONYM_LOOKUP = {}
for base_term, variants in SYNONYM_MAP.items():
    SYNONYM_LOOKUP.setdefault(base_term, set()).update(variants)
    for variant in variants:
        SYNONYM_LOOKUP.setdefault(variant, set()).add(base_term)
        SYNONYM_LOOKUP[variant].update(variants)

IMPORTANT_TECH_TERMS = set(
    [
        "python", "java", "c++", "csharp", "javascript", "typescript", "go", "rust", "php",
        "ruby", "scala", "kotlin", "rest", "graphql", "fastapi", "flask", "django", "spring",
        "express", "react", "vue", "angular", "api", "sql", "postgresql", "mysql", "mongodb",
        "redis", "elasticsearch", "cassandra", "database", "algorithm", "sorting", "searching",
        "tree", "graph", "hash", "datastructure", "machine learning", "ml"
    ]
)
for base_term, variants in SYNONYM_MAP.items():
    IMPORTANT_TECH_TERMS.add(base_term)
    IMPORTANT_TECH_TERMS.update(variants)

STREAM_FOCUS_TERMS = {
    "ENGINEERING": {
        "python", "java", "c++", "csharp", "javascript", "typescript", "go", "rust", "kotlin",
        "rest", "graphql", "fastapi", "flask", "django", "spring", "express", "react", "vue",
        "angular", "api", "sql", "postgresql", "mysql", "mongodb", "redis", "elasticsearch"
    },
    "DATA SCIENCE": {
        "python", "sql", "machine learning", "ml", "statistics", "algorithm", "model",
        "regression", "classification", "analysis", "data"
    },
    "ACCOUNTING": {
        "accounting", "finance", "audit", "tax", "ledger", "reconciliation", "balance sheet",
        "financial reporting", "payroll", "excel"
    }
}


def load_embedding_model():
    global tokenizer, embedding_model, _embedding_loaded
    if _embedding_loaded:
        return
    try:
        tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
        embedding_model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5")
        embedding_model.eval()
        _embedding_loaded = True
    except Exception:
        tokenizer = None
        embedding_model = None
        _embedding_loaded = False


def compute_embedding_similarity(text1, text2):
    global embedding_model, tokenizer
    load_embedding_model()
    if embedding_model is None or tokenizer is None:
        return None

    with torch.no_grad():
        try:
            encoded = tokenizer(
                [text1, text2],
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            outputs = embedding_model(**encoded)
            token_embeddings = outputs.last_hidden_state
            attention_mask = encoded["attention_mask"].unsqueeze(-1).float()
            summed = (token_embeddings * attention_mask).sum(dim=1)
            counts = attention_mask.sum(dim=1).clamp(min=1e-9)
            mean_pooled = summed / counts
            normalized = F.normalize(mean_pooled, p=2, dim=1)
            similarity = (normalized[0] * normalized[1]).sum().clamp(-1.0, 1.0)
            return float((similarity + 1.0) / 2.0)
        except Exception:
            print("Embedding similarity failed; using TF-IDF fallback")
            return None


def _compute_embedding_vector(text: str) -> Optional[torch.Tensor]:
    global embedding_model, tokenizer
    load_embedding_model()
    if embedding_model is None or tokenizer is None:
        return None

    with torch.no_grad():
        try:
            encoded = tokenizer(
                [text],
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            outputs = embedding_model(**encoded)
            token_embeddings = outputs.last_hidden_state
            attention_mask = encoded["attention_mask"].unsqueeze(-1).float()
            summed = (token_embeddings * attention_mask).sum(dim=1)
            counts = attention_mask.sum(dim=1).clamp(min=1e-9)
            mean_pooled = summed / counts
            normalized = F.normalize(mean_pooled, p=2, dim=1)
            return normalized[0]
        except Exception:
            print("Embedding vector failed; using TF-IDF fallback")
            return None


def _make_embedding_cache_key(file_path: str, text: str) -> str:
    text_hash = md5(text.encode("utf-8")).hexdigest()
    return f"{file_path}:{text_hash}"


def _get_cached_resume_embedding(file_path: str, text: str) -> Optional[torch.Tensor]:
    cache_key = _make_embedding_cache_key(file_path, text)
    cached = RESUME_EMBEDDING_CACHE.get(cache_key)
    if cached is not None:
        return cached

    vector = _compute_embedding_vector(text)
    if vector is not None:
        RESUME_EMBEDDING_CACHE[cache_key] = vector
    return vector


def _get_cached_jd_embedding(text: str) -> Optional[torch.Tensor]:
    cache_key = md5(text.encode("utf-8")).hexdigest()
    cached = JD_EMBEDDING_CACHE.get(cache_key)
    if cached is not None:
        return cached

    vector = _compute_embedding_vector(text)
    if vector is not None:
        JD_EMBEDDING_CACHE[cache_key] = vector
    return vector


def _extract_pdf_text(file_path: str) -> str:
    content = ""
    try:
        file = open(file_path, "rb")
        reader = PyPDF2.PdfReader(file)
        number_of_pages = len(reader.pages)
        for page_number in range(number_of_pages):
            page = reader.pages[page_number]
            text = page.extract_text()
            if text:
                content += text
        file.close()
    except Exception:
        try:
            file.close()
        except Exception:
            pass
        return ""
    return content


def _clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[0-9]+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def get_cached_resume_bundle(file_path: str) -> Dict[str, Any]:
    cached = RESUME_TEXT_CACHE.get(file_path)
    if cached:
        return cached

    raw_text = _extract_pdf_text(file_path)
    cleaned_text = _clean_text(raw_text)
    tokens = cleaned_text.split()
    token_set = set(tokens)
    cached = {
        "raw": raw_text,
        "cleaned": cleaned_text,
        "tokens": tokens,
        "token_set": token_set,
    }
    RESUME_TEXT_CACHE[file_path] = cached
    return cached

def extract_and_clean_resume(file_path):
    bundle = get_cached_resume_bundle(file_path)
    return bundle.get("cleaned", "")


def extract_resume_text(file_path: str) -> str:
    bundle = get_cached_resume_bundle(file_path)
    return bundle.get("raw", "")


def prompt_job_description():
    while True:
        job_description = input("Paste Job Description and press ENTER: ").strip()
        if job_description:
            return job_description
        print("Error: Job description cannot be empty. Please try again.")


def extract_key_technical_terms(text):
    key_terms = {
        'programming_languages': ['python', 'java', 'c++', 'csharp', 'javascript', 'typescript', 'go', 'rust', 'php', 'ruby', 'scala', 'kotlin'],
        'apis_frameworks': ['rest', 'graphql', 'fastapi', 'flask', 'django', 'spring', 'express', 'react', 'vue', 'angular', 'api'],
        'databases': ['sql', 'postgresql', 'mysql', 'mongodb', 'redis', 'elasticsearch', 'cassandra', 'database'],
        'algorithms': ['algorithm', 'sorting', 'searching', 'tree', 'graph', 'hash', 'datastructure']
    }
    text_lower = text.lower()
    found_terms = []
    for category, terms in key_terms.items():
        for term in terms:
            if term_in_text_with_synonyms(term, text_lower):
                found_terms.append(term)
    return list(set(found_terms))


def extract_stream_focus_terms(text: str, stream: Optional[str]) -> List[str]:
    if not stream:
        return []
    stream_terms = STREAM_FOCUS_TERMS.get(stream.upper())
    if not stream_terms:
        return []
    text_lower = text.lower()
    found_terms = []
    for term in stream_terms:
        if term_in_text_with_synonyms(term, text_lower) or term in text_lower:
            found_terms.append(term)
    return list(set(found_terms))


def remove_stopwords(text, technical_terms):
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'from', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can',
        'as', 'if', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
        'we', 'they', 'what', 'which', 'who', 'when', 'where', 'why', 'how', 'all',
        'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'too',
        'very', 'no', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'your'
    }
    
    words = text.split()
    technical_set = set(technical_terms)
    filtered_words = [w for w in words if w not in stopwords or w in technical_set]
    return ' '.join(filtered_words)


def term_in_text_with_synonyms(term, text_lower):
    if term in text_lower:
        return True
    for variant in SYNONYM_LOOKUP.get(term, []):
        if variant in text_lower:
            return True
    return False


def generate_ngrams(tokens, length):
    for index in range(len(tokens) - length + 1):
        yield " ".join(tokens[index:index + length])


def _build_ngram_set(tokens: List[str], max_len: int) -> set:
    if max_len <= 1:
        return set()
    ngrams = set()
    for length in range(2, max_len + 1):
        for ngram in generate_ngrams(tokens, length):
            ngrams.add(ngram)
    return ngrams


def fuzzy_match_term(term, tokens, threshold=0.85):
    term = term.lower().strip()
    if not term:
        return False
    term_parts = term.split()
    if len(term_parts) == 1:
        for token in tokens:
            if SequenceMatcher(None, term, token).ratio() >= threshold:
                return True
        return False

    for ngram in generate_ngrams(tokens, len(term_parts)):
        if SequenceMatcher(None, term, ngram).ratio() >= threshold:
            return True
    return False


def term_matches_resume(term, resume_content, resume_tokens, token_set=None, term_set=None, allow_fuzzy: bool = False):
    term = term.lower().strip()
    if not term:
        return False

    if term_set is not None:
        if term in term_set:
            return True
    elif token_set is not None and " " not in term:
        if term in token_set:
            return True
    else:
        if term in resume_content:
            return True

    for variant in SYNONYM_LOOKUP.get(term, []):
        if term_set is not None:
            if variant in term_set:
                return True
        elif token_set is not None and " " not in variant:
            if variant in token_set:
                return True
        else:
            if variant in resume_content:
                return True

    if allow_fuzzy:
        variants = [term] + list(SYNONYM_LOOKUP.get(term, []))
        for variant in variants:
            if fuzzy_match_term(variant, resume_tokens):
                return True
    return False


def term_match_detail(term, resume_content, resume_tokens, token_set=None, term_set=None, threshold=0.85, allow_fuzzy: bool = False):
    term = term.lower().strip()
    if not term:
        return False, None

    if term_set is not None:
        if term in term_set:
            return True, "exact"
    elif token_set is not None and " " not in term:
        if term in token_set:
            return True, "exact"
    else:
        if term in resume_content:
            return True, "exact"

    for variant in SYNONYM_LOOKUP.get(term, []):
        if term_set is not None:
            if variant in term_set:
                return True, "synonym"
        elif token_set is not None and " " not in variant:
            if variant in token_set:
                return True, "synonym"
        else:
            if variant in resume_content:
                return True, "synonym"

    if allow_fuzzy:
        variants = [term] + list(SYNONYM_LOOKUP.get(term, []))
        for variant in variants:
            if fuzzy_match_term(variant, resume_tokens, threshold=threshold):
                return True, "fuzzy"
    return False, None


def get_stream_term_multiplier(stream: Optional[str], term: str) -> float:
    if not stream:
        return 1.0
    stream_terms = STREAM_FOCUS_TERMS.get(stream.upper())
    if not stream_terms:
        return 1.0
    return 1.35 if term in stream_terms else 1.0


def build_evidence_map(matched_terms: List[str], resume_text: str, max_snippets: int = 2) -> Dict[str, List[str]]:
    if not matched_terms or not resume_text:
        return {}

    evidence_map: Dict[str, List[str]] = {}
    for term in matched_terms:
        snippets = []
        search_terms = [term] + list(SYNONYM_LOOKUP.get(term, []))
        for search_term in search_terms:
            escaped = re.escape(search_term)
            pattern = re.compile(r".{0,60}" + escaped + r".{0,60}", re.IGNORECASE)
            for match in pattern.finditer(resume_text):
                snippet = match.group(0).strip()
                snippet = " ".join(snippet.split())
                if snippet and snippet not in snippets:
                    snippets.append(snippet)
                if len(snippets) >= max_snippets:
                    break
            if len(snippets) >= max_snippets:
                break
        if snippets:
            evidence_map[term] = snippets
    return evidence_map


def estimate_confidence(similarity_score: float, matched_count: int, fuzzy_ratio: float) -> str:
    if similarity_score >= 0.65 and matched_count >= 4 and fuzzy_ratio <= 0.4:
        return "High"
    if similarity_score >= 0.45 and matched_count >= 2:
        return "Medium"
    return "Low"


def safe_progress_callback(progress_callback: Optional[Callable[[int], None]], percent: int) -> None:
    if not progress_callback:
        return
    try:
        clamped = max(0, min(100, int(percent)))
        progress_callback(clamped)
    except Exception:
        pass


def rank_resumes(
    job_description_text,
    resume_folder=None,
    use_embedding=True,
    max_resume_chars=8000,
    progress_callback: Optional[Callable[[int], None]] = None,
    stream: Optional[str] = None,
    mandatory_skills: Optional[List[str]] = None,
    return_details: bool = False,
):
    last_progress = 0

    def emit_progress(percent: int) -> None:
        nonlocal last_progress
        next_value = max(last_progress, percent)
        last_progress = next_value
        safe_progress_callback(progress_callback, next_value)

    emit_progress(5)
    key_terms_in_jd = extract_key_technical_terms(job_description_text)
    stream_terms = extract_stream_focus_terms(job_description_text, stream)
    mandatory_terms = []
    if mandatory_skills:
        mandatory_terms = [term.strip().lower() for term in mandatory_skills if term and term.strip()]

    skill_terms = list(set(key_terms_in_jd + stream_terms + mandatory_terms))
    skill_terms_set = set(skill_terms)
    skill_terms_with_synonyms = set(skill_terms)
    for term in skill_terms:
        skill_terms_with_synonyms.update(SYNONYM_LOOKUP.get(term, []))
    max_skill_term_len = max((len(term.split()) for term in skill_terms_with_synonyms), default=1)

    cleaned_job_desc = job_description_text.lower()
    cleaned_job_desc = re.sub(r'[0-9]+', '', cleaned_job_desc)
    cleaned_job_desc = cleaned_job_desc.translate(str.maketrans('', '', string.punctuation))
    cleaned_job_desc = remove_stopwords(cleaned_job_desc, skill_terms)

    uploads_folder = os.path.abspath(os.path.join(os.getcwd(), "uploads"))
    base_folder = os.path.abspath(resume_folder) if resume_folder else uploads_folder
    if not os.path.isdir(base_folder):
        return []

    try:
        pdf_files = [f for f in os.listdir(base_folder) if f.lower().endswith('.pdf')]
    except Exception:
        return []

    if not pdf_files:
        return []

    resume_folder = base_folder
    emit_progress(10)

    resume_contents = []
    resume_entries = []
    for pdf_file in pdf_files:
        resume_path = os.path.join(resume_folder, pdf_file)
        bundle = get_cached_resume_bundle(resume_path)
        resume_content = bundle.get("cleaned", "")
        resume_raw = bundle.get("raw", "")
        resume_tokens = bundle.get("tokens", [])
        resume_token_set = bundle.get("token_set", set())
        resume_term_set = resume_token_set | _build_ngram_set(resume_tokens, max_skill_term_len)
        resume_contents.append(resume_content[:max_resume_chars])
        resume_entries.append({
            "filename": pdf_file,
            "path": resume_path,
            "cleaned": resume_content,
            "raw": resume_raw,
            "tokens": resume_tokens,
            "token_set": resume_token_set,
            "term_set": resume_term_set,
        })

    results = {}
    vectorizer = TfidfVectorizer()
    try:
        vectors = vectorizer.fit_transform([cleaned_job_desc] + resume_contents)
        similarity_matrix = cosine_similarity(vectors[0:1], vectors[1:])
        tfidf_scores = similarity_matrix.flatten()
    except ValueError:
        tfidf_scores = [0.0 for _ in resume_contents]

    emit_progress(12)

    detailed_results = []

    ranked_indices = sorted(range(len(tfidf_scores)), key=lambda i: tfidf_scores[i], reverse=True)
    top_k = min(TOP_K_FOR_EMBEDDING, len(ranked_indices))
    top_k_indices = set(ranked_indices[:top_k])

    jd_embedding = None
    if use_embedding and top_k_indices:
        if not _embedding_loaded:
            emit_progress(15)
        jd_embedding = _get_cached_jd_embedding(job_description_text)

    total_resumes = len(resume_entries)
    for index, entry in enumerate(resume_entries):
        pdf_file = entry["filename"]
        resume_content = entry["cleaned"]
        resume_similarity_content = resume_content[:max_resume_chars]
        resume_tokens = entry["tokens"]
        resume_token_set = entry["token_set"]
        resume_raw = entry["raw"]
        resume_term_set = entry["term_set"]
        tfidf_similarity = tfidf_scores[index]
        allow_fuzzy = index in top_k_indices

        embedding_similarity = None
        if use_embedding and index in top_k_indices and jd_embedding is not None:
            resume_embedding = _get_cached_resume_embedding(entry["path"], resume_similarity_content)
            if resume_embedding is not None:
                similarity = (jd_embedding * resume_embedding).sum().clamp(-1.0, 1.0)
                embedding_similarity = float((similarity + 1.0) / 2.0)

        if embedding_similarity is not None:
            similarity_score = (0.4 * tfidf_similarity) + (0.6 * embedding_similarity)
        else:
            similarity_score = tfidf_similarity

        matched_terms = list(skill_terms_set & resume_term_set)
        match_types = {term: "exact" for term in matched_terms}
        fuzzy_matches = 0

        if allow_fuzzy:
            unmatched_terms = skill_terms_set - set(matched_terms)
            for term in unmatched_terms:
                synonyms = SYNONYM_LOOKUP.get(term, [])
                if any(variant in resume_term_set for variant in synonyms):
                    matched_terms.append(term)
                    match_types[term] = "synonym"
                    continue
                if fuzzy_match_term(term, resume_tokens):
                    matched_terms.append(term)
                    match_types[term] = "fuzzy"
                    fuzzy_matches += 1

        weighted_matches = 0.0
        for term in matched_terms:
            weight = 1.5 if term in IMPORTANT_TECH_TERMS else 1.0
            weight *= get_stream_term_multiplier(stream, term)
            weighted_matches += weight
        term_bonus = min(weighted_matches * 0.05, 0.15)
        bonus_score = term_bonus
        final_score = similarity_score + bonus_score

        missing_mandatory = []
        if mandatory_terms:
            for term in mandatory_terms:
                if term in resume_term_set:
                    continue
                if allow_fuzzy:
                    synonyms = SYNONYM_LOOKUP.get(term, [])
                    if any(variant in resume_term_set for variant in synonyms):
                        continue
                    if fuzzy_match_term(term, resume_tokens):
                        continue
                missing_mandatory.append(term)
        must_have_pass = len(missing_mandatory) == 0

        fuzzy_ratio = 1.0
        if matched_terms:
            fuzzy_ratio = fuzzy_matches / len(matched_terms)
        confidence_level = estimate_confidence(similarity_score, len(matched_terms), fuzzy_ratio)

        evidence_map = {}
        if allow_fuzzy:
            evidence_map = build_evidence_map(matched_terms, resume_raw, max_snippets=2)

        match_percentage = final_score * 100
        results[pdf_file] = match_percentage

        detailed_results.append({
            "filename": pdf_file,
            "match_percentage": match_percentage,
            "must_have_pass": must_have_pass,
            "missing_mandatory_skills": missing_mandatory,
            "confidence_level": confidence_level,
            "evidence_map": evidence_map,
            "score_breakdown": {
                "similarity_score": similarity_score,
                "bonus_score": bonus_score,
                "final_score": final_score,
            },
        })

        percent = 10 + int(((index + 1) / max(1, total_resumes)) * 60)
        emit_progress(percent)

    emit_progress(70)

    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    if not return_details:
        return sorted_results

    detailed_map = {item["filename"]: item for item in detailed_results}
    sorted_detailed_results = []
    for filename, match_percentage in sorted_results:
        detail = detailed_map.get(filename, {})
        if detail:
            sorted_detailed_results.append(detail)
        else:
            sorted_detailed_results.append({
                "filename": filename,
                "match_percentage": match_percentage,
                "must_have_pass": True,
                "missing_mandatory_skills": [],
                "confidence_level": "Low",
                "evidence_map": {},
                "score_breakdown": {
                    "similarity_score": match_percentage / 100,
                    "bonus_score": 0.0,
                    "final_score": match_percentage / 100,
                },
            })
    return sorted_detailed_results


def _get_best_candidate_name(sorted_results: List[Any]) -> Optional[str]:
    if not sorted_results:
        return None
    first = sorted_results[0]
    if isinstance(first, dict):
        return first.get("filename")
    if isinstance(first, (list, tuple)) and len(first) > 0:
        return first[0]
    return None


def _get_best_candidate_must_have(sorted_results: List[Any]) -> Optional[bool]:
    if not sorted_results:
        return None
    first = sorted_results[0]
    if isinstance(first, dict):
        return first.get("must_have_pass")
    return None


def build_recommendation_message(total_scored: int, must_have_pass: Optional[bool], override: Optional[Dict[str, Any]]) -> str:
    if override:
        if override.get("force_reject") is True:
            return "Recruiter override: Not aligned with requirements."
        if override.get("force_accept") is True:
            return "Recruiter override: Recommended for recruiter review."

    if must_have_pass is False:
        return "Currently not aligned with requirements (missing mandatory skills)."
    if total_scored >= 35:
        return "Recommended for recruiter review."
    return "Currently not aligned with requirements."


def analyze_best_candidate(
    sorted_results,
    resume_folder=None,
    override: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable[[int], None]] = None,
):
    recommendation_message = ""

    if not sorted_results:
        empty_scored = pd.DataFrame([], columns=['   score earned'])
        return None, empty_scored, 0, "Status: No resumes available for analysis."

    resume_folder = resume_folder or os.path.abspath(os.path.join(os.getcwd(), "uploads"))
    best_candidate = _get_best_candidate_name(sorted_results)
    must_have_pass = _get_best_candidate_must_have(sorted_results)
    safe_progress_callback(progress_callback, 80)
    if not best_candidate:
        empty_scored = pd.DataFrame([], columns=['   score earned'])
        return None, empty_scored, 0, "Status: No resumes available for analysis."

    best_path = os.path.join(resume_folder, best_candidate)
    best_bundle = get_cached_resume_bundle(best_path)
    resume_tokens = best_bundle.get("tokens", [])
    resume_token_set = best_bundle.get("token_set", set())

    Area_with_key_term = {'Data science': ['algorithm', 'analytics', 'hadoop', 'machine learning', 'data mining', 'python',
                                           'statistics', 'data', 'statistical analysis', 'data wrangling', 'algebra', 'Probability',
                                           'visualization'],

                          'Programming': ['python', 'r programming', 'sql', 'c++', 'scala', 'julia', 'tableau',
                                          'javasript', 'powerbI', 'code', 'coding', 'javascript', 'python'],

                          'Experience': ['project', 'years', 'company', 'excellency', 'promotion', 'award',
                                         'outsourcing', 'work in progress'],

                          'Management skill': ['administration', 'budget', 'cost', 'direction', 'feasibility analysis',
                                               'finance', 'leader', 'leadership', 'management', 'milestones', 'planning',
                                               'problem', 'project', 'risk', 'schedule', 'stakeholders', 'English'],

                          'Data analytics': ['api', 'big data', 'clustering', 'code', 'coding', 'data', 'database',
                                             'data mining', 'data science', 'deep learning', 'hadoop',
                                             'hypothesis test', 'machine learning', 'dbms', 'modeling', 'nlp',
                                             'predictive', 'text mining', 'visualuzation'],

                          'Statistics': ['parameter', 'vaiable', 'ordinal', 'ratio', 'nominal', 'interval', 'descriptive',
                                         'inferential', 'linear', 'correlations', 'probability',
                                         'regression', 'mean', 'variance', 'standard deviation'],

                          'Machine learning': ['supervised learning', 'unsupervised learning', 'ann', 'artificial neural network',
                                               'overfitting', 'computer vision', 'natural language processing',
                                               'database'],

                          'Data analyst': ['data collection', 'data cleaning', 'data Processing', 'interpreting data',
                                           'streamlining data', 'visualizing data', 'statistics',
                                           'tableau', 'tables', 'analytical'],

                          'Software': ['django', 'cloud', 'gcp', 'aws', 'javacript', 'react', 'redux',
                                       'es6', 'node.js', 'typescript', 'html', 'css', 'ui', 'ci/cd', 'cashflow'],

                          'Web skill': ['web design', 'branding', 'graphic design', 'seo', 'marketing', 'logo design', 'video editing',
                                        'es6', 'node.js', 'typescript', 'html/css',
                                        'ci/cd'],

                          'Personal Skill': ['leadership', 'team work', 'integrity', 'public speaking', 'team leadership', 'problem solving', 'loyalty', 'quality', 'performance improvement', 'six sigma',
                                             'quality circles', 'quality tools', 'process improvement', 'capability analysis', 'control'],

                          'Accounting': ['communication', 'sales', 'sales process', 'solution selling', 'crm',
                                         'sales management', 'sales operations', 'marketing', 'direct sales', 'trends', 'b2b', 'marketing strategy', 'saas',
                                         'business development'],

                          'Sales & marketing': ['retail', 'manufacture', 'corporate', 'goodssale', 'consumer',
                                                'package', 'fmcg', 'account', 'management', 'lead generation', 'cold calling', 'customer service',
                                                'inside sales', 'sales', 'promotion'],

                          'Graphic': ['brand identity', 'editorial design', 'design', 'branding', 'logo design',
                                      'letterhead design', 'business card design', 'brand strategy', 'stationery design', 'graphic design',
                                      'exhibition graphic design'],

                          'Content skill': ['editing', 'creativity', 'content idea', 'problem solving', 'writer',
                                            'content thinker', 'copy editor', 'researchers', 'technology geek', 'public speaking', 'online marketing'],

                          'Graphical content': ['photographer', 'videographer', 'graphic artist', 'copywriter', 'search engine optimization',
                                                'seo', 'social media', 'page insight', 'gain audience'],

                          'Finanace': ['financial reporting', 'budgeting', 'forecasting', 'strong analytical thinking', 'financial planning',
                                       'payroll tax', 'accounting', 'productivity', 'reporting costs', 'balance sheet',
                                       'financial statements'],

                          'Health/Medical': ['abdominal surgery', 'laparoscopy', 'trauma surgery', 'adult intensive care',
                                             'pain management', 'cardiology', 'patient', 'surgery', 'hospital', 'healthcaret', 'doctor', 'medicine'],

                          'Language': ['english', 'malay', 'mandarin', 'bangla', 'hindi', 'tamil']
                          }


    data_science = 0
    code_proficiency = 0
    experience_field = 0
    management_skill = 0
    data_analytics = 0
    statistical_skill = 0
    machine_learning = 0
    Data_analyst = 0
    Softwareskill = 0
    Webskill = 0
    PersonalSkill = 0
    accounting_skill = 0
    Sales_marketing = 0
    Graphic_skill = 0
    Content_skill = 0
    Graphical_content = 0
    Finanace_skill = 0
    Health_Medical = 0
    Languages = 0

    domain_terms = set()
    for terms in Area_with_key_term.values():
        domain_terms.update(terms)
    max_domain_term_len = max((len(term.split()) for term in domain_terms), default=1)
    resume_term_set = resume_token_set | _build_ngram_set(resume_tokens, max_domain_term_len)

    scores = []

    for domain in Area_with_key_term.keys():

        if domain == 'Data science':
            data_science = len(set(Area_with_key_term[domain]) & resume_term_set)
            scores.append(data_science)

        elif domain == 'Programming':
            code_proficiency = len(set(Area_with_key_term[domain]) & resume_term_set)
            scores.append(code_proficiency)

        elif domain == 'Experience':
            experience_field = len(set(Area_with_key_term[domain]) & resume_term_set)
            scores.append(experience_field)

        elif domain == 'Data analytics':
            data_analytics = len(set(Area_with_key_term[domain]) & resume_term_set)
            scores.append(data_analytics)

        elif domain == 'Management skill':
            management_skill = len(set(Area_with_key_term[domain]) & resume_term_set)
            scores.append(management_skill)

        elif domain == 'Statistics':
            statistical_skill = len(set(Area_with_key_term[domain]) & resume_term_set)
            scores.append(statistical_skill)

        elif domain == 'Data analyst':
            Data_analyst = len(set(Area_with_key_term[domain]) & resume_term_set)
            scores.append(Data_analyst)

        elif domain == 'Software':
            Softwareskill = len(set(Area_with_key_term[domain]) & resume_term_set)
            scores.append(Softwareskill)

        elif domain == 'Web skill':
            Webskill = len(set(Area_with_key_term[domain]) & resume_term_set)
            scores.append(Webskill)

        elif domain == 'Personal Skill':
            PersonalSkill = len(set(Area_with_key_term[domain]) & resume_term_set)
            scores.append(PersonalSkill)

        elif domain == 'Accounting':
            accounting_skill = len(set(Area_with_key_term[domain]) & resume_term_set)
            scores.append(accounting_skill)

        elif domain == 'Sales & marketing':
            Sales_marketing = len(set(Area_with_key_term[domain]) & resume_term_set)
            scores.append(Sales_marketing)

        elif domain == 'Graphic':
            Graphic_skill = len(set(Area_with_key_term[domain]) & resume_term_set)
            scores.append(Graphic_skill)

        elif domain == 'Content skill':
            Content_skill = len(set(Area_with_key_term[domain]) & resume_term_set)
            scores.append(Content_skill)

        elif domain == 'Graphical content':
            Graphical_content = len(set(Area_with_key_term[domain]) & resume_term_set)
            scores.append(Graphical_content)

        elif domain == 'Finanace':
            Finanace_skill = len(set(Area_with_key_term[domain]) & resume_term_set)
            scores.append(Finanace_skill)

        elif domain == 'Health/Medical':
            Health_Medical = len(set(Area_with_key_term[domain]) & resume_term_set)
            scores.append(Health_Medical)

        elif domain == 'Language':
            Languages = len(set(Area_with_key_term[domain]) & resume_term_set)
            scores.append(Languages)

        else:
            machine_learning = len(set(Area_with_key_term[domain]) & resume_term_set)
            scores.append(machine_learning)

    scored = pd.DataFrame(scores, index=Area_with_key_term.keys(), columns=[
        '   score earned']).sort_values(by='   score earned', ascending=False)

    total_scored = sum(scores)

    safe_progress_callback(progress_callback, 90)

    recommendation_message = build_recommendation_message(total_scored, must_have_pass, override)

    safe_progress_callback(progress_callback, 100)

    return best_candidate, scored, total_scored, recommendation_message


def main():
    return run_ats()


def run_ats():
    job_description_text = prompt_job_description()
    uploads_folder = os.path.abspath(os.path.join(os.getcwd(), "uploads"))

    sorted_results = rank_resumes(job_description_text)

    if not sorted_results:
        return {
            "ranking": [],
            "best_candidate": None,
            "domain_scores": pd.DataFrame([], columns=['   score earned']),
            "total_score": 0,
            "recommendation": "Status: No resumes available for analysis.",
        }

    print("\nTop Candidates:")
    for rank, (filename, percentage) in enumerate(sorted_results, start=1):
        print(f"{rank}. {filename} → {percentage:.1f}%")

    best_candidate = sorted_results[0][0]
    print(f"\nBest Candidate Selected: {best_candidate}")
    if os.path.isdir(uploads_folder):
        reader = PyPDF2.PdfReader(os.path.join(uploads_folder, best_candidate))
        print("The resume contains", len(reader.pages), "pages.")

    best_candidate, scored, total_scored, recommendation = analyze_best_candidate(sorted_results)

    result = {
        "ranking": sorted_results,
        "best_candidate": best_candidate,
        "domain_scores": scored,
        "total_score": total_scored,
        "recommendation": recommendation,
    }
    return result


if __name__ == "__main__":
    main()
