"""
Comprehensive Resume and Job Description Matching System
Implements all phases: PDF extraction, LLM ensemble, semantic similarity, skills extraction, and multi-layer validation.
Enhanced with Final Similarity Score calculation.
"""
import os
import re
import io
import pdfplumber
import fitz  # PyMuPDF
import numpy as np
from typing import Optional, List, Dict, Any
from sentence_transformers import SentenceTransformer, util
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import requests
import spacy
from fuzzywuzzy import fuzz
from fuzzywuzzy import process as fuzzy_process
from dotenv import load_dotenv
import time

# Load environment variables from .env file
load_dotenv()

# Load spaCy model for NER
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# ========== Phase 1: Enhanced PDF Processing ==========
class PDFExtractor:
    """Extracts text from PDFs using pdfplumber and PyMuPDF as fallback. OCR removed."""
    @staticmethod
    def extract_text(pdf_bytes: bytes) -> str:
        # Try pdfplumber first
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                text = "\n".join(page.extract_text() or '' for page in pdf.pages)
                if text.strip():
                    return PDFExtractor.clean_text(text)
        except Exception:
            pass
        # Fallback to PyMuPDF
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text = "\n".join(page.get_text() for page in doc)
            doc.close()
            if text.strip():
                return PDFExtractor.clean_text(text)
        except Exception:
            pass
        # If both fail, return empty string
        return ""

    @staticmethod
    def clean_text(text: str) -> str:
        # Remove excessive whitespace, fix line breaks, basic formatting recovery
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        return text.strip()

# ========== Phase 2: Advanced LLM Integration ==========
class ImprovedLLMEnsemble:
    """Smart LLM ensemble with fallback strategy instead of calling all APIs"""
    def __init__(self, groq_api_key: Optional[str] = None, cohere_api_key: Optional[str] = None):
        self.groq_api_key = groq_api_key
        self.cohere_api_key = cohere_api_key
        self.llm_endpoints = [
            ("Groq (Llama3-70B)", self.query_groq),
            ("Together (Mixtral)", self.query_huggingface),
            ("Together (CodeLlama)", self.query_together),
            ("Cohere", self.query_cohere)
        ]
        self.success_rates = {}
        self.response_times = {}

    def query_groq(self, prompt: str) -> Optional[str]:
        print("[LLMEnsemble] Calling Groq API...")
        if not self.groq_api_key:
            print("[LLMEnsemble] Groq API key not provided. Skipping Groq API call.")
            return None
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.groq_api_key}", "Content-Type": "application/json"}
        data = {
            "model": "llama3-70b-8192",  # Best free Groq model
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 1000
        }
        try:
            r = requests.post(url, headers=headers, json=data, timeout=30)
            print(f"[LLMEnsemble] Groq API response status: {r.status_code}")
            if r.status_code == 200:
                print("[LLMEnsemble] Groq API returned a response.")
                return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[LLMEnsemble] Groq API call failed: {e}")
            return None
    
    def query_huggingface(self, prompt: str) -> Optional[str]:
        # Now using Together AI API for Mixtral-8x7B-Instruct-v0.1
        print("[LLMEnsemble] Calling Together AI API for Mixtral-8x7B-Instruct-v0.1...")
        together_api_key = os.getenv("TOGETHER_API_KEY")
        url = "https://api.together.xyz/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        if together_api_key:
            headers["Authorization"] = f"Bearer {together_api_key}"
        data = {
            "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 1000
        }
        try:
            r = requests.post(url, headers=headers, json=data, timeout=30)
            print(f"[LLMEnsemble] Together AI (Mixtral) API response status: {r.status_code}")
            if r.status_code == 200:
                print("[LLMEnsemble] Together AI (Mixtral) API returned a response.")
                return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[LLMEnsemble] Together AI (Mixtral) API call failed: {e}")
        return None

    def query_together(self, prompt: str) -> Optional[str]:
        print("[LLMEnsemble] Calling Together.ai API...")
        url = "https://api.together.xyz/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": "codellama/CodeLlama-34b-Instruct-hf",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 1000
        }
        try:
            r = requests.post(url, headers=headers, json=data, timeout=30)
            print(f"[LLMEnsemble] Together.ai API response status: {r.status_code}")
            if r.status_code == 200:
                print("[LLMEnsemble] Together.ai API returned a response.")
                return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[LLMEnsemble] Together.ai API call failed: {e}")
            return None

    def query_cohere(self, prompt: str) -> Optional[str]:
        print("[LLMEnsemble] Calling Cohere API...")
        if not self.cohere_api_key:
            print("[LLMEnsemble] Cohere API key not provided. Skipping Cohere API call.")
            return None
        url = "https://api.cohere.ai/v1/generate"
        headers = {"Authorization": f"Bearer {self.cohere_api_key}", "Content-Type": "application/json"}
        data = {"model": "command", "prompt": prompt, "max_tokens": 500}
        try:
            r = requests.post(url, headers=headers, json=data, timeout=30)
            print(f"[LLMEnsemble] Cohere API response status: {r.status_code}")
            if r.status_code == 200:
                print("[LLMEnsemble] Cohere API returned a response.")
                return r.json()["generations"][0]["text"]
        except Exception as e:
            print(f"[LLMEnsemble] Cohere API call failed: {e}")
            return None
    
    def advanced_prompt(self, resume_text: str, job_description: str) -> str:
        # Chain-of-thought, few-shot, JSON schema
        return f"""
Analyze the following resume and job description for compatibility. Provide a JSON with:
- compatibility_score (0-100)
- strengths (list)
- gaps (list)
- recommendations (list)

RESUME: {resume_text[:2000]}
JOB DESCRIPTION: {job_description[:2000]}
"""

    def get_smart_response(self, resume_text: str, job_description: str, strategy: str = "fallback") -> Dict[str, Any]:
        prompt = self.advanced_prompt(resume_text, job_description)
        if strategy == "fallback":
            return self._fallback_strategy(prompt)
        elif strategy == "ensemble":
            return self._ensemble_strategy(prompt)
        elif strategy == "best":
            return self._best_api_strategy(prompt)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _fallback_strategy(self, prompt: str) -> Dict[str, Any]:
        for api_name, api_func in self.llm_endpoints:
            print(f"[LLM] Trying {api_name}...")
            try:
                start_time = time.time()
                response = api_func(prompt)
                response_time = time.time() - start_time
                if response:
                    parsed_response = self._parse_and_validate_response(response)
                    if parsed_response:
                        self._update_success_rate(api_name, True, response_time)
                        print(f"[LLM] ✅ {api_name} succeeded in {response_time:.2f}s")
                        return {**parsed_response, "api_used": api_name, "response_time": response_time}
                self._update_success_rate(api_name, False, response_time)
            except Exception as e:
                print(f"[LLM] ❌ {api_name} failed: {e}")
                self._update_success_rate(api_name, False, 0)
                continue
        print("[LLM] ⚠️ All APIs failed, using default response")
        return self._get_default_response()

    def _ensemble_strategy(self, prompt: str, max_apis: int = 2) -> Dict[str, Any]:
        responses = []
        apis_called = 0
        for api_name, api_func in self.llm_endpoints:
            if apis_called >= max_apis:
                break
            try:
                response = api_func(prompt)
                if response:
                    parsed = self._parse_and_validate_response(response)
                    if parsed:
                        responses.append({**parsed, "api_name": api_name})
                        apis_called += 1
            except Exception as e:
                print(f"[LLM] {api_name} failed: {e}")
                continue
        if not responses:
            return self._get_default_response()
        return self._aggregate_responses(responses)

    def _best_api_strategy(self, prompt: str) -> Dict[str, Any]:
        if not self.success_rates:
            return self._fallback_strategy(prompt)
        best_api = max(self.success_rates.keys(), key=lambda x: self.success_rates[x]["success_rate"] - sum(self.response_times.get(x, [10]))/max(len(self.response_times.get(x, [1])),1))
        api_func = None
        for api_name, func in self.llm_endpoints:
            if api_name == best_api:
                api_func = func
                break
        if api_func:
            try:
                response = api_func(prompt)
                if response:
                    parsed = self._parse_and_validate_response(response)
                    if parsed:
                        return {**parsed, "api_used": best_api}
            except Exception:
                pass
        return self._fallback_strategy(prompt)

    def _parse_and_validate_response(self, response: str) -> Optional[Dict[str, Any]]:
        try:
            import json
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                return None
            json_str = json_match.group()
            parsed = json.loads(json_str)
            required_fields = ["compatibility_score", "strengths", "gaps", "recommendations"]
            if not all(field in parsed for field in required_fields):
                return None
            score = parsed.get("compatibility_score", 0)
            if not (0 <= score <= 100):
                return None
            return parsed
        except Exception as e:
            print(f"[LLM] JSON parsing failed: {e}")
            return None

    def _aggregate_responses(self, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(responses) == 1:
            return responses[0]
        scores = [r.get("compatibility_score", 0) for r in responses]
        avg_score = sum(scores) / len(scores)
        all_strengths = []
        all_gaps = []
        all_recommendations = []
        for r in responses:
            all_strengths.extend(r.get("strengths", []))
            all_gaps.extend(r.get("gaps", []))
            all_recommendations.extend(r.get("recommendations", []))
        def dedupe_list(lst):
            seen = set()
            result = []
            for item in lst:
                if item not in seen:
                    seen.add(item)
                    result.append(item)
            return result
        return {
            "compatibility_score": avg_score,
            "strengths": dedupe_list(all_strengths),
            "gaps": dedupe_list(all_gaps),
            "recommendations": dedupe_list(all_recommendations),
            "apis_used": [r.get("api_name", "unknown") for r in responses],
            "ensemble_size": len(responses)
        }

    def _update_success_rate(self, api_name: str, success: bool, response_time: float):
        if api_name not in self.success_rates:
            self.success_rates[api_name] = {"successes": 0, "total": 0}
        self.success_rates[api_name]["total"] += 1
        if success:
            self.success_rates[api_name]["successes"] += 1
        total = self.success_rates[api_name]["total"]
        successes = self.success_rates[api_name]["successes"]
        self.success_rates[api_name]["success_rate"] = successes / total
        if response_time > 0:
            if api_name not in self.response_times:
                self.response_times[api_name] = []
            self.response_times[api_name].append(response_time)
            self.response_times[api_name] = self.response_times[api_name][-10:]

    def _get_default_response(self) -> Dict[str, Any]:
        return {
            "compatibility_score": 50,
            "strengths": ["Unable to analyze - API unavailable"],
            "gaps": ["Unable to analyze - API unavailable"],
            "recommendations": ["Please try again later or check API keys"],
            "api_used": "default",
            "error": "All LLM APIs failed"
        }

    def get_api_stats(self) -> Dict[str, Any]:
        stats = {}
        for api_name in self.success_rates:
            stats[api_name] = {
                "success_rate": self.success_rates[api_name]["success_rate"],
                "total_calls": self.success_rates[api_name]["total"],
                "avg_response_time": sum(self.response_times.get(api_name, [0])) / len(self.response_times.get(api_name, [1]))
            }
        return stats

# ========== Phase 3: BERT-Based Semantic Enhancement ==========
class EnhancedBERTSemanticEngine:
    """
    Enhanced BERT engine with specialized models for resume/job matching
    """
    def __init__(self, resume_bert_model: Optional[str] = None, load_specialized_models: bool = True):
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.distilbert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.resume_bert_model = None
        self.resume_model_name = None
        if resume_bert_model:
            self.resume_bert_model = self._load_resume_model(resume_bert_model)
            self.resume_model_name = resume_bert_model
        if self.resume_bert_model is None and load_specialized_models:
            self.resume_bert_model, self.resume_model_name = self._load_best_available_resume_model()
        self.specialized_models = {}
        if load_specialized_models:
            self._load_specialized_models()

    def _load_resume_model(self, model_name: str) -> Optional[SentenceTransformer]:
        try:
            print(f"[BERT] Loading resume-specific model: {model_name}")
            model = SentenceTransformer(model_name)
            print(f"[BERT] ✅ Successfully loaded: {model_name}")
            return model
        except Exception as e:
            print(f"[BERT] ❌ Failed to load {model_name}: {e}")
            return None

    def _load_best_available_resume_model(self) -> tuple[Optional[SentenceTransformer], Optional[str]]:
        candidate_models = [
            "sentence-transformers/all-mpnet-base-v2",
            "sentence-transformers/all-roberta-large-v1",
            "nlpaueb/legal-bert-base-uncased",
            "ProsusAI/finbert",
            "sentence-transformers/multi-qa-mpnet-base-dot-v1",
            "sentence-transformers/all-distilroberta-v1",
            "sentence-transformers/paraphrase-mpnet-base-v2"
        ]
        for model_name in candidate_models:
            model = self._load_resume_model(model_name)
            if model is not None:
                print(f"[BERT] 🎯 Using {model_name} as resume-specific model")
                return model, model_name
        print("[BERT] ⚠️ No specialized resume model could be loaded")
        return None, None

    def _load_specialized_models(self):
        specialized_candidates = {
            "business_model": [
                "sentence-transformers/all-mpnet-base-v2",
                "ProsusAI/finbert"
            ],
            "technical_model": [
                "sentence-transformers/all-roberta-large-v1",
                "microsoft/codebert-base"
            ],
            "quality_model": [
                "sentence-transformers/paraphrase-mpnet-base-v2",
                "sentence-transformers/multi-qa-mpnet-base-dot-v1"
            ]
        }
        for category, models in specialized_candidates.items():
            for model_name in models:
                try:
                    model = SentenceTransformer(model_name)
                    self.specialized_models[category] = {
                        'model': model,
                        'name': model_name
                    }
                    print(f"[BERT] ✅ Loaded {category}: {model_name}")
                    break
                except Exception as e:
                    print(f"[BERT] ❌ Failed to load {model_name}: {e}")
                    continue

    def semantic_similarity(self, text1: str, text2: str) -> float:
        emb1 = self.semantic_model.encode(text1, convert_to_tensor=True)
        emb2 = self.semantic_model.encode(text2, convert_to_tensor=True)
        score = float(util.pytorch_cos_sim(emb1, emb2).item())
        return score

    def resume_specific_similarity(self, text1: str, text2: str) -> Optional[float]:
        if self.resume_bert_model:
            try:
                emb1 = self.resume_bert_model.encode(text1, convert_to_tensor=True)
                emb2 = self.resume_bert_model.encode(text2, convert_to_tensor=True)
                score = float(util.pytorch_cos_sim(emb1, emb2).item())
                print(f"[BERT] Resume-specific similarity: {score:.4f} using {self.resume_model_name}")
                return score
            except Exception as e:
                print(f"[BERT] Error in resume-specific similarity: {e}")
                return None
                print("[BERT] No resume-specific model available")
                return None

    def ensemble_resume_similarity(self, resume_text: str, job_description: str) -> Dict[str, Any]:
        scores = {}
        model_details = {}
        try:
            scores['semantic_base'] = self.semantic_similarity(resume_text, job_description)
            model_details['semantic_base'] = 'all-MiniLM-L6-v2'
        except Exception as e:
            print(f"[BERT] Error with base semantic model: {e}")
        resume_score = self.resume_specific_similarity(resume_text, job_description)
        if resume_score is not None:
            scores['resume_specific'] = resume_score
            model_details['resume_specific'] = self.resume_model_name
        for category, model_info in self.specialized_models.items():
            try:
                model = model_info['model']
                emb1 = model.encode(resume_text, convert_to_tensor=True)
                emb2 = model.encode(job_description, convert_to_tensor=True)
                score = float(util.pytorch_cos_sim(emb1, emb2).item())  # Convert numpy.float to Python float
                scores[category] = score
                model_details[category] = model_info['name']
                print(f"[BERT] {category} similarity: {score:.4f}")
            except Exception as e:
                print(f"[BERT] Error with {category} model: {e}")
                continue
        
        # Automatically select the best model based on highest similarity score
        best_model = self._select_best_model(scores, model_details)
        
        ensemble_score = self._calculate_ensemble_score(scores)
        domain_analysis = self._analyze_domain_suitability(resume_text, job_description)
        return {
            'individual_scores': scores,
            'model_details': model_details,
            'ensemble_score': ensemble_score,
            'domain_analysis': domain_analysis,
            'models_used': len(scores),
            'primary_resume_model': self.resume_model_name,
            'best_model': best_model,
            'confidence': self._calculate_confidence(scores)
        }
    
    def _select_best_model(self, scores: Dict[str, float], model_details: Dict[str, str]) -> Dict[str, Any]:
        """Automatically select the best model based on highest similarity score"""
        if not scores:
            return {"model_name": "none", "score": 0.0, "category": "none"}
        
        # Find the model with the highest score
        best_category = max(scores.keys(), key=lambda k: scores[k])
        best_score = scores[best_category]
        best_model_name = model_details.get(best_category, best_category)
        
        print(f"[BERT] 🎯 Best model selected: {best_category} ({best_model_name}) with score: {best_score:.4f}")
        
        return {
            "model_name": best_model_name,
            "score": best_score,
            "category": best_category,
            "all_scores": scores,
            "all_models": model_details
        }

    def _calculate_ensemble_score(self, scores: Dict[str, float]) -> float:
        if not scores:
            return 0.0
        weights = {
            'semantic_base': 0.2,
            'resume_specific': 0.35,
            'business_model': 0.25,
            'technical_model': 0.15,
            'quality_model': 0.2
        }
        weighted_sum = 0.0
        total_weight = 0.0
        for score_type, score in scores.items():
            weight = weights.get(score_type, 0.1)
            weighted_sum += weight * score
            total_weight += weight
        return weighted_sum / total_weight if total_weight > 0 else np.mean(list(scores.values()))

    def _analyze_domain_suitability(self, resume_text: str, job_description: str) -> Dict[str, Any]:
        resume_lower = resume_text.lower()
        job_lower = job_description.lower()
        tech_keywords = [
            'python', 'java', 'javascript', 'programming', 'software', 'developer',
            'algorithm', 'database', 'api', 'framework', 'cloud', 'machine learning',
            'ai', 'data science', 'devops', 'kubernetes', 'docker'
        ]
        business_keywords = [
            'finance', 'accounting', 'business', 'management', 'strategy', 'marketing',
            'sales', 'consulting', 'operations', 'project management', 'leadership'
        ]
        legal_keywords = [
            'legal', 'compliance', 'regulation', 'policy', 'governance', 'audit',
            'risk management', 'contract', 'intellectual property'
        ]
        tech_score = sum(1 for keyword in tech_keywords if keyword in resume_lower or keyword in job_lower)
        business_score = sum(1 for keyword in business_keywords if keyword in resume_lower or keyword in job_lower)
        legal_score = sum(1 for keyword in legal_keywords if keyword in resume_lower or keyword in job_lower)
        max_score = max(tech_score, business_score, legal_score)
        if max_score == 0:
            primary_domain = 'general'
        elif tech_score == max_score:
            primary_domain = 'technical'
        elif business_score == max_score:
            primary_domain = 'business'
        else:
            primary_domain = 'legal'
        return {
            'primary_domain': primary_domain,
            'domain_scores': {
                'technical': tech_score,
                'business': business_score,
                'legal': legal_score
            },
            'specialization_strength': float(max_score / (len(resume_text.split()) + len(job_description.split())) * 1000)  # Convert to Python float
        }

    def _calculate_confidence(self, scores: Dict[str, float]) -> float:
        if not scores:
            return 0.0
        model_confidence = min(len(scores) / 4.0, 1.0)
        score_values = list(scores.values())
        if len(score_values) > 1:
            consistency = 1.0 - min(float(np.std(score_values)), 0.5) * 2  # Convert numpy.float to Python float
        else:
            consistency = 0.7
        resume_model_bonus = 0.1 if 'resume_specific' in scores else 0.0
        return min(1.0, (model_confidence * 0.4 + consistency * 0.5 + resume_model_bonus + 0.1))

    def context_embedding(self, text: str) -> np.ndarray:
        inputs = self.distilbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.distilbert_model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return cls_embedding[0].tolist()  # Convert numpy array to list

    def skills_similarity(self, skills1: List[str], skills2: List[str]) -> float:
        if not skills1 or not skills2:
            return 0.0
        model_to_use = self.resume_bert_model if self.resume_bert_model else self.semantic_model
        emb1 = model_to_use.encode(skills1, convert_to_tensor=True)
        emb2 = model_to_use.encode(skills2, convert_to_tensor=True)
        sim_matrix = util.pytorch_cos_sim(emb1, emb2)
        best_sim_1 = float(sim_matrix.max(dim=1).values.mean().item())  # Convert numpy.float to Python float
        best_sim_2 = float(sim_matrix.max(dim=0).values.mean().item())  # Convert numpy.float to Python float
        semantic_skill_sim = (best_sim_1 + best_sim_2) / 2
        from fuzzywuzzy import fuzz
        fuzzy_matches = 0
        total_comparisons = 0
        for skill1 in skills1:
            for skill2 in skills2:
                similarity_ratio = fuzz.ratio(skill1.lower(), skill2.lower()) / 100.0
                fuzzy_matches += similarity_ratio
                total_comparisons += 1
        fuzzy_skill_sim = fuzzy_matches / total_comparisons if total_comparisons > 0 else 0.0
        set1 = set(skill.lower() for skill in skills1)
        set2 = set(skill.lower() for skill in skills2)
        jaccard_sim = len(set1.intersection(set2)) / len(set1.union(set2)) if set1.union(set2) else 0.0
        final_score = (0.5 * semantic_skill_sim + 0.3 * fuzzy_skill_sim + 0.2 * jaccard_sim)
        skill_coverage_bonus = min(len(set1.intersection(set2)) / max(len(set1), len(set2), 1) * 0.1, 0.2)
        return min(1.0, final_score + skill_coverage_bonus)

    def get_model_info(self) -> Dict[str, Any]:
        return {
            'primary_semantic_model': 'all-MiniLM-L6-v2',
            'resume_specific_model': self.resume_model_name,
            'specialized_models': {k: v['name'] for k, v in self.specialized_models.items()},
            'total_models_loaded': 1 + (1 if self.resume_bert_model else 0) + len(self.specialized_models),
            'resume_model_available': self.resume_bert_model is not None
        }

# ========== Phase 4: Dynamic Skills System ==========
class SkillsExtractor:
    """Extracts and matches skills using NER, fuzzy matching, and context classification."""
    def __init__(self, skill_db: Optional[List[str]] = None):
        # Optionally load a dynamic skills database
        self.skill_db = skill_db or [
            # Programming Languages
            "python", "java", "javascript", "typescript", "c++", "c#", "php", "ruby", "go", "rust", "scala", "kotlin", "swift",
            # Web Technologies
            "react", "angular", "vue", "html", "css", "sass", "less", "bootstrap", "tailwind", "jquery",
            # Backend Frameworks
            "django", "flask", "fastapi", "spring", "express", "node.js", "laravel", "rails",
            # Databases
            "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch", "sqlite", "oracle",
            # Cloud & DevOps
            "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "git", "terraform", "ansible",
            # Data Science & ML
            "machine learning", "deep learning", "ai", "data science", "nlp", "computer vision",
            "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras", "jupyter",
            # Other Technologies
            "api", "rest", "graphql", "microservices", "agile", "scrum", "ci/cd", "testing",
            "linux", "bash", "powershell", "nginx", "apache", "redis", "rabbitmq"
        ]

    def extract_skills(self, text: str) -> List[str]:
        doc = nlp(text)
        # Extract entities labeled as ORG, PRODUCT, SKILL, etc.
        skills = [ent.text for ent in doc.ents if ent.label_ in ("ORG", "PRODUCT", "SKILL", "WORK_OF_ART")]
        
        # Enhanced fuzzy matching with better thresholds
        matched_skills = set()
        text_lower = text.lower()
        
        for skill in self.skill_db:
            skill_lower = skill.lower()
            # Multiple matching strategies
            if (skill_lower in text_lower or 
                fuzz.partial_ratio(skill_lower, text_lower) > 80 or
                fuzz.token_sort_ratio(skill_lower, text_lower) > 85):
                matched_skills.add(skill)
        
        # Add NER skills if they are close to known skills
        for s in skills:
            if len(s) > 2:  # Avoid very short matches
                match, score = fuzzy_process.extractOne(s, self.skill_db)
                if score > 75:  # Lower threshold for better recall
                    matched_skills.add(match)
        
        # Add common programming languages and technologies that might be missed
        tech_patterns = [
            r'\b(python|java|javascript|js|react|angular|vue|node|sql|mysql|postgresql|mongodb|aws|azure|gcp|docker|kubernetes|git|html|css|api|rest|graphql|machine learning|ml|ai|data science|pandas|numpy|scikit-learn|tensorflow|pytorch|django|flask|spring|express)\b'
        ]
        
        for pattern in tech_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if match in [s.lower() for s in self.skill_db]:
                    matched_skills.add(next(s for s in self.skill_db if s.lower() == match))
        
        return list(matched_skills)

    def classify_skill_context(self, text: str, skill: str) -> str:
        # Simple context classification: required, optional, mentioned
        text = text.lower()
        skill = skill.lower()
        if f"required: {skill}" in text or f"must have {skill}" in text:
            return "required"
        elif f"preferred: {skill}" in text or f"nice to have {skill}" in text:
            return "optional"
        elif skill in text:
            return "mentioned"
        return "none"

# ========== CORRECTED Skills Extractor ==========
class ImprovedSkillsExtractor(SkillsExtractor):
    """Enhanced skills extraction with better matching and AI/ML focus"""
    def __init__(self, skill_db: Optional[List[str]] = None):
        super().__init__(skill_db)
        
        # Enhanced skill patterns from JavaScript code
        self.skill_patterns = {
            # AI/ML Core Skills
            'llm': ['llm', 'large language model', 'language model', 'gpt', 'chatgpt'],
            'langchain': ['langchain', 'lang chain', 'langchain framework'],
            'crewai': ['crewai', 'crew ai', 'crew-ai'],
            'autogen': ['autogen', 'auto gen', 'auto-gen'],
            'openai': ['openai', 'open ai', 'gpt-4', 'gpt4', 'chatgpt', 'gpt-3'],
            'claude': ['claude', 'anthropic', 'claude-3', 'claude3'],
            'mistral': ['mistral', 'mistral-7b', 'mistral-8x7b'],
            'nlp': ['nlp', 'natural language processing', 'text processing', 'text analysis'],
            'vector_search': ['vector search', 'faiss', 'pinecone', 'embeddings', 'similarity search', 'vector database'],
            'speech_to_text': ['speech to text', 'whisper', 'speech recognition', 'asr', 'audio processing'],
            'machine_learning': ['machine learning', 'ml', 'ai', 'artificial intelligence', 'predictive modeling'],
            'transformers': ['transformers', 'bert', 'attention', 'hugging face', 'huggingface', 'transformer models'],
            'pytorch': ['pytorch', 'torch', 'pytorch lightning'],
            'tensorflow': ['tensorflow', 'tf', 'keras'],
            'python': ['python', 'python3', 'python 3'],
            'deep_learning': ['deep learning', 'neural networks', 'cnn', 'rnn', 'lstm', 'transformer'],
            
            # Technical Skills
            'api_integration': ['api', 'rest api', 'integration', 'web services', 'microservices'],
            'caching': ['caching', 'redis', 'memcached', 'cache'],
            'optimization': ['optimization', 'performance tuning', 'performance optimization'],
            'frontend': ['frontend', 'react', 'javascript', 'typescript', 'vue', 'angular'],
            'backend': ['backend', 'node.js', 'express', 'fastapi', 'flask', 'django', 'spring'],
            'databases': ['database', 'sql', 'mongodb', 'postgresql', 'mysql', 'redis'],
            'cloud': ['aws', 'azure', 'gcp', 'cloud', 'amazon web services', 'google cloud'],
            'docker': ['docker', 'containerization', 'containers'],
            'git': ['git', 'version control', 'github', 'gitlab'],
            
            # Soft Skills
            'collaboration': ['collaborate', 'team work', 'cross-functional', 'teamwork'],
            'research': ['research', 'prototyping', 'experimentation', 'r&d'],
            'problem_solving': ['problem solving', 'analytical', 'debugging', 'troubleshooting']
        }

        self.skill_relations = {
            'llm': ['machine_learning', 'nlp', 'transformers', 'openai', 'claude'],
            'langchain': ['llm', 'python', 'api_integration'],
            'vector_search': ['machine_learning', 'python', 'databases'],
            'nlp': ['machine_learning', 'python', 'transformers'],
            'machine_learning': ['python', 'pytorch', 'tensorflow', 'deep_learning']
        }
        
        # Skill importance levels
        self.high_importance = ['llm', 'langchain', 'crewai', 'nlp', 'python', 'machine_learning']
        self.medium_importance = ['vector_search', 'openai', 'claude', 'transformers', 'pytorch']

    def extract_skills(self, text: str) -> List[str]:
        """Enhanced skill extraction with pattern matching"""
        text_lower = text.lower()
        found_skills = set()
        
        # Use parent class method for basic extraction
        basic_skills = super().extract_skills(text)
        found_skills.update(basic_skills)
        
        # Enhanced pattern matching
        for skill, patterns in self.skill_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    found_skills.add(skill)
                    break
        
        # Add variations and related terms
        for skill in list(found_skills):
            if skill in self.skill_relations:
                # Add related skills that might be mentioned
                for related_skill in self.skill_relations[skill]:
                    if any(pattern in text_lower for pattern in self.skill_patterns.get(related_skill, [])):
                        found_skills.add(related_skill)
        
        return list(found_skills)

    def get_skill_importance(self, skill: str) -> str:
        """Determine skill importance level"""
        if skill in self.high_importance:
            return 'high'
        elif skill in self.medium_importance:
            return 'medium'
        return 'low'

    def calculate_skills_match(self, resume_skills: List[str], job_skills: List[str]) -> float:
        """Calculate skills match score with importance weighting"""
        if not job_skills:
            return 0.0
        
        total_score = 0.0
        weight_sum = 0.0
        
        for job_skill in job_skills:
            importance = self.get_skill_importance(job_skill)
            weight = 3 if importance == 'high' else 2 if importance == 'medium' else 1
            
            if job_skill in resume_skills:
                total_score += weight * 1.0  # Perfect match
            else:
                # Check for related skills
                related_score = self.get_related_skill_score(job_skill, resume_skills)
                total_score += weight * related_score
            
            weight_sum += weight
        
        return total_score / weight_sum if weight_sum > 0 else 0.0

    def get_related_skill_score(self, target_skill: str, available_skills: List[str]) -> float:
        """Get score for related skills when exact match not found"""
        related_skills = self.skill_relations.get(target_skill, [])
        match_count = sum(1 for skill in related_skills if skill in available_skills)
        
        return min(0.7, match_count * 0.3) if match_count > 0 else 0.0

    def generate_skills_diagnostics(self, resume_skills: List[str], job_skills: List[str]) -> Dict[str, Any]:
        """Generate diagnostic information about skills matching"""
        resume_set = set(resume_skills)
        job_set = set(job_skills)
        
        # Calculate direct matches and missing skills
        direct_matches = resume_set.intersection(job_set)
        missing_skills = job_set - resume_set
        
        # Find missing critical skills
        critical_missing = [
            skill for skill in job_skills 
            if self.get_skill_importance(skill) == 'high' and skill not in resume_set
        ]
        
        # Find related skills that could compensate
        related_compensations = {}
        for missing_skill in critical_missing:
            related = self.skill_relations.get(missing_skill, [])
            available_related = [skill for skill in related if skill in resume_set]
            if available_related:
                related_compensations[missing_skill] = available_related
        
        return {
            'skills_gap': len(job_set) - len(direct_matches),
            'critical_skills_missing': critical_missing,
            'related_compensations': related_compensations,
            'coverage_percentage': len(direct_matches) / len(job_set) if job_set else 0,
            'resume_skills_count': len(resume_skills),
            'job_skills_count': len(job_skills),
            # Add fields that frontend expects
            'direct_matches': list(direct_matches),
            'direct_match_count': len(direct_matches),
            'total_job_skills': len(job_set),
            'missing_skills': list(missing_skills)
        }

    def generate_skills_recommendations(self, diagnostics: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate recommendations based on skills diagnostics"""
        recommendations = []
        
        if diagnostics['critical_skills_missing']:
            missing_skills = diagnostics['critical_skills_missing']
            recommendations.append({
                'type': 'critical',
                'title': 'Critical Skills Gap',
                'description': f"Missing key skills: {', '.join(missing_skills)}. Consider highlighting related experience or pursuing certifications."
            })
        
        if diagnostics['coverage_percentage'] < 0.5:
            recommendations.append({
                'type': 'coverage',
                'title': 'Low Skills Coverage',
                'description': f"Only {diagnostics['coverage_percentage']*100:.1f}% of required skills found. Focus on acquiring missing core competencies."
            })
        
        if diagnostics['related_compensations']:
            comp_skills = list(diagnostics['related_compensations'].keys())
            recommendations.append({
                'type': 'compensation',
                'title': 'Related Skills Available',
                'description': f"While missing {', '.join(comp_skills)}, you have related skills that could demonstrate transferable knowledge."
            })
        
        return recommendations

# ========== CORRECTED Multi-Layer Validation ==========
class CorrectedMultiLayerValidator:
    """Fixed scoring system with proper weighting and skill matching"""
    def __init__(self):
        # Adjusted weights - LLM gets higher weight since it's most accurate
        self.similarity_weights_with_resume_bert = {
            "semantic_score": 0.20,      # Reduced - often too conservative
            "skills_score": 0.30,        # Increased - critical for tech roles
            "llm_score": 0.40,           # Increased - most comprehensive
            "resume_bert_score": 0.10    # Reduced - good but not always reliable
        }
        self.similarity_weights_without_resume_bert = {
            "semantic_score": 0.25,      # Slightly higher when no resume BERT
            "skills_score": 0.35,        # Higher importance
            "llm_score": 0.40,           # Primary scorer
            "resume_bert_score": 0.0
        }
        # Confidence calculation weights
        self.confidence_weights = {
            "semantic_score": 0.25,
            "skills_score": 0.35,
            "llm_score": 0.30,
            "resume_bert_score": 0.10
        }
    def calculate_final_similarity(self, scores: Dict[str, float]) -> Dict[str, Any]:
        """Enhanced final similarity calculation with score validation"""
        # Validate and potentially adjust individual scores
        adjusted_scores = self._validate_and_adjust_scores(scores)
        
        # Check if LLM score is high and adjust weights accordingly
        llm_score = adjusted_scores.get("llm_score", 0)
        if llm_score > 1.0:  # Convert to 0-1 range for comparison
            llm_score = llm_score / 100.0
        
        # Choose weights based on resume BERT availability and LLM score
        has_resume_bert = adjusted_scores.get("resume_bert_score") is not None
        
        # If LLM score is very high (>= 80%), give it more weight
        if llm_score >= 0.8:
            print(f"[VALIDATOR] High LLM score ({llm_score:.1%}) detected, applying LLM-dominant weighting")
            if has_resume_bert:
                weights = {
                    "semantic_score": 0.15,      # Reduced
                    "skills_score": 0.25,        # Reduced
                    "llm_score": 0.50,           # Increased significantly
                    "resume_bert_score": 0.10    # Same
                }
            else:
                weights = {
                    "semantic_score": 0.15,      # Reduced
                    "skills_score": 0.25,        # Reduced
                    "llm_score": 0.60,           # Increased significantly
                    "resume_bert_score": 0.0
                }
        else:
            # Use normal weights
            weights = (self.similarity_weights_with_resume_bert if has_resume_bert 
                      else self.similarity_weights_without_resume_bert)
        
        total_score = 0.0
        total_weight = 0.0
        used_components = []
        component_contributions = {}
        for score_type, weight in weights.items():
            if adjusted_scores.get(score_type) is not None and weight > 0:
                score_value = adjusted_scores[score_type]
                # Normalize LLM score to 0-1 range if it's in 0-100 range
                if score_type == "llm_score" and score_value > 1.0:
                    score_value = score_value / 100.0
                contribution = weight * score_value
                total_score += contribution
                total_weight += weight
                used_components.append(score_type)
                component_contributions[score_type] = {
                    'score': score_value,
                    'weight': weight,
                    'contribution': contribution
                }
        # Normalize by actual weights used
        if total_weight > 0:
            final_score = total_score / total_weight
        else:
            final_score = 0.0
        # Ensure score is in valid range
        final_score = max(0.0, min(1.0, final_score))
        return {
            "score": final_score,
            "percentage": final_score * 100,
            "weights_used": weights,
            "components_used": used_components,
            "component_contributions": component_contributions,
            "original_scores": scores,
            "adjusted_scores": adjusted_scores,
            "has_resume_bert": has_resume_bert,
            "total_weight_used": total_weight,
            "llm_dominant": llm_score >= 0.8
        }
    def _validate_and_adjust_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Validate scores and apply corrections for known issues"""
        adjusted = scores.copy()
        
        # LLM score validation and adjustment
        llm_score = scores.get("llm_score", 0)
        if llm_score > 1.0:  # Convert to 0-1 range for comparison
            llm_score = llm_score / 100.0
        
        # If LLM score is very high (>= 80%), it should dominate the final score
        if llm_score >= 0.8:
            print(f"[VALIDATOR] High LLM score detected ({llm_score:.1%}), applying dominance adjustment")
            # Boost other scores to align with LLM assessment
            if scores.get("skills_score") is not None:
                skills_score = scores["skills_score"]
                if llm_score - skills_score > 0.2:  # 20% difference threshold
                    adjustment_factor = min(0.25, (llm_score - skills_score) * 0.6)
                    adjusted["skills_score"] = min(1.0, skills_score + adjustment_factor)
                    print(f"[VALIDATOR] Adjusted skills score from {skills_score:.3f} to {adjusted['skills_score']:.3f}")
            
            if scores.get("semantic_score") is not None:
                semantic_score = scores["semantic_score"]
                if llm_score - semantic_score > 0.25:  # 25% difference threshold
                    adjustment = min(0.2, (llm_score - semantic_score) * 0.5)
                    adjusted["semantic_score"] = min(1.0, semantic_score + adjustment)
                    print(f"[VALIDATOR] Adjusted semantic score from {semantic_score:.3f} to {adjusted['semantic_score']:.3f}")
        
        # Regular validation for non-high LLM scores
        else:
            # Skills score validation and adjustment
            if scores.get("skills_score") is not None:
                skills_score = scores["skills_score"]
                # If skills score seems too low compared to LLM assessment, adjust upward
                if llm_score - skills_score > 0.15:  # 15% difference threshold
                    adjustment_factor = min(0.15, (llm_score - skills_score) * 0.5)
                    adjusted["skills_score"] = min(1.0, skills_score + adjustment_factor)
                    print(f"[VALIDATOR] Adjusted skills score from {skills_score:.3f} to {adjusted['skills_score']:.3f}")
            
            # Semantic score validation
            if scores.get("semantic_score") is not None:
                semantic_score = scores["semantic_score"]
                # If semantic score is very low but other scores are high, apply correction
                other_scores = [s for k, s in scores.items() 
                              if k != "semantic_score" and s is not None]
                if other_scores:
                    avg_other = np.mean([s if s <= 1.0 else s/100.0 for s in other_scores])
                    if avg_other - semantic_score > 0.2:  # 20% difference
                        adjustment = min(0.1, (avg_other - semantic_score) * 0.3)
                        adjusted["semantic_score"] = min(1.0, semantic_score + adjustment)
                        print(f"[VALIDATOR] Adjusted semantic score from {semantic_score:.3f} to {adjusted['semantic_score']:.3f}")
        
        return adjusted
    def get_similarity_category(self, score: float) -> str:
        """Updated categorization to be more realistic"""
        if score >= 0.85:
            return "Excellent Match (85-100%)"
        elif score >= 0.70:
            return "Good Match (70-84%)"
        elif score >= 0.55:
            return "Fair Match (55-69%)"
        elif score >= 0.40:
            return "Poor Match (40-54%)"
        else:
            return "Very Poor Match (<40%)"
    def enhanced_skills_analysis(self, resume_skills: List[str], job_skills: List[str]) -> Dict[str, Any]:
        """Detailed skills analysis to debug scoring issues"""
        resume_lower = [s.lower().strip() for s in resume_skills]
        job_lower = [s.lower().strip() for s in job_skills]
        # Direct matches
        direct_matches = set(resume_lower).intersection(set(job_lower))
        # Fuzzy matches
        fuzzy_matches = []
        for job_skill in job_lower:
            if job_skill not in direct_matches:
                for resume_skill in resume_lower:
                    if resume_skill not in direct_matches:
                        similarity = fuzz.ratio(job_skill, resume_skill)
                        if similarity >= 80:  # High similarity threshold
                            fuzzy_matches.append((job_skill, resume_skill, similarity))
        # Calculate various metrics
        total_job_skills = len(job_lower)
        direct_match_count = len(direct_matches)
        fuzzy_match_count = len(fuzzy_matches)
        # Coverage metrics
        direct_coverage = direct_match_count / total_job_skills if total_job_skills > 0 else 0
        total_coverage = (direct_match_count + fuzzy_match_count) / total_job_skills if total_job_skills > 0 else 0
        # Missing critical skills
        missing_skills = set(job_lower) - direct_matches
        for fuzzy_job, _, _ in fuzzy_matches:
            missing_skills.discard(fuzzy_job)
        return {
            "direct_matches": list(direct_matches),
            "direct_match_count": direct_match_count,
            "fuzzy_matches": fuzzy_matches,
            "fuzzy_match_count": fuzzy_match_count,
            "total_job_skills": total_job_skills,
            "direct_coverage": direct_coverage,
            "total_coverage": total_coverage,
            "missing_skills": list(missing_skills),
            "resume_skills_count": len(resume_lower),
            "skill_density": len(resume_lower) / total_job_skills if total_job_skills > 0 else 0
        }
    def confidence_score(self, scores: Dict[str, float]) -> float:
        # Weighted average for confidence calculation
        total = 0.0
        total_weight = 0.0
        for k, v in scores.items():
            if v is not None and k in self.confidence_weights:
                weight = self.confidence_weights[k]
                score_value = v if v <= 1.0 else v / 100.0  # Normalize if needed
                total += weight * score_value
                total_weight += weight
        return total / total_weight if total_weight > 0 else 0.0

    def detect_anomaly(self, scores: Dict[str, float]) -> bool:
        # Flag if scores are inconsistent
        vals = []
        for k, v in scores.items():
            if v is not None:
                # Normalize scores to 0-1 range for comparison
                normalized_v = v if v <= 1.0 else v / 100.0
                vals.append(normalized_v)
        if len(vals) < 2:
            return False
        std = float(np.std(vals))  # Convert numpy.float to Python float
        return bool(std > 0.3)  # Convert numpy.bool to Python bool

# ========== UPDATED Main Matcher Class ==========
class CorrectedResumeJobMatcher(EnhancedBERTSemanticEngine): # Inherit from EnhancedBERTSemanticEngine
    """Enhanced matcher with corrected scoring"""
    def __init__(self, groq_api_key: Optional[str] = None, cohere_api_key: Optional[str] = None, 
                 resume_bert_model: Optional[str] = None, skill_db: Optional[List[str]] = None,
                 load_specialized_models: bool = True):
        self.pdf_extractor = PDFExtractor()
        self.llm_ensemble = ImprovedLLMEnsemble(groq_api_key=groq_api_key, cohere_api_key=cohere_api_key)
        self.bert_engine = EnhancedBERTSemanticEngine(
            resume_bert_model=resume_bert_model, 
            load_specialized_models=load_specialized_models
        )
        self.skills_extractor = ImprovedSkillsExtractor(skill_db=skill_db)
        self.validator = CorrectedMultiLayerValidator()

    def match(self, resume_pdf_bytes: bytes, job_description: str) -> Dict[str, Any]:
        resume_text = self.pdf_extractor.extract_text(resume_pdf_bytes)
        resume_skills = self.skills_extractor.extract_skills(resume_text)
        job_skills = self.skills_extractor.extract_skills(job_description)
        
        # Enhanced skills analysis with diagnostics
        skills_diagnostics = self.skills_extractor.generate_skills_diagnostics(resume_skills, job_skills)
        skills_recommendations = self.skills_extractor.generate_skills_recommendations(skills_diagnostics)
        
        # Calculate scores
        ensemble_result = self.bert_engine.ensemble_resume_similarity(resume_text, job_description)
        semantic_score = self.bert_engine.semantic_similarity(resume_text, job_description)
        skills_score = self.bert_engine.skills_similarity(resume_skills, job_skills)
        resume_bert_score = self.bert_engine.resume_specific_similarity(resume_text, job_description)
        llm_result = self.llm_ensemble.get_smart_response(resume_text, job_description)
        llm_score = llm_result.get("compatibility_score", 50)
        
        # Enhanced skills matching with importance weighting
        enhanced_skills_score = self.skills_extractor.calculate_skills_match(resume_skills, job_skills)
        
        scores = {
            "semantic_score": semantic_score,
            "skills_score": skills_score,
            "enhanced_skills_score": enhanced_skills_score,
            "llm_score": llm_score,
            "resume_bert_score": resume_bert_score
        }
        
        final_similarity_result = self.validator.calculate_final_similarity(scores)
        final_score = final_similarity_result["score"]
        similarity_category = self.validator.get_similarity_category(final_score)
        confidence = self.validator.confidence_score(scores)
        anomaly = self.validator.detect_anomaly(scores)
        
        # Generate comprehensive diagnostics
        diagnostics = {
            "skills_diagnostics": skills_diagnostics,
            "skills_recommendations": skills_recommendations,
            "semantic_weakness": semantic_score < 0.3,
            "llm_discrepancy": abs((llm_score / 100) - final_score) > 0.15,
            "score_consistency": self._check_score_consistency(scores),
            "critical_gaps": skills_diagnostics.get("critical_skills_missing", []),
            "coverage_analysis": {
                "skills_coverage": skills_diagnostics.get("coverage_percentage", 0),
                "semantic_alignment": semantic_score,
                "llm_assessment": llm_score / 100
            }
        }
        
        return {
            "final_similarity_score": final_score,
            "final_similarity_percentage": final_score * 100,
            "similarity_category": similarity_category,
            "final_similarity_details": final_similarity_result,
            "skills_analysis": skills_diagnostics,
            "skills_recommendations": skills_recommendations,
            "ensemble_analysis": ensemble_result,
            "model_info": self.bert_engine.get_model_info(),
            "resume_text": resume_text,
            "resume_skills": resume_skills,
            "job_skills": job_skills,
            "semantic_score": semantic_score,
            "skills_score": skills_score,
            "enhanced_skills_score": enhanced_skills_score,
            "resume_bert_score": resume_bert_score,
            "llm_score": llm_score,
            "llm_details": llm_result,
            "confidence": confidence,
            "anomaly": anomaly,
            "component_scores": scores,
            "diagnostics": diagnostics,
            "debug_info": {
                "expected_score_range": "85-90%",
                "score_adjustments_made": final_similarity_result.get("adjusted_scores", {}),
                "primary_scoring_component": "llm_score" if llm_score > 80 else "enhanced_skills_score",
                "skills_importance_analysis": self._analyze_skills_importance(resume_skills, job_skills)
            }
        }
    
    def _check_score_consistency(self, scores: Dict[str, float]) -> Dict[str, Any]:
        """Check if component scores are consistent"""
        score_values = [v for v in scores.values() if v is not None]
        if len(score_values) < 2:
            return {"consistent": True, "std": 0.0}
        
        std = float(np.std(score_values))  # Convert numpy.float to Python float
        mean = float(np.mean(score_values))  # Convert numpy.float to Python float
        cv = float(std / mean if mean > 0 else 0)  # Convert numpy.float to Python float
        
        return {
            "consistent": bool(cv < 0.3),  # Convert numpy.bool to Python bool
            "std": std,
            "mean": mean,
            "coefficient_of_variation": cv
        }
    
    def _analyze_skills_importance(self, resume_skills: List[str], job_skills: List[str]) -> Dict[str, Any]:
        """Analyze skills by importance level"""
        resume_high = [s for s in resume_skills if self.skills_extractor.get_skill_importance(s) == 'high']
        resume_medium = [s for s in resume_skills if self.skills_extractor.get_skill_importance(s) == 'medium']
        resume_low = [s for s in resume_skills if self.skills_extractor.get_skill_importance(s) == 'low']
        
        job_high = [s for s in job_skills if self.skills_extractor.get_skill_importance(s) == 'high']
        job_medium = [s for s in job_skills if self.skills_extractor.get_skill_importance(s) == 'medium']
        job_low = [s for s in job_skills if self.skills_extractor.get_skill_importance(s) == 'low']
        
        return {
            "resume_skills_by_importance": {
                "high": resume_high,
                "medium": resume_medium,
                "low": resume_low
            },
            "job_skills_by_importance": {
                "high": job_high,
                "medium": job_medium,
                "low": job_low
            },
            "high_importance_match": len(set(resume_high).intersection(set(job_high))),
            "medium_importance_match": len(set(resume_medium).intersection(set(job_medium)))
        }

if __name__ == "__main__":
    print("=== CORRECTED Resume-Job Matcher ===")
    resume_pdf_path = input("Enter the path to the resume PDF file: ").strip()
    job_description_path = input("Enter the path to the job description text file: ").strip()
    # Automatically select the best BERT model based on similarity scores
    resume_bert_model = None
    try:
        with open(resume_pdf_path, "rb") as f:
            resume_pdf_bytes = f.read()
    except Exception as e:
        print(f"Error reading resume PDF: {e}")
        exit(1)
    try:
        with open(job_description_path, "r", encoding="utf-8") as f:
            job_description = f.read()
    except Exception as e:
        print(f"Error reading job description: {e}")
        exit(1)
    groq_api_key = os.getenv("GROQ_API_KEY")
    cohere_api_key = os.getenv("COHERE_API_KEY")
    matcher = CorrectedResumeJobMatcher(
        groq_api_key=groq_api_key,
        cohere_api_key=cohere_api_key,
        resume_bert_model=resume_bert_model
    )
    result = matcher.match(resume_pdf_bytes, job_description)
    print("\n" + "="*60)
    print("           FINAL MATCHING RESULTS (CORRECTED)")
    print("="*60)
    final_score = result["final_similarity_score"]
    final_percentage = result["final_similarity_percentage"]
    category = result["similarity_category"]
    print(f"\n🎯 FINAL SIMILARITY SCORE: {final_score:.4f} ({final_percentage:.2f}%)")
    print(f"📊 MATCH CATEGORY: {category}")
    print(f"🔍 CONFIDENCE LEVEL: {result['confidence']:.3f}")
    print(f"⚠️  ANOMALY DETECTED: {result['anomaly']}")
    print(f"\n" + "-"*50)
    print("COMPONENT BREAKDOWN:")
    print(f"├── Semantic Similarity: {result['semantic_score']:.3f} ({result['semantic_score']*100:.1f}%)")
    print(f"├── Skills Matching: {result['skills_score']:.3f} ({result['skills_score']*100:.1f}%)")
    print(f"├── Enhanced Skills Score: {result['enhanced_skills_score']:.3f} ({result['enhanced_skills_score']*100:.1f}%)")
    print(f"├── Resume-BERT Score: {result['resume_bert_score']:.3f} ({result['resume_bert_score']*100:.1f}%)")
    print(f"└── LLM Assessment: {result['llm_score']:.3f} ({result['llm_score']*100:.1f}%)")
    weights_info = result["final_similarity_details"]
    print(f"\n" + "-"*50)
    print("WEIGHTING STRATEGY:")
    if weights_info.get("llm_dominant", False):
        print("🎯 LLM-DOMINANT WEIGHTING (High LLM score detected)")
    for component, weight in weights_info["weights_used"].items():
        if weight > 0 and component in weights_info["components_used"]:
            print(f"├── {component}: {weight*100:.0f}%")
    print(f"\n" + "-"*50)
    print("📊 DETAILED PROFESSIONAL ANALYSIS")
    print("-"*50)
    
    # Enhanced Model Analysis
    model_info = result["model_info"]
    print(f"\n🤖 AI MODEL ANALYSIS:")
    print(f"├── Primary Semantic Model: {model_info.get('primary_semantic_model', 'N/A')}")
    print(f"├── Resume-Specific Model: {model_info.get('resume_specific_model', 'N/A')}")
    print(f"├── Total Models Loaded: {model_info.get('total_models_loaded', 0)}")
    print(f"└── Resume Model Available: {'✅ Yes' if model_info.get('resume_model_available') else '❌ No'}")
    
    # Best Model Selection
    ensemble_analysis = result.get("ensemble_analysis", {})
    best_model = ensemble_analysis.get("best_model", {})
    if best_model:
        print(f"\n🎯 OPTIMAL MODEL SELECTION:")
        print(f"├── Best Model: {best_model.get('model_name', 'N/A')}")
        print(f"├── Category: {best_model.get('category', 'N/A')}")
        print(f"├── Score: {best_model.get('score', 0):.4f}")
        print(f"└── All Model Scores:")
        all_scores = best_model.get('all_scores', {})
        for model, score in all_scores.items():
            print(f"    • {model}: {score:.4f}")
    
    # Enhanced LLM Analysis
    llm_details = result["llm_details"]
    print(f"\n🧠 LLM INTELLIGENCE ANALYSIS:")
    print(f"├── API Used: {llm_details.get('api_used', 'N/A')}")
    print(f"├── Response Time: {llm_details.get('response_time', 0):.2f}s")
    print(f"├── Compatibility Score: {llm_details.get('compatibility_score', 0)}/100")
    print(f"└── Analysis Quality: {'High' if llm_details.get('response_time', 0) < 5 else 'Medium'}")
    
    if llm_details.get('strengths'):
        print(f"\n💪 KEY STRENGTHS IDENTIFIED:")
        for i, strength in enumerate(llm_details['strengths'][:5], 1):
            print(f"   {i}. {strength}")
    
    if llm_details.get('gaps'):
        print(f"\n🎯 AREAS FOR IMPROVEMENT:")
        for i, gap in enumerate(llm_details['gaps'][:5], 1):
            print(f"   {i}. {gap}")
    
    if llm_details.get('recommendations'):
        print(f"\n💡 STRATEGIC RECOMMENDATIONS:")
        for i, rec in enumerate(llm_details['recommendations'][:5], 1):
            print(f"   {i}. {rec}")
    
    # Enhanced Skills Analysis
    print(f"\n🔧 ENHANCED SKILLS ANALYSIS:")
    skills_analysis = result["skills_analysis"]
    print(f"├── Skills Coverage: {skills_analysis['coverage_percentage']*100:.1f}%")
    print(f"├── Skills Gap: {skills_analysis['skills_gap']}")
    print(f"├── Resume Skills: {skills_analysis['resume_skills_count']}")
    print(f"├── Job Requirements: {skills_analysis['job_skills_count']}")
    print(f"└── Critical Skills Missing: {len(skills_analysis['critical_skills_missing'])}")
    
    # Show skills by importance
    importance_analysis = result["debug_info"]["skills_importance_analysis"]
    print(f"\n📊 SKILLS BY IMPORTANCE:")
    print(f"├── High Importance Matches: {importance_analysis['high_importance_match']}")
    print(f"├── Medium Importance Matches: {importance_analysis['medium_importance_match']}")
    print(f"├── Resume High-Value Skills: {len(importance_analysis['resume_skills_by_importance']['high'])}")
    print(f"└── Job High-Value Requirements: {len(importance_analysis['job_skills_by_importance']['high'])}")
    
    if skills_analysis['critical_skills_missing']:
        print(f"\n❌ CRITICAL SKILLS MISSING:")
        for skill in skills_analysis['critical_skills_missing']:
            print(f"   • {skill}")
    
    if skills_analysis.get('related_compensations'):
        print(f"\n🔄 RELATED SKILLS COMPENSATION:")
        for missing, related in skills_analysis['related_compensations'].items():
            print(f"   • Missing '{missing}' but have: {', '.join(related)}")
    
    # Show skills recommendations
    if result.get("skills_recommendations"):
        print(f"\n💡 SKILLS RECOMMENDATIONS:")
        for rec in result["skills_recommendations"]:
            print(f"   • {rec['title']}: {rec['description']}")
    
    # Professional Assessment
    print(f"\n📈 PROFESSIONAL ASSESSMENT:")
    confidence = result['confidence']
    anomaly = result['anomaly']
    
    if confidence > 0.8:
        confidence_level = "High"
    elif confidence > 0.6:
        confidence_level = "Medium"
    else:
        confidence_level = "Low"
    
    print(f"├── Assessment Confidence: {confidence_level} ({confidence:.1%})")
    print(f"├── Data Consistency: {'✅ Good' if not anomaly else '⚠️ Inconsistent'}")
    print(f"├── Recommendation: ", end="")
    
    if final_percentage >= 80:
        print("Strongly Recommended")
    elif final_percentage >= 65:
        print("Recommended with Minor Concerns")
    elif final_percentage >= 50:
        print("Consider with Development Plan")
    else:
        print("Not Recommended")
    
    print(f"└── Next Steps: ", end="")
    if final_percentage >= 80:
        print("Proceed to interview")
    elif final_percentage >= 65:
        print("Schedule technical assessment")
    elif final_percentage >= 50:
        print("Request additional training/certifications")
    else:
        print("Consider alternative roles or candidates")
    
    # Detailed Component Analysis
    print(f"\n🔍 DETAILED COMPONENT ANALYSIS:")
    print("-" * 50)
    print(f"📊 SEMANTIC SIMILARITY ANALYSIS:")
    print(f"   • Score: {result['semantic_score']:.3f} ({result['semantic_score']*100:.1f}%)")
    print(f"   • Model Used: {model_info.get('primary_semantic_model', 'N/A')}")
    print(f"   • Analysis: {'Strong semantic alignment' if result['semantic_score'] > 0.7 else 'Moderate alignment' if result['semantic_score'] > 0.5 else 'Weak alignment'}")
    
    print(f"\n🔧 SKILLS MATCHING ANALYSIS:")
    print(f"   • Basic Skills Score: {result['skills_score']:.3f} ({result['skills_score']*100:.1f}%)")
    print(f"   • Enhanced Skills Score: {result['enhanced_skills_score']:.3f} ({result['enhanced_skills_score']*100:.1f}%)")
    print(f"   • Skills Coverage: {skills_analysis['coverage_percentage']*100:.1f}%")
    print(f"   • Direct Matches: {skills_analysis['direct_match_count']}/{skills_analysis['total_job_skills']}")
    print(f"   • Missing Skills: {len(skills_analysis['missing_skills'])}")
    
    print(f"\n🤖 RESUME-BERT ANALYSIS:")
    print(f"   • Score: {result['resume_bert_score']:.3f} ({result['resume_bert_score']*100:.1f}%)")
    print(f"   • Model Used: {model_info.get('resume_specific_model', 'N/A')}")
    print(f"   • Analysis: {'Strong resume-specific alignment' if result['resume_bert_score'] > 0.7 else 'Moderate alignment' if result['resume_bert_score'] > 0.5 else 'Weak alignment'}")
    
    print(f"\n🧠 LLM INTELLIGENCE ANALYSIS:")
    print(f"   • Score: {result['llm_score']:.1f}/100")
    print(f"   • API Used: {llm_details.get('api_used', 'N/A')}")
    print(f"   • Response Time: {llm_details.get('response_time', 0):.2f}s")
    print(f"   • Analysis Quality: {'High' if llm_details.get('response_time', 0) < 5 else 'Medium'}")
    
    # Score Adjustments Made
    if result.get("debug_info", {}).get("score_adjustments_made"):
        print(f"\n⚙️ SCORE ADJUSTMENTS APPLIED:")
        adjustments = result["debug_info"]["score_adjustments_made"]
        for component, original_score in result["component_scores"].items():
            if component in adjustments and adjustments[component] != original_score:
                print(f"   • {component}: {original_score:.3f} → {adjustments[component]:.3f}")
    
    # Weighting Strategy Details
    print(f"\n⚖️ WEIGHTING STRATEGY DETAILS:")
    print(f"   • Strategy Used: {'LLM-Dominant' if weights_info.get('llm_dominant') else 'Standard'}")
    print(f"   • Total Weight Used: {weights_info.get('total_weight_used', 0):.2f}")
    print(f"   • Components Used: {', '.join(weights_info.get('components_used', []))}")
    
    # Model Performance Analysis
    print(f"\n🎯 MODEL PERFORMANCE ANALYSIS:")
    if best_model:
        print(f"   • Best Model: {best_model.get('model_name', 'N/A')}")
        print(f"   • Best Score: {best_model.get('score', 0):.4f}")
        print(f"   • Category: {best_model.get('category', 'N/A')}")
        print(f"   • All Model Scores:")
        all_scores = best_model.get('all_scores', {})
        for model, score in all_scores.items():
            print(f"     - {model}: {score:.4f}")
    
    # Skills Importance Analysis
    importance_analysis = result["debug_info"]["skills_importance_analysis"]
    print(f"\n📊 SKILLS IMPORTANCE BREAKDOWN:")
    print(f"   • High Importance Matches: {importance_analysis['high_importance_match']}")
    print(f"   • Medium Importance Matches: {importance_analysis['medium_importance_match']}")
    print(f"   • Resume High-Value Skills: {len(importance_analysis['resume_skills_by_importance']['high'])}")
    print(f"   • Job High-Value Requirements: {len(importance_analysis['job_skills_by_importance']['high'])}")
    
    # Confidence and Anomaly Analysis
    print(f"\n🔍 CONFIDENCE & ANOMALY ANALYSIS:")
    print(f"   • Confidence Score: {confidence:.3f}")
    print(f"   • Confidence Level: {confidence_level}")
    print(f"   • Anomaly Detected: {anomaly}")
    print(f"   • Score Consistency: {'✅ Consistent' if not anomaly else '⚠️ Inconsistent'}")
    
    print(f"\n" + "="*60)
    print("📋 EXECUTIVE SUMMARY")
    print("="*60)
    print(f"🎯 Final Score: {final_percentage:.1f}% ({category})")
    print(f"🔍 Confidence: {confidence_level}")
    print(f"⚡ Key Strength: {'Technical Skills' if result['skills_score'] > 0.7 else 'Experience' if result['resume_bert_score'] > 0.6 else 'LLM Assessment'}")
    print(f"⚠️  Primary Gap: {'Missing Skills' if skills_analysis['missing_skills'] else 'Experience Level' if result['resume_bert_score'] < 0.5 else 'Semantic Match'}")
    print("="*60)