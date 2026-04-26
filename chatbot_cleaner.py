"""
EduGate AI Assistant System - Complete Prototype
A school AI assistant with RAG pipeline, role-based access, and LLM generation
Fully testable in Google Colab with GPU support
"""

# ============================================================================
# SECTION 1: DEPENDENCIES AND INSTALLATION
# ============================================================================

print("Installing dependencies...")
import subprocess
import sys

def install_packages():
    packages = [
        'torch', 'transformers', 'accelerate', 'bitsandbytes',
        'sentence-transformers', 'faiss-cpu', 'pyyaml', 'httpx'
    ]
    for package in packages:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])
    print("All dependencies installed successfully!")

try:
    import torch
    import faiss
except ImportError:
    install_packages()

import json
import yaml
import torch
import faiss
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ============================================================================
# SECTION 2: YAML PROMPT CONFIGURATION
# ============================================================================

PROMPT_CONFIG_YAML = """
system_role: |
  You are EduBot, a Student Support Specialist at EduGate School.
  You are friendly, professional, empathetic, and always human-like in your responses.
  You help students, teachers, parents, and administrators with their questions and needs.

  CRITICAL RULES:
  - NEVER reveal that you are an AI or a model. You are EduBot, a real support specialist.
  - NEVER mention "documents", "policies", "retrieved information", or "based on the context"
  - ALWAYS answer as if you naturally know this information from your work at the school
  - Be conversational and natural - don't sound robotic or formulaic
  - Use the provided context to inform your answer, but present it as your own knowledge

behavior_rules:
  - Always be respectful and professional
  - Use retrieved context to ground your answers but speak naturally
  - When referencing policies, do so naturally
  - If you don't have enough information, politely say so as a human would
  - Never make up information
  - Stay within school-related topics
  - Protect user privacy and data

tone_guidelines:
  policy_question:
    style: formal and authoritative but conversational
    approach: Explain policies naturally as if you're familiar with them from working at the school

  grade_inquiry:
    style: informative and encouraging
    approach: Be supportive while providing factual information, celebrate successes

  attendance:
    style: informative and encouraging
    approach: Present data clearly and offer help if needed, be understanding

  payment_info:
    style: polite and clear
    approach: Be precise with financial information but warm

  complaint:
    style: empathetic and understanding
    approach: Acknowledge feelings genuinely and offer solutions

  chitchat:
    style: friendly and warm
    approach: Keep it brief and school-appropriate, be personable

  off_topic:
    style: gentle and redirecting
    approach: Politely guide back to school-related topics

  violation:
    style: firm but polite
    approach: Clearly state boundaries and refuse inappropriate requests

response_template: |
  You are EduBot, a Student Support Specialist at EduGate School. You're having a conversation with a {role}.

  Here is relevant information from school records and policies that you have access to:
  {context}

  The {role} asks: "{question}"

  Respond naturally and conversationally as EduBot would. {tone_instruction}

  IMPORTANT:
  - Do NOT mention "documents", "policies found", "retrieved information", or "based on the context"
  - Do NOT list policy numbers or citations
  - Speak as if you naturally know this information from working at the school
  - Be warm, helpful, and human-like
  - Keep your response concise and friendly (2-4 sentences for simple queries)

  Your response:
"""

# ============================================================================
# SECTION 3: MOCK DATA (kept for schedule/materials - not used for grades/attendance)
# ============================================================================

MOCK_USERS = {
    "user001": {"userId": "user001", "name": "Alice Johnson", "role": "student", "grade": "11", "section": "A", "year": "2025"},
    "user002": {"userId": "user002", "name": "Bob Smith", "role": "teacher", "grade": None},
    "user003": {"userId": "user003", "name": "Carol White", "role": "parent", "grade": None},
    "user004": {"userId": "user004", "name": "David Brown", "role": "admin", "grade": None},
}

MOCK_MATERIALS_BY_GRADE_YEAR = {
    ("1", "2015"): ["French", "English", "Math"],
    ("10", "2024"): ["Mathematics", "Physics", "English", "History"],
    ("11", "2025"): ["Mathematics", "Physics", "Chemistry", "English", "Philosophy"],
}

MOCK_STUDENT_MATERIALS = {
    "user001": {
        "year": "2025",
        "materials": ["Mathematics", "Physics", "English", "History"]
    }
}

MOCK_SCHEDULES = {
    ("11", "A", "2025"): {
        "Monday": ["Math", "Physics", "English"],
        "Tuesday": ["Chemistry", "Math", "Philosophy"],
        "Wednesday": ["Physics", "English", "Math"],
        "Thursday": ["Chemistry", "Philosophy", "English"],
        "Friday": ["Math", "Physics", "Sports"],
    },
    ("10", "B", "2024"): {
        "Monday": ["Math", "History", "English"],
        "Tuesday": ["Physics", "Math", "English"],
        "Wednesday": ["History", "Physics", "Math"],
        "Thursday": ["English", "Math", "Sports"],
        "Friday": ["Physics", "History", "English"],
    }
}

# ============================================================================
# SECTION 4: POLICY DATABASE
# ============================================================================

POLICIES = [
    {
        "id": "1.1",
        "question": "What is the attendance policy?",
        "answer": "Students must maintain at least 90% attendance to be eligible for final examinations. Absences due to medical reasons require a doctor's note within 48 hours. More than 10 unexcused absences may result in academic probation.",
        "category": "attendance",
        "lastUpdated": "2024-09-01"
    },
    {
        "id": "1.2",
        "question": "How is grading calculated?",
        "answer": "Grading is based on: 40% midterm exams, 40% final exams, 10% assignments, and 10% class participation. The grading scale is: A (90-100), B (80-89), C (70-79), D (60-69), F (below 60).",
        "category": "grading",
        "lastUpdated": "2024-09-01"
    },
    {
        "id": "2.1",
        "question": "What is the privacy policy for student data?",
        "answer": "Student academic records are confidential and protected under FERPA. Only students, their parents/guardians, and authorized school staff can access these records. Data is encrypted and stored securely.",
        "category": "privacy",
        "lastUpdated": "2024-09-01"
    },
    {
        "id": "3.1",
        "question": "What are the tuition payment deadlines?",
        "answer": "Tuition must be paid by the first day of each semester. Late payments incur a 5% penalty after 15 days. Payment plans are available for families facing financial hardship - contact the bursar's office.",
        "category": "fees",
        "lastUpdated": "2024-09-01"
    },
    {
        "id": "4.1",
        "question": "What is the code of conduct for students?",
        "answer": "Students must respect all members of the school community, attend classes regularly, complete assignments on time, and follow dress code guidelines. Violations may result in warnings, detention, or suspension depending on severity.",
        "category": "conduct",
        "lastUpdated": "2024-09-01"
    },
    {
        "id": "5.1",
        "question": "What is the dress code policy?",
        "answer": "Students must wear the school uniform or approved alternative attire. The uniform includes a blazer, tie, and appropriate footwear. Students are not permitted to wear casual clothing such as jeans, t-shirts, or sneakers unless specifically allowed by the school administration.",
        "category": "dress_code",
        "lastUpdated": "2024-09-01"
    },
    {
        "id": "6.1",
        "question": "Who is the director of the school",
        "answer": "Charbel Chamoun is the director of the school. He is 19 years old. He is born in 16 August 2007.",
        "category": "info",
        "lastUpdated": "2024-09-01"
    }
]

# ============================================================================
# SECTION 5: ROLE-BASED FUNCTION REGISTRY
# ============================================================================

class PermissionError(Exception):
    pass

# NOTE: getGrades and getAttendance use LARAVEL_API_URL which is defined later.
# They are called only after LARAVEL_API_URL is set.

async def getGrades(token: str, role: str) -> Dict[str, Any]:
    if role not in ["student", "teacher", "admin"]:
        raise PermissionError(f"Role '{role}' is not authorized to view grades")

    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{LARAVEL_API_URL}/api/grades/my",
            headers={
                "Authorization": f"Bearer {token}",
                "ngrok-skip-browser-warning": "true"
            }
        )

    print(f"[DEBUG] getGrades status: {resp.status_code}")

    if resp.status_code != 200:
        return {"success": False, "message": f"Could not fetch grades from server (status {resp.status_code})"}

    raw = resp.json()
    grades = [
        {
            "subject": item["subject"]["name"],
            "score": item["score"],
            "max_score": item["max_score"],
            "term": item["term"],
            "date": item["date"],
        }
        for item in raw
    ]
    print(f"[DEBUG] getGrades parsed {len(grades)} grades")
    return {"success": True, "grades": grades}


async def getAttendance(token: str, role: str) -> Dict[str, Any]:
    if role not in ["student", "teacher", "admin"]:
        raise PermissionError(f"Role '{role}' is not authorized to view attendance")

    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{LARAVEL_API_URL}/api/attendance/my",
            headers={
                "Authorization": f"Bearer {token}",
                "ngrok-skip-browser-warning": "true"
            }
        )

    print(f"[DEBUG] getAttendance status: {resp.status_code}")

    if resp.status_code != 200:
        return {"success": False, "message": f"Could not fetch attendance from server (status {resp.status_code})"}

    raw = resp.json()
    total = len(raw)
    present = sum(1 for r in raw if r["status"] == "present")
    absent = total - present
    percentage = round((present / total) * 100, 1) if total > 0 else 0
    recent_absences = [r["date"] for r in raw if r["status"] == "absent"][:5]

    print(f"[DEBUG] getAttendance: total={total}, present={present}, absent={absent}, percentage={percentage}")
    return {
        "success": True,
        "attendance": {
            "total_days": total,
            "present": present,
            "absent": absent,
            "percentage": percentage,
            "recent_absences": recent_absences
        }
    }



async def getParentChildren(token: str) -> Dict[str, Any]:
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{LARAVEL_API_URL}/api/parent/children",
            headers={
                "Authorization": f"Bearer {token}",
                "ngrok-skip-browser-warning": "true"
            }
        )
    if resp.status_code != 200:
        return {"success": False, "message": "Could not fetch children"}

    raw = resp.json()
    children = [
        {
            "id": child["id"],
            "name": child["user"]["name"],
            "class": child["school_class"]["name"],
            "section": child["school_class"]["section"],
        }
        for child in raw
    ]
    return {"success": True, "children": children}


async def getGradesForChild(token: str, studentId: int) -> Dict[str, Any]:
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{LARAVEL_API_URL}/api/parent/children/{studentId}/grades",
            headers={
                "Authorization": f"Bearer {token}",
                "ngrok-skip-browser-warning": "true"
            }
        )
    if resp.status_code != 200:
        return {"success": False, "message": "Could not fetch grades"}

    raw = resp.json()
    grades = [
        {
            "subject": item["subject"]["name"],
            "score": item["score"],
            "max_score": item["max_score"],
            "term": item["term"],
        }
        for item in raw
    ]
    return {"success": True, "grades": grades}


async def getAttendanceForChild(token: str, studentId: int) -> Dict[str, Any]:
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{LARAVEL_API_URL}/api/parent/children/{studentId}/attendance",
            headers={
                "Authorization": f"Bearer {token}",
                "ngrok-skip-browser-warning": "true"
            }
        )
    if resp.status_code != 200:
        return {"success": False, "message": "Could not fetch attendance"}

    raw = resp.json()
    total = len(raw)
    present = sum(1 for r in raw if r["status"] == "present")
    absent = total - present
    percentage = round((present / total) * 100, 1) if total > 0 else 0
    recent_absences = [r["date"] for r in raw if r["status"] == "absent"][:5]

    return {
        "success": True,
        "attendance": {
            "total_days": total,
            "present": present,
            "absent": absent,
            "percentage": percentage,
            "recent_absences": recent_absences
        }
    }    


def getStudentMaterials(userId: str, role: str) -> Dict[str, Any]:
    if role != "student":
        raise PermissionError("Only students can view their materials")
    if userId in MOCK_STUDENT_MATERIALS:
        return {
            "success": True,
            "userId": userId,
            "year": MOCK_STUDENT_MATERIALS[userId]["year"],
            "materials": MOCK_STUDENT_MATERIALS[userId]["materials"]
        }
    return {"success": False, "message": "No materials found for this student"}


def getMaterialsByGradeYear(grade: str, year: str) -> Dict[str, Any]:
    key = (grade, year)
    if key in MOCK_MATERIALS_BY_GRADE_YEAR:
        return {
            "success": True,
            "grade": grade,
            "year": year,
            "materials": MOCK_MATERIALS_BY_GRADE_YEAR[key]
        }
    return {"success": False, "message": "No materials found for this grade/year"}


async def getSchedule(token: str, role: str) -> Dict[str, Any]:
    if role not in ["student", "admin", "teacher"]:
        raise PermissionError("You are not allowed to view schedules")

    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{LARAVEL_API_URL}/api/schedule/my",
            headers={
                "Authorization": f"Bearer {token}",
                "ngrok-skip-browser-warning": "true"
            }
        )

    print(f"[DEBUG] getSchedule status: {resp.status_code}")
    if resp.status_code != 200:
        return {"success": False, "message": "Could not fetch schedule"}

    raw = resp.json()
    # Convert to simple format: {"Monday": ["Math 08:00-09:30", ...], ...}
    schedule = {}
    for day, entries in raw.items():
        schedule[day] = [
            f"{e['subject']['name']} ({e['start_time']}-{e['end_time']}, {e['room']})"
            for e in entries
        ]

    return {"success": True, "schedule": schedule}

# ============================================================================
# SECTION 6: HYBRID INTENT CLASSIFIER
# ============================================================================

class IntentClassifier:
    INTENT_EXAMPLES = {
        "policy_question": [
            "What is the dress code?",
            "What can I wear to school?",
            "Are jeans allowed?",
            "What is the attendance policy?",
            "What are the school rules?",
            "Who is the director?"
        ],
        "grade_inquiry": [
            "What are my grades?",
            "Show me my results",
            "How did I do in math?",
            "What is my GPA?"
        ],
        "attendance": [
            "What is my attendance?",
            "How many days was I absent?",
            "Did I miss class?"
        ],
        "payment_info": [
            "What fees do I owe?",
            "Did I pay the tuition?",
            "Show my payments"
        ],
        "complaint": [
            "I have a problem",
            "I want to complain",
            "I'm not happy with this"
        ],
        "materials": [
            "What are my materials this year?",
            "What subjects do we have?",
            "What are the materials for grade 1 in 2015?"
        ],
        "schedule": [
            "What is my schedule?",
            "Show me my timetable",
            "What classes do I have this week?"
        ],
    }

    INTENT_KEYWORDS = {
        "policy_question": ["policy", "rule", "regulation", "guideline", "allowed", "permitted", "code of conduct"],
        "grade_inquiry": ["grade", "score", "marks", "result", "performance", "gpa", "transcript"],
        "attendance": ["attendance", "absent", "present", "attendance rate", "missing class", "attendance record"],
        "payment_info": ["payment", "fee", "tuition", "bill", "invoice", "pay", "cost", "price"],
        "complaint": ["complaint", "problem", "issue", "concern", "unhappy", "dissatisfied", "frustrated"],
        "chitchat": ["hello", "hi", "how are you", "good morning", "thanks", "thank you", "bye"],
        "violation": ["hack", "cheat", "steal", "illegal", "bypass", "exploit", "password"],
        "materials": ["material", "subject", "course", "what do we study", "what are my materials"],
        "schedule": ["schedule", "timetable", "my classes", "my periods", "class time"],
    }

    def __init__(self, llm_classifier=None):
        self.llm_classifier = llm_classifier
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def classify_by_semantic_similarity(self, message: str, threshold: float = 0.5) -> Optional[str]:
        message_emb = self.embedder.encode([message])
        best_intent = None
        best_score = 0.0
        for intent, examples in self.INTENT_EXAMPLES.items():
            example_embs = self.embedder.encode(examples)
            scores = np.dot(example_embs, message_emb[0]) / (
                np.linalg.norm(example_embs, axis=1) * np.linalg.norm(message_emb[0]) + 1e-8
            )
            max_score = float(scores.max())
            if max_score > best_score:
                best_score = max_score
                best_intent = intent
        if best_score >= threshold:
            return best_intent
        return None

    def classify_by_keywords(self, message: str) -> Optional[str]:
        message_lower = message.lower()
        for intent, keywords in self.INTENT_KEYWORDS.items():
            for keyword in keywords:
                if keyword in message_lower:
                    return intent
        return None

    def classify_with_llm(self, message: str) -> str:
        if self.llm_classifier is None:
            if "?" in message:
                return "policy_question"
            return "off_topic"
        return "off_topic"

    def classify(self, message: str) -> str:
        intent = self.classify_by_keywords(message)
        if intent:
            return intent
        intent = self.classify_by_semantic_similarity(message)
        if intent:
            return intent
        return self.classify_with_llm(message)

# ============================================================================
# SECTION 7: RAG PIPELINE
# ============================================================================

class RAGPipeline:
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        print(f"Loading embedding model: {embedding_model_name}...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.embedding_model.get_embedding_dimension()
        self.index = None
        self.documents = []
        self.metadata = []
        print("RAG pipeline initialized")

    def build_index(self, documents: List[str], metadata: List[Dict] = None):
        print(f"Building FAISS index for {len(documents)} documents...")
        self.documents = documents
        self.metadata = metadata if metadata else [{}] * len(documents)
        embeddings = self.embedding_model.encode(documents, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(embeddings)
        print(f"FAISS index built with {self.index.ntotal} vectors")

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if self.index is None:
            return []
        query_embedding = self.embedding_model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        distances, indices = self.index.search(query_embedding, top_k)
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.documents):
                results.append({
                    "document": self.documents[idx],
                    "metadata": self.metadata[idx],
                    "distance": float(distance)
                })
        return results

# ============================================================================
# SECTION 8: LLM ENGINE
# ============================================================================

class LLMEngine:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        print(f"Initializing LLM: {model_name}")

    def load_model(self):
        try:
            print("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

            print("Loading model with 4-bit quantization (this may take 1-2 minutes)...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
            print("✓ Model loaded successfully!")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise e

    def generate_from_messages(self, system_message: str, user_message: str,
                               max_new_tokens=300, temperature=0.2):
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        return self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()

    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.7) -> str:
        if self.model is None or self.tokenizer is None:
            return "Error: Model not loaded"
        try:
            messages = [
                {"role": "system", "content": "You are EduBot, a helpful and friendly school support specialist."},
                {"role": "user", "content": prompt}
            ]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            return self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        except Exception as e:
            print(f"Error during generation: {e}")
            return "I apologize, but I'm having trouble generating a response right now."

# ============================================================================
# SECTION 9: EDUBOT MAIN CLASS
# ============================================================================

class EduBot:
    def __init__(self):
        print("=" * 80)
        print("INITIALIZING EDUBOT SYSTEM")
        print("=" * 80)
        self.prompt_config = yaml.safe_load(PROMPT_CONFIG_YAML)
        self.intent_classifier = IntentClassifier()
        self.rag_pipeline = RAGPipeline()
        self.llm_engine = None
        self._build_knowledge_base()
        print("EduBot initialized successfully")

    def _build_knowledge_base(self):
        print("\nBuilding knowledge base...")
        documents = []
        metadata = []
        for policy in POLICIES:
            doc_text = f"Policy regarding {policy['category']}: {policy['question']} Answer: {policy['answer']}"
            documents.append(doc_text)
            metadata.append(policy)
        documents.append("EduBot can help with grades, attendance, payments, policies, and general school questions.")
        metadata.append({"type": "general_help"})
        self.rag_pipeline.build_index(documents, metadata)

    def load_llm(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct"):
        self.llm_engine = LLMEngine(model_name)
        self.llm_engine.load_model()

    def _retrieve_context(self, message: str, intent: str) -> str:
        results = self.rag_pipeline.search(message, top_k=2)
        if not results:
            return "No specific policy or guideline found for this query."
        context_parts = []
        for result in results:
            metadata = result['metadata']
            if 'answer' in metadata:
                context_parts.append(metadata['answer'])
            elif 'document' in result:
                context_parts.append(result['document'])
        return "\n\n".join(context_parts)

    async def _call_function(self, intent: str, token: str, role: str,
                            message: str = "") -> Optional[Dict[str, Any]]:
        functions_called = []

        try:
            # ── STUDENT ──
            if role == "student":
                if intent == "grade_inquiry":
                    result = await getGrades(token, role)
                    functions_called.append("getGrades")
                    return {"result": result, "functions": functions_called}

                elif intent == "attendance":
                    result = await getAttendance(token, role)
                    functions_called.append("getAttendance")
                    return {"result": result, "functions": functions_called}
                
                elif intent == "schedule":
                    result = await getSchedule(token, role)
                    functions_called.append("getSchedule")
                    return {"result": result, "functions": functions_called}

            # ── PARENT ──
            elif role == "parent":
                if intent in ["grade_inquiry", "attendance"]:
                    # Step 1: fetch all children
                    children_result = await getParentChildren(token)
                    functions_called.append("getParentChildren")

                    if not children_result["success"]:
                        return {"result": children_result, "functions": functions_called}

                    children = children_result["children"]

                    # Step 2: try to find a specific child by name in the message
                    message_lower = message.lower()
                    matched_child = None
                    for child in children:
                        if child["name"].lower().split()[0] in message_lower or \
                        child["name"].lower() in message_lower:
                            matched_child = child
                            break

                    # Step 3: if name found → fetch that child's data
                    if matched_child:
                        if intent == "grade_inquiry":
                            result = await getGradesForChild(token, matched_child["id"])
                            functions_called.append("getGradesForChild")
                            result["child_name"] = matched_child["name"]
                        else:
                            result = await getAttendanceForChild(token, matched_child["id"])
                            functions_called.append("getAttendanceForChild")
                            result["child_name"] = matched_child["name"]
                        return {"result": result, "functions": functions_called}

                    # Step 4: no name found → return all children list
                    return {
                        "result": {
                            "success": True,
                            "children_list": children,
                            "requested_intent": intent  
                        },
                        "functions": functions_called
                    }

        except PermissionError as e:
            return {"error": str(e), "functions": functions_called}

        return None

    def _format_prompt(self, message: str, intent: str, role: str, context: str,
                       function_result: Optional[Dict] = None) -> str:
        tone_guidelines = self.prompt_config.get('tone_guidelines', {})
        tone_info = tone_guidelines.get(intent, {})
        tone_instruction = f"Use a {tone_info.get('style', 'professional')} tone. {tone_info.get('approach', '')}"

        if function_result and 'result' in function_result:
            result_data = function_result['result']
            if result_data.get('success'):
                if 'materials' in result_data:
                    mats = result_data['materials']
                    materials_info = "Materials:\n"
                    for m in mats:
                        materials_info += f"- {m}\n"
                    context = materials_info + "\n" + context
                elif 'schedule' in result_data:
                    sched = result_data['schedule']
                    schedule_info = "Class schedule:\n"
                    for day, subjects in sched.items():
                        schedule_info += f"{day}: {', '.join(subjects)}\n"
                    context = schedule_info + "\n" + context

        elif function_result and 'error' in function_result:
            context = f"PERMISSION ISSUE: {function_result['error']}\n\n" + context

        template = self.prompt_config.get('response_template', '')
        prompt = template.format(
            context=context,
            role=role,
            question=message,
            tone_instruction=tone_instruction
        )
        system_role = self.prompt_config.get('system_role', '')
        return f"{system_role}\n\n{prompt}"

    def _format_grades_response(self, grades: list) -> str:
        from collections import defaultdict
        lines = ["Here are your current grades:\n"]
        by_subject = defaultdict(list)
        for g in grades:
            by_subject[g['subject']].append(g)
        for subject, entries in by_subject.items():
            lines.append(f"📚 {subject}:")
            for e in entries:
                lines.append(f"   • {e['term']}: {e['score']}/{e['max_score']}")
        return "\n".join(lines)

    def _format_attendance_response(self, att: dict) -> str:
        lines = [
            "Here is your attendance record:\n",
            f"📅 Total days: {att['total_days']}",
            f"✅ Present: {att['present']}",
            f"❌ Absent: {att['absent']}",
            f"📊 Attendance rate: {att['percentage']}%",
        ]
        if att.get('recent_absences'):
            lines.append(f"🗓 Recent absences: {', '.join(att['recent_absences'])}")
        if att['percentage'] >= 90:
            lines.append("\nGreat job maintaining excellent attendance! 🎉")
        else:
            lines.append("\n⚠️ Remember, 90% attendance is required for final exams.")
        return "\n".join(lines)

    def _llm_rephrase_facts(self, facts_text: str, intent: str) -> str:
        if not self.llm_engine or not self.llm_engine.model:
            return facts_text
        system_message = f"""You are a school assistant. Present ONLY the following facts to the student.
DO NOT add anything. DO NOT make up any information. DO NOT say anything not in the facts below.
If the facts contain grades, list every single one exactly as given.
If the facts contain attendance, report the exact numbers given.

FACTS (use ONLY these, nothing else):
{facts_text}"""
        user_message = f"Present these {intent} facts to the student in a friendly but precise way. Only use what is listed above."
        return self.llm_engine.generate_from_messages(
            system_message=system_message,
            user_message=user_message,
            max_new_tokens=400,
            temperature=0.1
        )

    async def handle_chat(self, requesterId: str, role: str, targetUserId: str,
                          message: str, token: str) -> Dict[str, Any]:
        print(f"\n[EduBot] Processing message from {role} ({requesterId}): {message}")

        # Step 1: Classify intent
        intent = self.intent_classifier.classify(message)
        print(f"[Intent] Classified as: {intent}")

        # Step 2: Handle violations
        if intent == "violation":
            return {
                "text": "I'm sorry, but I cannot assist with that request. As EduBot, I'm here to help with legitimate school-related questions and support. Please ask me about grades, attendance, policies, or other educational topics.",
                "intent": intent,
                "functionsCalled": []
            }

        # Step 3: Handle chitchat
        if intent == "chitchat":
            chitchat_responses = {
                "hello": "Hello! I'm EduBot, your Student Support Specialist. How can I assist you today?",
                "hi": "Hi there! I'm here to help with any school-related questions you have.",
                "thanks": "You're very welcome! Feel free to reach out anytime you need assistance.",
                "thank you": "My pleasure! I'm always here to help.",
                "bye": "Goodbye! Have a great day, and don't hesitate to return if you need anything."
            }
            for key, response in chitchat_responses.items():
                if key in message.lower():
                    return {"text": response, "intent": intent, "functionsCalled": []}

        # Step 4: Retrieve context
        context = self._retrieve_context(message, intent)
        print(f"[RAG] Retrieved context from knowledge base")

        # Step 5: Scope check
        if requesterId != targetUserId and role != "admin" and role != "parent":
            return {
                "text": "Sorry, you're not allowed to view other users' private information.",
                "intent": intent,
                "functionsCalled": []
            }

        # Step 6: Call functions
        function_result = await self._call_function(intent, token, role, message)

        if function_result and 'error' in function_result:
            friendly_map = {
                "grade_inquiry": "grades",
                "attendance": "attendance records"
            }
            thing = friendly_map.get(intent, "this information")
            return {
                "text": f"Sorry, you're not allowed to view {thing}. Please contact the school administration if you think this is a mistake.",
                "intent": intent,
                "functionsCalled": []
            }

        functions_called = function_result.get('functions', []) if function_result else []

        # Step 7: Handle off-topic
        if intent == "off_topic":
            return {
                "text": "I appreciate your message! However, I'm EduBot, and I'm here specifically to help with school-related questions like grades, attendance, policies, and payments. Is there anything related to your education or school experience I can help you with?",
                "intent": intent,
                "functionsCalled": functions_called
            }
        

        # Handle parent children list (when no specific child named)
        if function_result and "result" in function_result:
            result_data = function_result["result"]
            if result_data.get("children_list"):
                children = result_data["children_list"]
                intent_word = "grades" if result_data.get("requested_intent") == "grade_inquiry" else "attendance"
                names = [f"• {c['name']} ({c['class']} {c['section']})" for c in children]
                return {
                    "text": f"You have {len(children)} children enrolled:\n" + "\n".join(names) +
                            f"\n\nWhich child's {intent_word} would you like to see? Just mention their name!",
                    "intent": intent,
                    "functionsCalled": function_result.get("functions", [])
                }

        # Step 8: Handle grades
        if intent == "grade_inquiry" and function_result and "result" in function_result:
            result_data = function_result["result"]
            if result_data.get("success") and "grades" in result_data:
                grades = result_data.get("grades", [])
                child_name = result_data.get("child_name", "")
                from collections import defaultdict
                by_subject = defaultdict(list)
                for g in grades:
                    by_subject[g['subject']].append(g)
                header = f"Here are {child_name}'s grades:\n" if child_name else "Here are your current grades:\n"
                lines = [header]
                for subject, entries in by_subject.items():
                    lines.append(f"📚 {subject}:")
                    for e in entries:
                        lines.append(f"   • {e['term']}: {e['score']}/{e['max_score']}")
                return {"text": "\n".join(lines), "intent": intent, "functionsCalled": functions_called}
            else:
                return {
                    "text": f"Sorry, I couldn't retrieve the grades right now. {result_data.get('message', '')}",
                    "intent": intent,
                    "functionsCalled": functions_called
                }

        # Step 9: Handle attendance
        if intent == "attendance" and function_result and "result" in function_result:
            result_data = function_result["result"]
            if result_data.get("success") and "attendance" in result_data:
                att = result_data["attendance"]
                child_name = result_data.get("child_name", "")
                header = f"Here is {child_name}'s attendance record:\n" if child_name else "Here is your attendance record:\n"
                lines = [
                    header,
                    f"📅 Total days: {att['total_days']}",
                    f"✅ Present: {att['present']}",
                    f"❌ Absent: {att['absent']}",
                    f"📊 Attendance rate: {att['percentage']}%",
                ]
                if att.get('recent_absences'):
                    lines.append(f"🗓 Recent absences: {', '.join(att['recent_absences'])}")
                if att['percentage'] >= 90:
                    lines.append("\nGreat job maintaining excellent attendance! 🎉")
                else:
                    lines.append("\n⚠️ Remember, 90% attendance is required for final exams.")
                return {"text": "\n".join(lines), "intent": intent, "functionsCalled": functions_called}
            else:
                return {
                    "text": f"Sorry, I couldn't retrieve the attendance right now. {result_data.get('message', '')}",
                    "intent": intent,
                    "functionsCalled": functions_called
                }

        # Step 10: Handle schedule
        if intent == "schedule" and function_result and "result" in function_result:
            result_data = function_result["result"]
            if result_data.get("success"):
                sched = result_data.get("schedule", {})
                facts_lines = [f"- {day}: {', '.join(subjects)}" for day, subjects in sched.items()]
                facts_text = "\n".join(facts_lines)
                friendly_text = self._llm_rephrase_facts(facts_text, intent)
                return {"text": friendly_text, "intent": intent, "functionsCalled": functions_called}

        # Step 11: Handle materials
        if intent == "materials" and function_result and "result" in function_result:
            result_data = function_result["result"]
            if result_data.get("success"):
                mats = result_data.get("materials", [])
                facts_lines = [f"- {m}" for m in mats]
                facts_text = "\n".join(facts_lines)
                friendly_text = self._llm_rephrase_facts(facts_text, intent)
                return {"text": friendly_text, "intent": intent, "functionsCalled": functions_called}

        # Step 12: LLM for everything else (policy questions, complaints, etc.)
        if self.llm_engine and self.llm_engine.model:
            print("[LLM] 🤖 Generating natural, human-like response...")
            prompt = self._format_prompt(message, intent, role, context, function_result)
            response_text = self.llm_engine.generate(prompt, max_new_tokens=300, temperature=0.7)
            print("[LLM] ✓ Response generated")
        else:
            print("[Fallback] ⚠ Using pre-written response (LLM not loaded)")
            response_text = self._generate_fallback_response(intent, context, function_result)

        return {"text": response_text, "intent": intent, "functionsCalled": functions_called}

    def _generate_fallback_response(self, intent: str, context: str,
                                    function_result: Optional[Dict]) -> str:
        if function_result and 'error' in function_result:
            return f"I apologize, but I'm unable to access that information. If you believe you should have access to this, please contact the administration office."

        if not function_result and context:
            if intent == "policy_question":
                return f"Great question! {context}\n\nLet me know if you need any clarification!"

        return "I'd be happy to help! Could you please provide more details about your question?"

# ============================================================================
# SECTION 10: FASTAPI APP
# ============================================================================

from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import Response
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import httpx

app = FastAPI(title="EduGate API", description="School AI Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ⚠️ IMPORTANT: Update this with your Laravel ngrok URL every session
LARAVEL_API_URL = "https://ef57-80-77-189-240.ngrok-free.app"

# Initialize EduBot globally
edubot = EduBot()
edubot.load_llm("Qwen/Qwen2.5-3B-Instruct")


class ChatRequest(BaseModel):
    targetUserId: Optional[str] = None
    message: str


class ChatResponse(BaseModel):
    text: str
    intent: str
    functionsCalled: list


async def get_authenticated_user(authorization: str) -> dict:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.replace("Bearer ", "")
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(
                f"{LARAVEL_API_URL}/api/me",
                headers={
                    "Authorization": f"Bearer {token}",
                    "ngrok-skip-browser-warning": "true"
                }
            )
        except Exception:
            raise HTTPException(status_code=503, detail="Could not reach Laravel auth service")
    if resp.status_code != 200:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return resp.json()


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, authorization: str = Header(...)):
    try:
        auth_user = await get_authenticated_user(authorization)
        requester_id = str(auth_user["id"])
        role = auth_user["role"]
        target_user_id = request.targetUserId or requester_id

        response = await edubot.handle_chat(
            requesterId=requester_id,
            role=role,
            targetUserId=target_user_id,
            message=request.message,
            token=authorization.replace("Bearer ", "")  # ← extract here
        )
        return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.options("/chat")
async def options_chat():
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization, ngrok-skip-browser-warning",
        }
    )

# ============================================================================
# SECTION 11: MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    try:
        import google.colab
        IN_COLAB = True
    except:
        IN_COLAB = False

    if IN_COLAB:
        print("Detected Google Colab environment")

    print("==========================================")
    print("🔐 NGROK AUTHENTICATION REQUIRED")
    print("Get your token from: https://dashboard.ngrok.com/get-started/your-authtoken")
    print("==========================================")

    ngrok_token = input("👉 Enter your ngrok authtoken: ").strip()
    if not ngrok_token:
        raise ValueError("❌ No ngrok token provided. Restart and enter your token.")

    laravel_url = input("👉 Enter your Laravel ngrok URL (e.g. https://xxxx.ngrok-free.app): ").strip()
    if not laravel_url:
        raise ValueError("❌ No Laravel URL provided.")
    LARAVEL_API_URL = laravel_url

    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "fastapi", "uvicorn", "pyngrok", "nest_asyncio"])

    from pyngrok import ngrok
    import nest_asyncio
    import uvicorn
    import threading

    ngrok.set_auth_token(ngrok_token)
    nest_asyncio.apply()

    public_url = ngrok.connect(8000)
    print("\n==========================================")
    print("🌍 Public API URL:", public_url.public_url)
    print("➡️  Postman endpoint: POST", public_url.public_url + "/chat")
    print("==========================================\n")

    def run_server():
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    print("✅ FastAPI server is running in background.")
    print("🚀 You can now send requests from Postman or the React app.")

else:
    print("EduBot module loaded.")

# ============================================================================
# END OF FILE
# ============================================================================
