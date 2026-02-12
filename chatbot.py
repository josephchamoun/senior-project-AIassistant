"""
EduGate AI Assistant System - Complete Prototype
A school AI assistant with RAG pipeline, role-based access, and LLM generation
Fully testable in Google Colab with GPU support
"""

# ============================================================================
# SECTION 1: DEPENDENCIES AND INSTALLATION
# ============================================================================

# Install required packages
print("Installing dependencies...")
import subprocess
import sys

def install_packages():
    packages = [
        'torch',
        'transformers',
        'accelerate',
        'bitsandbytes',
        'sentence-transformers',
        'faiss-cpu',
        'pyyaml'
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
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
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
  - When referencing policies, do so naturally (e.g., "According to our school policy..." or "Our attendance requirements state...")
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
  - If citing a policy, do it naturally (e.g., "Our school policy requires..." or "According to our guidelines...")
  - Keep your response concise and friendly (2-4 sentences for simple queries)

  Your response:
"""

# ============================================================================
# SECTION 3: MOCK DATA STRUCTURES
# ============================================================================

# TODO: Replace this with Laravel DB or API call
MOCK_USERS = {
    "user001": {"userId": "user001", "name": "Alice Johnson", "role": "student", "grade": "11", "section": "A", "year": "2025"},
    "user002": {"userId": "user002", "name": "Bob Smith", "role": "teacher", "grade": None},
    "user003": {"userId": "user003", "name": "Carol White", "role": "parent", "grade": None},
    "user004": {"userId": "user004", "name": "David Brown", "role": "admin", "grade": None},
}


# TODO: Replace this with Laravel DB or API call
MOCK_GRADES = {
    "user001": [
        {"subject": "Mathematics", "grade": "A", "score": 92, "semester": "Fall 2024"},
        {"subject": "English", "grade": "B+", "score": 87, "semester": "Fall 2024"},
        {"subject": "Physics", "grade": "A-", "score": 90, "semester": "Fall 2024"},
        {"subject": "History", "grade": "B", "score": 85, "semester": "Fall 2024"},
    ]
}

# TODO: Replace this with Laravel DB or API call
MOCK_ATTENDANCE = {
    "user001": {
        "total_days": 120,
        "present": 112,
        "absent": 8,
        "percentage": 93.3,
        "recent_absences": ["2024-11-15", "2024-11-22", "2024-12-03"]
    }
}

# TODO: Replace this with Laravel DB or API call
MOCK_PAYMENTS = {
    "user003": [
        {"description": "Tuition Fee - Fall 2024", "amount": 5000, "status": "Paid", "due_date": "2024-09-01"},
        {"description": "Lab Fee", "amount": 200, "status": "Paid", "due_date": "2024-09-15"},
        {"description": "Tuition Fee - Spring 2025", "amount": 5000, "status": "Pending", "due_date": "2025-01-15"},
    ]
}

# TODO: Replace this with Laravel DB or API call
MOCK_LEARNING_MATERIALS = [
    {"id": "mat001", "title": "Algebra Basics", "subject": "Mathematics", "type": "pdf"},
    {"id": "mat002", "title": "Shakespeare Guide", "subject": "English", "type": "video"},
    {"id": "mat003", "title": "Physics Lab Manual", "subject": "Physics", "type": "pdf"},
]


# =========================
# MOCK MATERIALS BY GRADE & YEAR
# =========================

MOCK_MATERIALS_BY_GRADE_YEAR = {
    ("1", "2015"): ["French", "English", "Math"],
    ("10", "2024"): ["Mathematics", "Physics", "English", "History"],
    ("11", "2025"): ["Mathematics", "Physics", "Chemistry", "English", "Philosophy"],
}

# =========================
# MOCK STUDENT CURRENT MATERIALS
# =========================

MOCK_STUDENT_MATERIALS = {
    "user001": {  # Alice Johnson
        "year": "2025",
        "materials": ["Mathematics", "Physics", "English", "History"]
    }
}

# =========================
# MOCK SCHEDULES BY (GRADE, SECTION, YEAR)
# =========================

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
# SECTION 4: POLICY DATABASE (GROUND TRUTH)
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
        "category": "dress_code",
        "lastUpdated": "2024-09-01"
    }
]

# ============================================================================
# SECTION 5: ROLE-BASED FUNCTION REGISTRY
# ============================================================================

class PermissionError(Exception):
    """Custom exception for permission violations"""
    pass

def getGrades(userId: str, role: str) -> Dict[str, Any]:
    """
    Retrieve grades for a user with role-based access control

    TODO: Replace with Laravel DB or API call

    Args:
        userId: The user ID requesting grades
        role: The role of the requester (student, teacher, parent, admin)

    Returns:
        Dictionary containing grade information

    Raises:
        PermissionError: If user doesn't have permission to view grades
    """
    # Check permissions
    if role not in ["student", "teacher", "admin"]:
        raise PermissionError(f"Role '{role}' is not authorized to view grades")

    # TODO: Replace with actual database query
    if userId in MOCK_GRADES:
        return {
            "success": True,
            "userId": userId,
            "grades": MOCK_GRADES[userId]
        }
    else:
        return {
            "success": False,
            "message": "No grade information found for this user"
        }

def getAttendance(userId: str, role: str) -> Dict[str, Any]:
    """
    Retrieve attendance for a user with role-based access control

    TODO: Replace with Laravel DB or API call

    Args:
        userId: The user ID requesting attendance
        role: The role of the requester (student, teacher, parent, admin)

    Returns:
        Dictionary containing attendance information

    Raises:
        PermissionError: If user doesn't have permission to view attendance
    """
    # Check permissions
    if role not in ["student", "teacher", "admin"]:
        raise PermissionError(f"Role '{role}' is not authorized to view attendance")

    # TODO: Replace with actual database query
    if userId in MOCK_ATTENDANCE:
        return {
            "success": True,
            "userId": userId,
            "attendance": MOCK_ATTENDANCE[userId]
        }
    else:
        return {
            "success": False,
            "message": "No attendance information found for this user"
        }

def getPayments(userId: str, role: str) -> Dict[str, Any]:
    """
    Retrieve payment information with role-based access control

    TODO: Replace with Laravel DB or API call

    Args:
        userId: The user ID requesting payment info
        role: The role of the requester (parent, admin)

    Returns:
        Dictionary containing payment information

    Raises:
        PermissionError: If user doesn't have permission to view payments
    """
    # Check permissions
    if role not in ["parent", "admin"]:
        raise PermissionError(f"Role '{role}' is not authorized to view payment information")

    # TODO: Replace with actual database query
    if userId in MOCK_PAYMENTS:
        return {
            "success": True,
            "userId": userId,
            "payments": MOCK_PAYMENTS[userId]
        }
    else:
        return {
            "success": False,
            "message": "No payment information found for this user"
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


def getSchedule(grade: str, section: str, year: str, role: str) -> Dict[str, Any]:
    if role not in ["student", "admin", "teacher"]:
        raise PermissionError("You are not allowed to view schedules")

    key = (grade, section, year)
    if key in MOCK_SCHEDULES:
        return {
            "success": True,
            "grade": grade,
            "section": section,
            "year": year,
            "schedule": MOCK_SCHEDULES[key]
        }

    return {"success": False, "message": "No schedule found for this class"}

# ============================================================================
# SECTION 6: HYBRID INTENT CLASSIFIER
# ============================================================================

class IntentClassifier:
    """
    Hybrid intent classifier using keyword matching and LLM fallback
    """
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

    # Keyword patterns for each intent
    INTENT_KEYWORDS = {
        "policy_question": ["policy", "rule", "regulation", "guideline", "allowed", "permitted", "code of conduct"],
        "grade_inquiry": ["grade", "score", "marks", "result", "performance", "GPA", "transcript"],
        "attendance": ["attendance", "absent", "present", "attendance rate", "missing class"],
        "payment_info": ["payment", "fee", "tuition", "bill", "invoice", "pay", "cost", "price"],
        "complaint": ["complaint", "problem", "issue", "concern", "unhappy", "dissatisfied", "frustrated"],
        "chitchat": ["hello", "hi", "how are you", "good morning", "thanks", "thank you", "bye"],
        "violation": ["hack", "cheat", "steal", "illegal", "bypass", "exploit", "password"],
        "materials": ["material", "subject", "course", "what do we study", "what are my materials"],
        "schedule": ["schedule", "timetable", "my classes", "my periods", "class time"],

    }

    def __init__(self, llm_classifier=None):
        """
        Initialize the intent classifier

        Args:
            llm_classifier: Optional LLM-based classifier for fallback
        """
        self.llm_classifier = llm_classifier
        from sentence_transformers import SentenceTransformer

        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def classify_by_semantic_similarity(self, message: str, threshold: float = 0.5) -> Optional[str]:
        message_emb = self.embedder.encode([message])

        best_intent = None
        best_score = 0.0

        for intent, examples in self.INTENT_EXAMPLES.items():
            example_embs = self.embedder.encode(examples)

            # cosine similarity
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
        """
        Classify intent using keyword matching

        Args:
            message: User message

        Returns:
            Intent string or None if no match
        """
        message_lower = message.lower()

        # Check each intent's keywords
        for intent, keywords in self.INTENT_KEYWORDS.items():
            for keyword in keywords:
                if keyword in message_lower:
                    return intent

        return None

    def classify_with_llm(self, message: str) -> str:
        """
        Classify intent using LLM when keywords don't match

        Args:
            message: User message

        Returns:
            Intent string
        """
        # Simple heuristic fallback if no LLM available
        if self.llm_classifier is None:
            # Check if it's a question
            if "?" in message:
                return "policy_question"
            # Default to off_topic
            return "off_topic"

        # TODO: Implement LLM-based classification if needed
        return "off_topic"

    def classify(self, message: str) -> str:
        """
        Main classification method using hybrid approach

        Args:
            message: User message

        Returns:
            Classified intent
        """
        # 1) Try keyword (fast, cheap)
        intent = self.classify_by_keywords(message)
        if intent:
            return intent

        # 2) Try semantic similarity (smart)
        intent = self.classify_by_semantic_similarity(message)
        if intent:
            return intent

        # 3) Fallback to LLM or heuristic
        return self.classify_with_llm(message)

# ============================================================================
# SECTION 7: RAG PIPELINE - EMBEDDINGS AND FAISS
# ============================================================================

class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline using sentence-transformers and FAISS
    """

    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize RAG pipeline

        Args:
            embedding_model_name: Name of the sentence-transformer model
        """
        print(f"Loading embedding model: {embedding_model_name}...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

        # FAISS index
        self.index = None
        self.documents = []
        self.metadata = []

        print("RAG pipeline initialized")

    def build_index(self, documents: List[str], metadata: List[Dict] = None):
        """
        Build FAISS index from documents

        Args:
            documents: List of text documents to index
            metadata: Optional metadata for each document
        """
        print(f"Building FAISS index for {len(documents)} documents...")

        # Store documents
        self.documents = documents
        self.metadata = metadata if metadata else [{}] * len(documents)

        # Generate embeddings
        embeddings = self.embedding_model.encode(documents, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')

        # Create FAISS index
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(embeddings)

        print(f"FAISS index built with {self.index.ntotal} vectors")

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for relevant documents using similarity search

        Args:
            query: Query text
            top_k: Number of top results to return

        Returns:
            List of dictionaries containing document and metadata
        """
        if self.index is None:
            return []

        # Encode query
        query_embedding = self.embedding_model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')

        # Search FAISS index
        distances, indices = self.index.search(query_embedding, top_k)

        # Prepare results
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
# SECTION 8: LLM CONFIGURATION AND LOADING
# ============================================================================

class LLMEngine:
    """
    LLM engine using HuggingFace transformers with 4-bit quantization
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct"):
        """
        Initialize LLM with 4-bit quantization for Colab GPU

        Args:
            model_name: HuggingFace model name
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None

        print(f"Initializing LLM: {model_name}")

    def load_model(self):
        """
        Load model with 4-bit quantization
        """
        try:
            print("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Configure 4-bit quantization
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
                               max_new_tokens=150, temperature=0.2):
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

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
        """
        Generate text using the LLM

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        if self.model is None or self.tokenizer is None:
            return "Error: Model not loaded"

        try:
            # Format prompt for chat models
            messages = [
                {"role": "system", "content": "You are EduBot, a helpful and friendly school support specialist."},
                {"role": "user", "content": prompt}
            ]

            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.model.device)

            # Generate
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

            # Decode
            generated_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

            return generated_text.strip()

        except Exception as e:
            print(f"Error during generation: {e}")
            return "I apologize, but I'm having trouble generating a response right now."

# ============================================================================
# SECTION 9: EDUBOT ASSISTANT - MAIN HANDLER
# ============================================================================

class EduBot:
    """
    Main EduBot assistant class integrating all components
    """

    def __init__(self):
        """
        Initialize EduBot with all components
        """
        print("=" * 80)
        print("INITIALIZING EDUBOT SYSTEM")
        print("=" * 80)

        # Load prompt configuration
        self.prompt_config = yaml.safe_load(PROMPT_CONFIG_YAML)

        # Initialize components
        self.intent_classifier = IntentClassifier()
        self.rag_pipeline = RAGPipeline()
        self.llm_engine = None  # Will be loaded separately due to size

        # Build knowledge base
        self._build_knowledge_base()

        print("EduBot initialized successfully")

    def _build_knowledge_base(self):
        """
        Build FAISS index from policies and other documents
        """
        print("\nBuilding knowledge base...")

        # Prepare policy documents for indexing
        documents = []
        metadata = []

        for policy in POLICIES:
            # Index both question and answer for better retrieval
            doc_text = f"Policy regarding {policy['category']}: {policy['question']} Answer: {policy['answer']}"
            documents.append(doc_text)
            metadata.append(policy)

        # Add general help information
        # TODO: Add more learning materials and guidelines
        documents.append("EduBot can help with grades, attendance, payments, policies, and general school questions.")
        metadata.append({"type": "general_help"})

        # Build FAISS index
        self.rag_pipeline.build_index(documents, metadata)

    def load_llm(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct"):
        """
        Load LLM model (separate method due to size)

        Args:
            model_name: HuggingFace model name
        """
        self.llm_engine = LLMEngine(model_name)
        self.llm_engine.load_model()

    def _retrieve_context(self, message: str, intent: str) -> str:
        """
        Retrieve relevant context from knowledge base

        Args:
            message: User message
            intent: Classified intent

        Returns:
            Context text for the LLM
        """
        # Retrieve relevant documents
        results = self.rag_pipeline.search(message, top_k=2)

        if not results:
            return "No specific policy or guideline found for this query."

        # Build context - just the raw information without policy IDs
        context_parts = []

        for result in results:
            metadata = result['metadata']
            if 'answer' in metadata:
                # Just include the policy answer without the ID
                context_parts.append(metadata['answer'])
            elif 'document' in result:
                context_parts.append(result['document'])

        context_text = "\n\n".join(context_parts)
        return context_text

    def _call_function(self, intent: str, userId: str, role: str) -> Optional[Dict[str, Any]]:
        """
        Call appropriate function based on intent and role

        Args:
            intent: User intent
            userId: User ID
            role: User role

        Returns:
            Function result or None
        """
        functions_called = []

        try:
            if intent == "grade_inquiry":
                result = getGrades(userId, role)
                functions_called.append("getGrades")
                return {"result": result, "functions": functions_called}

            elif intent == "attendance":
                result = getAttendance(userId, role)
                functions_called.append("getAttendance")
                return {"result": result, "functions": functions_called}

            elif intent == "payment_info":
                result = getPayments(userId, role)
                functions_called.append("getPayments")
                return {"result": result, "functions": functions_called}
            
            elif intent == "materials":
                # If student asking about themselves
                if role == "student":
                    result = getStudentMaterials(userId, role)
                    functions_called.append("getStudentMaterials")
                    return {"result": result, "functions": functions_called}

            elif intent == "schedule":
                user = MOCK_USERS.get(userId)
                if not user:
                    return {"error": "User not found", "functions": functions_called}

                grade = user["grade"]
                section = user["section"]
                year = user["year"]

                result = getSchedule(grade, section, year, role)
                functions_called.append("getSchedule")
                return {"result": result, "functions": functions_called}


        except PermissionError as e:
            return {"error": str(e), "functions": functions_called}

        return None

    def _format_prompt(self, message: str, intent: str, role: str, context: str, function_result: Optional[Dict] = None) -> str:
        """
        Format the final prompt for the LLM

        Args:
            message: User message
            intent: Classified intent
            role: User role
            context: Retrieved context
            function_result: Optional function call result

        Returns:
            Formatted prompt
        """
        # Get tone instruction based on intent
        tone_guidelines = self.prompt_config.get('tone_guidelines', {})
        tone_info = tone_guidelines.get(intent, {})
        tone_instruction = f"Use a {tone_info.get('style', 'professional')} tone. {tone_info.get('approach', '')}"

        # Add function result to context if available
        if function_result and 'result' in function_result:
            result_data = function_result['result']
            if result_data.get('success'):
                # Format the data naturally for the LLM
                if 'grades' in result_data:
                    grades_info = "Student's current grades:\n"
                    for grade in result_data['grades']:
                        grades_info += f"- {grade['subject']}: {grade['grade']} ({grade['score']}%) - {grade['semester']}\n"
                    context = grades_info + "\n" + context

                elif 'attendance' in result_data:
                    att = result_data['attendance']
                    attendance_info = f"Student's attendance record:\n"
                    attendance_info += f"- Total days: {att['total_days']}\n"
                    attendance_info += f"- Days present: {att['present']}\n"
                    attendance_info += f"- Days absent: {att['absent']}\n"
                    attendance_info += f"- Attendance rate: {att['percentage']}%\n"
                    if att.get('recent_absences'):
                        attendance_info += f"- Recent absences: {', '.join(att['recent_absences'])}\n"
                    context = attendance_info + "\n" + context

                elif 'payments' in result_data:
                    payments_info = "Payment records:\n"
                    for payment in result_data['payments']:
                        payments_info += f"- {payment['description']}: ${payment['amount']} ({payment['status']}) - Due: {payment['due_date']}\n"
                    context = payments_info + "\n" + context
                elif 'materials' in result_data:
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

        # Format using template
        template = self.prompt_config.get('response_template', '')
        prompt = template.format(
            context=context,
            role=role,
            question=message,
            tone_instruction=tone_instruction
        )

        # Add system role
        system_role = self.prompt_config.get('system_role', '')
        full_prompt = f"{system_role}\n\n{prompt}"

        return full_prompt
    
        # Helper function
    def _llm_rephrase_facts(self, facts_text: str, intent: str) -> str:

        if not self.llm_engine or not self.llm_engine.model:
            return facts_text

        system_message = f"""
    You are a school assistant.

    The student asked about their **{intent}**.

    CRITICAL RULES:
    - You MUST use ALL facts below.
    - You MUST NOT omit any item.
    - You MUST NOT add anything not listed.

    FACTS:
    {facts_text}
    """


        user_message = "Answer the student in a natural way."


        return self.llm_engine.generate_from_messages(
            system_message=system_message,
            user_message=user_message,
            max_new_tokens=300,
            temperature=0.1
        )



    def handle_chat(self, requesterId: str, role: str, targetUserId: str, message: str) -> Dict[str, Any]:

        """
        Main chat handling function

        Args:
            requesterId: Requester ID
            targetUserId: Target user ID
            role: User role (student, teacher, parent, admin)
            message: User message
            role: User role (student, teacher, parent, admin)
            message: User message

        Returns:
            Response dictionary with text, intent, and functions called
        """
        print(f"\n[EduBot] Processing message from {role} ({requesterId}) about ({targetUserId}): {message}")


        # Step 1: Classify intent
        intent = self.intent_classifier.classify(message)
        print(f"[Intent] Classified as: {intent}")

        # Step 2: Handle violations immediately
        if intent == "violation":
            return {
                "text": "I'm sorry, but I cannot assist with that request. As EduBot, I'm here to help with legitimate school-related questions and support. Please ask me about grades, attendance, policies, or other educational topics.",
                "intent": intent,
                "functionsCalled": []
            }

        # Step 3: Handle chitchat briefly
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
                    return {
                        "text": response,
                        "intent": intent,
                        "functionsCalled": []
                    }

        # Step 4: Retrieve context from RAG
        context = self._retrieve_context(message, intent)
        print(f"[RAG] Retrieved context from knowledge base")
        # 🔐 Scope check: only admin can access other users
        if requesterId != targetUserId and role != "admin":
            return {
                "text": "Sorry, you’re not allowed to view other users’ private information.",
                "intent": intent,
                "functionsCalled": []
            }

        # Step 5: Call functions if needed
        function_result = self._call_function(intent, targetUserId, role)

        if function_result and 'error' in function_result:
            friendly_map = {
                "payment_info": "payment information",
                "grade_inquiry": "grades",
                "attendance": "attendance records"
            }

            thing = friendly_map.get(intent, "this information")

            return {
                "text": f"Sorry, you’re not allowed to view {thing}. Please contact the school administration if you think this is a mistake.",
                "intent": intent,
                "functionsCalled": []
            }

        functions_called = function_result.get('functions', []) if function_result else []

        # Step 6: Handle off-topic
        if intent == "off_topic":
            return {
                "text": "I appreciate your message! However, I'm EduBot, and I'm here specifically to help with school-related questions like grades, attendance, policies, and payments. Is there anything related to your education or school experience I can help you with?",
                "intent": intent,
                "functionsCalled": functions_called
            }
        if intent == "schedule" and function_result and "result" in function_result:
            result_data = function_result["result"]
            if result_data.get("success"):
                sched = result_data.get("schedule", {})

                # Build FACTS text
                facts_lines = []
                for day, subjects in sched.items():
                    facts_lines.append(f"- {day}: {', '.join(subjects)}")

                facts_text = "\n".join(facts_lines)

                # Ask LLM to rephrase (wording only)
                friendly_text = self._llm_rephrase_facts(facts_text, intent)


                return {
                    "text": friendly_text,
                    "intent": intent,
                    "functionsCalled": functions_called
                }

        if intent == "materials" and function_result and "result" in function_result:
            result_data = function_result["result"]
            if result_data.get("success"):
                mats = result_data.get("materials", [])

                facts_lines = [f"- {m}" for m in mats]
                facts_text = "\n".join(facts_lines)

                friendly_text = self._llm_rephrase_facts(facts_text, intent)

                return {
                    "text": friendly_text,
                    "intent": intent,
                    "functionsCalled": functions_called
                }


        # Step 7: Generate response using LLM (ALWAYS for non-chitchat/non-violation/non-off-topic)
        if self.llm_engine and self.llm_engine.model:
            print("[LLM] 🤖 Generating natural, human-like response...")
            prompt = self._format_prompt(message, intent, role, context, function_result)
            response_text = self.llm_engine.generate(prompt, max_new_tokens=300, temperature=0.7)
            print("[LLM] ✓ Response generated")
        else:
            print("[Fallback] ⚠ Using pre-written response (LLM not loaded)")
            # Fallback response without LLM
            response_text = self._generate_fallback_response(intent, context, function_result)

        # Step 8: Return structured response
        return {
            "text": response_text,
            "intent": intent,
            "functionsCalled": functions_called
        }

    def _generate_fallback_response(self, intent: str, context: str, function_result: Optional[Dict]) -> str:
        """
        Generate a response without LLM (for testing without model loaded)

        Args:
            intent: Classified intent
            context: Retrieved context
            function_result: Function call result

        Returns:
            Fallback response text
        """
        # Note: This is a simple fallback - the LLM will make this much more natural
        response_parts = []

        if function_result:
            if 'error' in function_result:
                return f"I apologize, but I'm unable to access that information. {function_result['error']} If you believe you should have access to this, please contact the administration office."

            if 'result' in function_result:
                result_data = function_result['result']
                if intent == "grade_inquiry" and result_data.get('success'):
                    grades = result_data['grades']
                    response_parts.append("I've pulled up your current grades. Here's how you're doing:")
                    for grade in grades:
                        response_parts.append(f"• {grade['subject']}: {grade['grade']} ({grade['score']}%)")
                    response_parts.append("\nYou're doing great! Keep up the good work.")

                elif intent == "attendance" and result_data.get('success'):
                    attendance = result_data['attendance']
                    response_parts.append(f"Looking at your attendance record, you're at {attendance['percentage']}% attendance rate. ")
                    response_parts.append(f"That's {attendance['present']} days present out of {attendance['total_days']} total days. ")
                    if attendance['percentage'] >= 90:
                        response_parts.append("Great job maintaining excellent attendance!")
                    else:
                        response_parts.append("Remember, we require 90% attendance to be eligible for final exams. Let me know if you need any support!")

                elif intent == "payment_info" and result_data.get('success'):
                    payments = result_data['payments']
                    response_parts.append("Here's your current payment status:")
                    for payment in payments:
                        status_emoji = "✓" if payment['status'] == 'Paid' else "⏳"
                        response_parts.append(f"{status_emoji} {payment['description']}: ${payment['amount']} ({payment['status']})")
                    response_parts.append("\nIf you have any questions about payments, feel free to ask!")

        if not response_parts and context:
            # For policy questions
            if intent == "policy_question":
                response_parts.append("Great question! Let me explain our policy on that.")
                response_parts.append(f"\n{context}")
                response_parts.append("\nLet me know if you need any clarification on this!")

        return "\n".join(response_parts) if response_parts else "I'd be happy to help! Could you please provide more details about your question?"

# ============================================================================
# SECTION 10: API-READY STRUCTURE (COMMENTED FOR FUTURE USE)
# ============================================================================


# Uncomment this section to enable FastAPI endpoint

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="EduGate API", description="School AI Assistant API")

# Initialize EduBot globally
edubot = EduBot()
edubot.load_llm("Qwen/Qwen2.5-3B-Instruct")

class ChatRequest(BaseModel):
    requesterId: str
    role: str
    targetUserId: str
    message: str


class ChatResponse(BaseModel):
    text: str
    intent: str
    functionsCalled: list

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        response = edubot.handle_chat(
            requesterId=request.requesterId,
            role=request.role,
            targetUserId=request.targetUserId,
            message=request.message
        )
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# To run in Colab with ngrok:
# !pip install fastapi uvicorn pyngrok
# from pyngrok import ngrok
# import nest_asyncio
# import uvicorn
#
# nest_asyncio.apply()
# ngrok_tunnel = ngrok.connect(8000)
# print('Public URL:', ngrok_tunnel.public_url)
# uvicorn.run(app, host='0.0.0.0', port=8000)


# ============================================================================
# SECTION 11: TESTING AND CLI INTERFACE FOR COLAB
# ============================================================================

def run_colab_test():
    """
    Interactive testing interface for Google Colab
    """
    print("\n" + "=" * 80)
    print("EDUBOT - INTERACTIVE TESTING MODE")
    print("=" * 80)

    # Initialize EduBot
    edubot = EduBot()

    # Ask if user wants to load LLM
    print("\nDo you want to load the LLM model?")
    print("OPTIONS:")
    print("1. 'yes' - Load Qwen 2.5 3B model (recommended, fast, works without auth)")
    print("2. 'llama' - Load Llama 3.1 8B (requires HuggingFace auth)")
    print("3. 'no' - Use fallback responses (quick testing)")

    load_llm = input("\nYour choice (yes/llama/no): ").strip().lower()

    if load_llm == 'yes':
        print("\n🚀 Loading Qwen 2.5 3B Instruct model (no authentication required)...")
        print("This model is excellent for conversational AI and works great in Colab!")
        try:
            edubot.load_llm("Qwen/Qwen2.5-3B-Instruct")
        except Exception as e:
            print(f"\n✗ Error loading LLM: {e}")
            print("Continuing with fallback responses...")
            import traceback
            traceback.print_exc()

    elif load_llm == 'llama':
        print("\nLoading Llama 3.1 8B model...")
        print("Note: This requires HuggingFace authentication.")
        print("Make sure you have:")
        print("1. Accepted the Llama model license on HuggingFace")
        print("2. Run: huggingface-cli login")

        try:
            edubot.load_llm("meta-llama/Meta-Llama-3.1-8B-Instruct")
        except Exception as e:
            print(f"\n✗ Error loading LLM: {e}")
            print("Continuing with fallback responses...")
    else:
        print("\nUsing fallback responses (no LLM generation)")

    # Select test user
    print("\n" + "-" * 80)
    print("SELECT A TEST USER:")
    print("-" * 80)
    for user_id, user_data in MOCK_USERS.items():
        print(f"{user_id}: {user_data['name']} ({user_data['role']})")

    selected_user_id = input("\nEnter user ID (or press Enter for user001): ").strip() or "user001"
    selected_user = MOCK_USERS.get(selected_user_id, MOCK_USERS["user001"])

    print(f"\nTesting as: {selected_user['name']} ({selected_user['role']})")
    print("-" * 80)

    # Example queries to try
    print("\nEXAMPLE QUERIES TO TRY:")
    print("- What's my grade in Mathematics?")
    print("- Can you tell me about my grades?")
    print("- What is the attendance policy?")
    print("- Show me my attendance record")
    print("- What are the payment deadlines?")
    print("- Tell me about the privacy policy")
    print("- I have a complaint about my grade")
    print("- Hello, how are you?")
    print("- Tell me about the weather")
    print("- Type 'quit' to exit")
    print("-" * 80)

    # Interactive loop
    while True:
        user_message = input(f"\n[{selected_user['name']}]: ").strip()

        if not user_message:
            continue

        if user_message.lower() in ['quit', 'exit', 'q']:
            print("\nThank you for testing EduBot! Goodbye.")
            break

        # Process message
        try:
            response = edubot.handle_chat(
                userId=selected_user_id,
                role=selected_user['role'],
                message=user_message
            )

            # Display response
            print(f"\n[EduBot]: {response['text']}")
            print(f"\n[Debug Info]")
            print(f"  Intent: {response['intent']}")
            print(f"  Functions Called: {response['functionsCalled']}")
            print(f"  LLM Status: {'✓ Active (Human-like responses)' if edubot.llm_engine and edubot.llm_engine.model else '✗ Using Fallback (Pre-written)'}")

        except Exception as e:
            print(f"\n[Error]: {str(e)}")
            import traceback
            traceback.print_exc()

# ============================================================================
# SECTION 12: MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Check if running in Colab
    try:
        import google.colab
        IN_COLAB = True
    except:
        IN_COLAB = False

    if IN_COLAB:
        print("Detected Google Colab environment")
        print("\n" + "=" * 80)

    # Disable CLI mode
    # run_colab_test()

    # ==============================
    # 🚀 API MODE (Colab + Ngrok)
    # ==============================

    print("==========================================")
    print("🔐 NGROK AUTHENTICATION REQUIRED")
    print("Please paste your ngrok authtoken below.")
    print("Get it from: https://dashboard.ngrok.com/get-started/your-authtoken")
    print("==========================================")

    ngrok_token = input("👉 Enter your ngrok authtoken: ").strip()

    if not ngrok_token:
        raise ValueError("❌ No ngrok token provided. Restart and enter your token.")

    # Install API + tunnel tools (safe to run multiple times in Colab)
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "fastapi", "uvicorn", "pyngrok", "nest_asyncio"])

    from pyngrok import ngrok
    import nest_asyncio
    import uvicorn
    import asyncio

    # Configure ngrok
    ngrok.set_auth_token(ngrok_token)

    # Allow nested event loop in Colab
    nest_asyncio.apply()

    # Open tunnel
    public_url = ngrok.connect(8000)
    print("\n==========================================")
    print("🌍 Public API URL:", public_url.public_url)
    print("➡️ Postman endpoint: POST", public_url.public_url + "/chat")
    print("==========================================\n")

    # ------------------------------
    # Start FastAPI server (Colab-safe)
    # ------------------------------
    # ------------------------------
  # Start FastAPI server (Colab-safe, threaded)
  # ------------------------------
    import threading

    def run_server():
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    print("✅ FastAPI server is running in background.")
    print("🚀 You can now send requests from Postman.")


else:
    print("EduBot module loaded. Use run_colab_test() to start interactive testing.")

# ============================================================================
# END OF FILE
# ============================================================================