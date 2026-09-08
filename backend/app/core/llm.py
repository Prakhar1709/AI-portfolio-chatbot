import os
import re
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from backend.app.config import settings
from backend.app.models.schemas import ChatMessage

class BaseLLMClient(ABC):
    @abstractmethod
    def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        conversation_history: List[ChatMessage] = None,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Returns dict with:
        - "text": str
        - "model": str
        - "provider": str
        """
        pass

class GeminiLLMClient(BaseLLMClient):
    def __init__(self, api_key: str = None, model_name: str = "gemini-1.5-flash"):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model_name = model_name
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        self.client = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=None
        )

    def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        conversation_history: List[ChatMessage] = None,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        import google.generativeai as genai
        
        # Build prompt with system instruction and history
        full_prompt = f"System Instructions:\n{system_prompt}\n\n"
        if conversation_history:
            full_prompt += "Conversation Context:\n"
            for msg in conversation_history[-4:]:
                full_prompt += f"{msg.role.capitalize()}: {msg.content}\n"
            full_prompt += "\n"
            
        full_prompt += f"User: {user_prompt}\nAssistant:"

        model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system_prompt,
            generation_config={"temperature": temperature}
        )
        response = model.generate_content(user_prompt)
        
        return {
            "text": response.text.strip(),
            "model": self.model_name,
            "provider": "gemini"
        }

class OpenAILLMClient(BaseLLMClient):
    def __init__(self, api_key: str = None, model_name: str = "gpt-4o-mini"):
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.model_name = model_name
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key)

    def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        conversation_history: List[ChatMessage] = None,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        messages = [{"role": "system", "content": system_prompt}]
        if conversation_history:
            for msg in conversation_history[-4:]:
                messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": user_prompt})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature
        )
        return {
            "text": response.choices[0].message.content.strip(),
            "model": self.model_name,
            "provider": "openai"
        }

class GroqLLMClient(BaseLLMClient):
    def __init__(self, api_key: str = None, model_name: str = "openai/gpt-oss-20b"):
        self.api_key = api_key or settings.GROQ_API_KEY
        self.model_name = model_name
        from openai import OpenAI
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=self.api_key
        )

    def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        conversation_history: List[ChatMessage] = None,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        messages = [{"role": "system", "content": system_prompt}]
        if conversation_history:
            for msg in conversation_history[-4:]:
                messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": user_prompt})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature
        )
        return {
            "text": response.choices[0].message.content.strip(),
            "model": self.model_name,
            "provider": "groq"
        }

class OfflineSmartPersonaClient(BaseLLMClient):
    """
    Intelligent offline synthesis engine customized for Prakhar's portfolio.
    """
    def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        conversation_history: List[ChatMessage] = None,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        # Extract context block from system prompt
        context_match = re.search(r'<RETRIEVED_PORTFOLIO_CONTEXT>(.*?)</RETRIEVED_PORTFOLIO_CONTEXT>', system_prompt, re.DOTALL)
        context_text = context_match.group(1).strip() if context_match else ""

        q_lower = user_prompt.lower()
        response_lines = []

        if "ltv" in q_lower or "retention" in q_lower or "customer" in q_lower:
            response_lines.append(f"### 📈 Customer LTV & Retention Analytics Project by **{settings.PERSONA_NAME}**:\n")
            response_lines.append("- **GitHub**: [github.com/Prakhar1709/Customer-Ltv-and-user-acquisition](https://github.com/Prakhar1709/Customer-Ltv-and-user-acquisition)")
            response_lines.append("- **Dataset**: Olist Brazilian E-commerce dataset (99K orders, 93,358 unique customers, $15.42M total revenue).")
            response_lines.append("- **Key Methodologies**: RFM Segmentation, Cohort Retention Analysis, Historical LTV Analysis, and 90-day repeat purchase prediction (`repeat_purchase_90d`).")
            response_lines.append("- **ML & XGBoost**: Handled 76.27:1 class imbalance using `scale_pos_weight`, achieved 0.6068 PR-AUC, and optimized probability threshold to `0.85`, which targeted 117 customers capturing **58.21% of repeat purchasers** and **81.73% of repeat-customer revenue** ($40,041).")
            response_lines.append("- **Dashboard**: Interactive Streamlit application for customer metrics, cohort tracking, and predictive targeting.")
        elif "fraud" in q_lower or "credit" in q_lower or "card" in q_lower:
            response_lines.append(f"### 🛡️ Credit Card Fraud Analytics & Detection Project by **{settings.PERSONA_NAME}**:\n")
            response_lines.append("- **GitHub**: [github.com/Prakhar1709/Credit-card-fraud-analytics-and-detection](https://github.com/Prakhar1709/Credit-card-fraud-analytics-and-detection)")
            response_lines.append("- **Dataset**: 284,807 transactions with 492 fraud cases (0.17% fraud rate, $60.13K fraud amount).")
            response_lines.append("- **SQL & Power BI**: Segmented transactions by amount buckets and hours; found transactions above $500 had ~0.369% fraud rate.")
            response_lines.append("- **ML & Imbalanced Learning**: Applied SMOTE (Synthetic Minority Over-sampling Technique) with Logistic Regression & Random Forest, GridSearchCV tuning, and threshold optimization.")
        elif "student" in q_lower or "performance" in q_lower or "flask" in q_lower or "indicator" in q_lower:
            response_lines.append(f"### 🎓 Student Performance Indicator by **{settings.PERSONA_NAME}**:\n")
            response_lines.append("- **GitHub**: [github.com/Prakhar1709/Student-performance-indicator](https://github.com/Prakhar1709/Student-performance-indicator)")
            response_lines.append("- **Objective**: End-to-end regression application predicting a student's Math score from demographic and academic features.")
            response_lines.append("- **Engineering Lifecycle**: Data ingestion, validation checks, EDA, modular preprocessing pipeline, and model comparison across Scikit-learn, CatBoost, and XGBoost.")
            response_lines.append("- **Production Features**: Centralized logging, custom exception handling, artifact persistence (`dill`), and interactive Flask web interface.")
        elif "project" in q_lower or "built" in q_lower or "portfolio" in q_lower:
            response_lines.append(f"Here are the core featured projects developed by **{settings.PERSONA_NAME}**:\n")
            response_lines.append("1. **Customer LTV & Retention Analytics** (`Prakhar1709/Customer-Ltv-and-user-acquisition`)\n   - E-commerce analytics on 93k customers ($15.42M revenue), RFM segmentation, cohort retention, and an XGBoost repeat-purchase model (PR-AUC 0.6068, 81.7% revenue coverage).")
            response_lines.append("2. **Credit Card Fraud Analytics & Detection** (`Prakhar1709/Credit-card-fraud-analytics-and-detection`)\n   - End-to-end fraud analytics across 284k transactions using SQL, Power BI, and SMOTE imbalanced classification.")
            response_lines.append("3. **Student Performance Indicator** (`Prakhar1709/Student-performance-indicator`)\n   - Modular ML regression lifecycle with CatBoost/XGBoost, centralized logging, exception handling, and Flask web deployment.")
        elif "skill" in q_lower or "stack" in q_lower or "tech" in q_lower or "language" in q_lower:
            response_lines.append(f"### 💻 Technical Skills & Stack for **{settings.PERSONA_NAME}**:\n")
            response_lines.append("- **Programming**: Python, SQL, C++")
            response_lines.append("- **AI & LLMs / RAG**: Large Language Models (LLMs), RAG, LangChain, LangGraph, ChromaDB, Embeddings, Prompt Engineering")
            response_lines.append("- **Machine Learning & Data Science**: Scikit-Learn, XGBoost, CatBoost, Pandas, NumPy, SMOTE Imbalanced Learning, Hyperparameter Tuning (GridSearchCV), Feature Engineering")
            response_lines.append("- **Web & App Frameworks**: FastAPI, Streamlit, Flask, REST APIs, HTML5/CSS3")
            response_lines.append("- **Business Intelligence & Visualization**: Power BI, RFM Segmentation, Cohort Retention Analysis, Matplotlib, Seaborn")
            response_lines.append("- **Engineering Practices**: Modular ML Pipelines, Centralized Logging, Custom Exception Handling, Git/GitHub")
        elif "education" in q_lower or "college" in q_lower or "university" in q_lower or "iiitdm" in q_lower or "degree" in q_lower:
            response_lines.append(f"### 🎓 Education & Background:\n")
            response_lines.append(f"- **Degree**: Bachelor of Technology (B.Tech)")
            response_lines.append(f"- **Institution**: **Indian Institute of Information Technology, Design and Manufacturing (IIITDM) Jabalpur**")
            response_lines.append(f"- **Graduation Year**: 2027")
            response_lines.append(f"- **Focus**: Machine Learning, Data Analytics, Applied Mathematics, and Generative AI systems.")
            response_lines.append(f"- **Internship**: Research Intern at **DRDO (Defence Research and Development Organisation), Delhi** (June-July 2025).")
        elif "contact" in q_lower or "hire" in q_lower or "email" in q_lower or "reach" in q_lower or "linkedin" in q_lower or "location" in q_lower:
            response_lines.append(f"### 📬 Get in Touch with {settings.PERSONA_NAME}:\n")
            response_lines.append("- **Email**: `workwithprakhar17@gmail.com`")
            response_lines.append("- **LinkedIn**: [linkedin.com/in/prakhar050/](https://www.linkedin.com/in/prakhar050/)")
            response_lines.append("- **GitHub**: [github.com/Prakhar1709](https://github.com/Prakhar1709)")
            response_lines.append("- **Location**: Faridabad, Haryana, India (Open to Remote / On-site opportunities)")
        else:
            response_lines.append(f"Based on **{settings.PERSONA_NAME}**'s portfolio knowledge base:\n")
            clean_context = re.sub(r'\[.*?\]', '', context_text)
            paragraphs = [p.strip() for p in clean_context.split('\n\n') if len(p.strip()) > 30][:3]
            for p in paragraphs:
                response_lines.append(f"- {p}")
            if not paragraphs:
                response_lines.append(f"Feel free to ask about Prakhar's projects (Customer LTV, Fraud Detection, Student Performance ML), technical skills, or education at IIITDM Jabalpur!")

        return {
            "text": "\n".join(response_lines),
            "model": "prakhar-portfolio-offline-engine",
            "provider": "offline-persona-engine"
        }

class LLMFactory:
    @staticmethod
    def get_client(provider_name: Optional[str] = None) -> BaseLLMClient:
        provider = (provider_name or settings.DEFAULT_LLM_PROVIDER).lower()

        if provider == "gemini" and settings.GEMINI_API_KEY:
            try:
                return GeminiLLMClient()
            except Exception as e:
                print(f"[LLM] Gemini client init failed ({e}), falling back.")

        if provider == "openai" and settings.OPENAI_API_KEY:
            try:
                return OpenAILLMClient()
            except Exception as e:
                print(f"[LLM] OpenAI client init failed ({e}), falling back.")

        if provider == "groq" and settings.GROQ_API_KEY:
            try:
                return GroqLLMClient()
            except Exception as e:
                print(f"[LLM] Groq client init failed ({e}), falling back.")

        return OfflineSmartPersonaClient()
