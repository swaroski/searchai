import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    def __init__(self):
        # Try Streamlit secrets first, then environment variables
        try:
            import streamlit as st
            # LLM API Keys (optional)
            self.OPENAI_API_KEY = st.secrets["general"].get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
            self.GEMINI_API_KEY = st.secrets["general"].get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
            
            # Default thresholds
            self.LOW_STOCK_THRESHOLD = int(st.secrets["general"].get("LOW_STOCK_THRESHOLD", os.getenv("LOW_STOCK_THRESHOLD", "10")))
            self.EXPIRY_WARNING_DAYS = int(st.secrets["general"].get("EXPIRY_WARNING_DAYS", os.getenv("EXPIRY_WARNING_DAYS", "30")))
            
            # Pagination
            self.ITEMS_PER_PAGE = int(st.secrets["general"].get("ITEMS_PER_PAGE", os.getenv("ITEMS_PER_PAGE", "50")))
        except:
            # Fallback to environment variables only
            self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
            self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
            
            # Default thresholds
            self.LOW_STOCK_THRESHOLD = int(os.getenv("LOW_STOCK_THRESHOLD", "10"))
            self.EXPIRY_WARNING_DAYS = int(os.getenv("EXPIRY_WARNING_DAYS", "30"))
            
            # Pagination
            self.ITEMS_PER_PAGE = int(os.getenv("ITEMS_PER_PAGE", "50"))
    
    def has_llm_api(self):
        """Check if any LLM API key is available"""
        return bool(self.OPENAI_API_KEY or self.GEMINI_API_KEY)

config = Config()