import pandas as pd
import streamlit as st
from typing import Optional, Dict, Any
import openai
import google.generativeai as genai
from .config import config

class LLMRecommender:
    def __init__(self):
        self.openai_client = None
        self.gemini_model = None
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize available LLM clients"""
        # Initialize OpenAI if API key is available
        if config.OPENAI_API_KEY:
            try:
                openai.api_key = config.OPENAI_API_KEY
                self.openai_client = openai
            except Exception as e:
                st.warning(f"Failed to initialize OpenAI: {e}")
        
        # Initialize Gemini if API key is available
        if config.GEMINI_API_KEY:
            try:
                genai.configure(api_key=config.GEMINI_API_KEY)
                self.gemini_model = genai.GenerativeModel("gemini-pro")
            except Exception as e:
                st.warning(f"Failed to initialize Gemini: {e}")
    
    def is_available(self) -> bool:
        """Check if any LLM service is available"""
        return bool(self.openai_client or self.gemini_model)
    
    def analyze_data(self, analysis_prompt: str) -> str:
        """Analyze any CSV data using available LLM"""
        if not self.is_available():
            return "No LLM API keys configured. Please set OPENAI_API_KEY or GEMINI_API_KEY in your .env file."
        
        try:
            if self.gemini_model:
                response = self.gemini_model.generate_content(analysis_prompt)
                return response.text.strip()
            elif self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a data analyst expert. Provide insights about CSV datasets."},
                        {"role": "user", "content": analysis_prompt}
                    ],
                    max_tokens=800,
                    temperature=0.7
                )
                return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error analyzing data: {str(e)}"
        
        return "No LLM service available."
    
    def analyze_column_insights(self, df: pd.DataFrame, column_name: str) -> str:
        """Analyze insights for a specific column"""
        if not self.is_available():
            return "LLM service not available for column analysis."
        
        if df.empty or column_name not in df.columns:
            return "Invalid data or column not found."
        
        # Analyze the column
        col_data = df[column_name]
        
        prompt = f"""
Analyze this column from a CSV dataset:

Column Name: {column_name}
Data Type: {col_data.dtype}
Total Values: {len(col_data):,}
Non-null Values: {col_data.count():,}
Unique Values: {col_data.nunique():,}
"""
        
        if pd.api.types.is_numeric_dtype(col_data):
            prompt += f"""
Statistics:
- Min: {col_data.min():.2f}
- Max: {col_data.max():.2f}
- Mean: {col_data.mean():.2f}
- Median: {col_data.median():.2f}
- Std Dev: {col_data.std():.2f}

Sample values: {col_data.dropna().head(10).tolist()}
"""
        elif pd.api.types.is_datetime64_any_dtype(col_data):
            prompt += f"""
Date Range: {col_data.min()} to {col_data.max()}
Sample values: {col_data.dropna().head(5).tolist()}
"""
        else:
            # Text/categorical column
            value_counts = col_data.value_counts().head(10)
            prompt += f"""
Top Values:
{value_counts.to_string()}

Sample values: {col_data.dropna().head(10).tolist()}
"""
        
        prompt += """

Please provide insights about this column:
1. What does this column likely represent?
2. Data quality observations
3. Interesting patterns or anomalies
4. Potential issues or recommendations
5. Suggestions for further analysis

Keep the analysis concise and actionable.
"""
        
        try:
            if self.gemini_model:
                response = self.gemini_model.generate_content(prompt)
                return response.text.strip()
            elif self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a data analyst. Provide insights about data columns."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=400,
                    temperature=0.7
                )
                return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating column analysis: {str(e)}"

# Create global instance
llm_recommender = LLMRecommender()