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
    
    def generate_reorder_recommendations(self, df: pd.DataFrame, 
                                       low_stock_threshold: int = 10) -> str:
        """Generate reorder recommendations using available LLM"""
        if df.empty:
            return "No inventory data available for analysis."
        
        if not self.is_available():
            return "No LLM API keys configured. Please set OPENAI_API_KEY or GEMINI_API_KEY in your .env file."
        
        # Analyze inventory data
        analysis = self._analyze_inventory(df, low_stock_threshold)
        
        # Generate recommendations using available LLM
        try:
            if self.gemini_model:
                return self._get_gemini_recommendations(analysis)
            elif self.openai_client:
                return self._get_openai_recommendations(analysis)
        except Exception as e:
            return f"Error generating recommendations: {str(e)}"
        
        return "No LLM service available."
    
    def _analyze_inventory(self, df: pd.DataFrame, low_stock_threshold: int) -> Dict[str, Any]:
        """Analyze inventory data to prepare for LLM recommendations"""
        from datetime import datetime, timedelta
        
        # Basic statistics
        total_products = len(df)
        total_value = (df['Stock'] * df['Price']).sum()
        
        # Low stock products
        low_stock_df = df[df['Stock'] <= low_stock_threshold]
        low_stock_products = low_stock_df[['Product Name', 'Category', 'Stock', 'Price']].to_dict('records')
        
        # Expiring products (next 30 days)
        cutoff_date = datetime.now() + timedelta(days=30)
        expiring_df = df[
            (df['Expiry Date'].notna()) & 
            (df['Expiry Date'] <= cutoff_date) &
            (df['Expiry Date'] >= datetime.now())
        ]
        expiring_products = expiring_df[['Product Name', 'Category', 'Stock', 'Expiry Date']].to_dict('records')
        
        # Out of stock products
        out_of_stock_df = df[df['Stock'] == 0]
        out_of_stock_products = out_of_stock_df[['Product Name', 'Category', 'Price']].to_dict('records')
        
        # Category analysis
        category_stats = df.groupby('Category').agg({
            'Stock': ['sum', 'mean'],
            'Price': 'mean',
            'Product Name': 'count'
        }).round(2)
        
        return {
            'total_products': total_products,
            'total_value': total_value,
            'low_stock_products': low_stock_products[:10],  # Limit to top 10
            'expiring_products': expiring_products[:10],
            'out_of_stock_products': out_of_stock_products[:10],
            'low_stock_threshold': low_stock_threshold,
            'categories': df['Category'].unique().tolist()
        }
    
    def _get_openai_recommendations(self, analysis: Dict[str, Any]) -> str:
        """Get recommendations using OpenAI GPT"""
        prompt = self._create_analysis_prompt(analysis)
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an inventory management expert. Provide concise, actionable reorder recommendations based on the inventory analysis."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"OpenAI API error: {str(e)}"
    
    def _get_gemini_recommendations(self, analysis: Dict[str, Any]) -> str:
        """Get recommendations using Google Gemini"""
        prompt = self._create_analysis_prompt(analysis)
        
        try:
            response = self.gemini_model.generate_content(
                f"""You are an inventory management expert. Provide concise, actionable reorder recommendations based on the inventory analysis.

{prompt}"""
            )
            
            return response.text.strip()
            
        except Exception as e:
            return f"Gemini API error: {str(e)}"
    
    def _create_analysis_prompt(self, analysis: Dict[str, Any]) -> str:
        """Create analysis prompt for LLM"""
        prompt = f"""
Inventory Analysis Summary:
- Total Products: {analysis['total_products']}
- Total Inventory Value: ${analysis['total_value']:,.2f}
- Low Stock Threshold: {analysis['low_stock_threshold']} units

OUT OF STOCK PRODUCTS ({len(analysis['out_of_stock_products'])} items):
"""
        
        for product in analysis['out_of_stock_products']:
            prompt += f"- {product['Product Name']} ({product['Category']}) - ${product['Price']:.2f}\n"
        
        prompt += f"\nLOW STOCK PRODUCTS ({len(analysis['low_stock_products'])} items):\n"
        for product in analysis['low_stock_products']:
            prompt += f"- {product['Product Name']} ({product['Category']}) - {product['Stock']} units left - ${product['Price']:.2f}\n"
        
        if analysis['expiring_products']:
            prompt += f"\nEXPIRING SOON ({len(analysis['expiring_products'])} items):\n"
            for product in analysis['expiring_products']:
                expiry_date = product['Expiry Date'].strftime('%Y-%m-%d') if hasattr(product['Expiry Date'], 'strftime') else str(product['Expiry Date'])
                prompt += f"- {product['Product Name']} ({product['Category']}) - {product['Stock']} units - Expires: {expiry_date}\n"
        
        prompt += f"\nCATEGORIES: {', '.join(analysis['categories'])}\n"
        
        prompt += """
Please provide:
1. Priority reorder recommendations (most urgent first)
2. Suggested reorder quantities based on current stock levels
3. Any patterns or insights about inventory management
4. Risk assessment for out-of-stock or expiring items

Keep recommendations concise and actionable.
"""
        
        return prompt
    
    def analyze_category_trends(self, df: pd.DataFrame, category: str) -> str:
        """Analyze trends for a specific category"""
        if not self.is_available():
            return "LLM service not available for trend analysis."
        
        if df.empty:
            return "No data available for analysis."
        
        # Filter by category
        category_df = df[df['Category'].str.contains(category, case=False, na=False)]
        
        if category_df.empty:
            return f"No products found in category: {category}"
        
        # Analyze category data
        analysis = {
            'category': category,
            'total_products': len(category_df),
            'avg_stock': category_df['Stock'].mean(),
            'avg_price': category_df['Price'].mean(),
            'total_value': (category_df['Stock'] * category_df['Price']).sum(),
            'low_stock_count': len(category_df[category_df['Stock'] <= 10]),
            'products': category_df[['Product Name', 'Stock', 'Price']].to_dict('records')[:5]
        }
        
        prompt = f"""
Category Analysis for: {analysis['category']}

Summary:
- Total Products: {analysis['total_products']}
- Average Stock Level: {analysis['avg_stock']:.1f} units
- Average Price: ${analysis['avg_price']:.2f}
- Total Category Value: ${analysis['total_value']:,.2f}
- Low Stock Items: {analysis['low_stock_count']}

Top Products:
"""
        
        for product in analysis['products']:
            prompt += f"- {product['Product Name']}: {product['Stock']} units @ ${product['Price']:.2f}\n"
        
        prompt += "\nProvide insights about this category's inventory health and recommendations."
        
        try:
            if self.gemini_model:
                response = self.gemini_model.generate_content(prompt)
                return response.text.strip()
            elif self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an inventory analyst. Provide insights about category performance."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=300,
                    temperature=0.7
                )
                return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating category analysis: {str(e)}"

# Create global instance
llm_recommender = LLMRecommender()