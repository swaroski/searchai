import streamlit as st
import pandas as pd
import sys
import os
from datetime import datetime, timedelta
import io

# Add app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.data_loader import data_loader
from app.llm_recommender import llm_recommender
from app.config import config

# Configure Streamlit page
st.set_page_config(
    page_title="CSV Data Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "csv_df" not in st.session_state:
    st.session_state.csv_df = pd.DataFrame()

if "filtered_df" not in st.session_state:
    st.session_state.filtered_df = pd.DataFrame()

if "current_page" not in st.session_state:
    st.session_state.current_page = 1

def reset_page():
    """Reset pagination to first page"""
    st.session_state.current_page = 1

def display_data_summary(df):
    """Display data summary statistics"""
    if df.empty:
        return
    
    summary = data_loader.get_data_summary(df)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rows", f"{summary['total_rows']:,}")
        st.metric("Total Columns", summary['total_columns'])
    
    with col2:
        st.metric("Numeric Columns", summary['numeric_columns'])
        st.metric("Text Columns", summary['text_columns'])
    
    with col3:
        st.metric("Date Columns", summary['date_columns'])
        memory_mb = summary['memory_usage'] / (1024 * 1024)
        st.metric("Memory Usage", f"{memory_mb:.1f} MB")
    
    with col4:
        # Show column names
        with st.expander("📋 Column Names"):
            for col in summary['column_names']:
                st.write(f"• {col}")

def display_data_table(df, page=1, items_per_page=50):
    """Display paginated data table"""
    if df.empty:
        st.info("No data to display.")
        return
    
    # Paginate data
    paginated_df, page_info = data_loader.paginate_dataframe(df, page, items_per_page)
    
    # Format data for display (basic formatting)
    display_df = paginated_df.copy()
    
    # Format numeric columns to reasonable precision
    for col in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[col]):
            display_df[col] = display_df[col].round(2)
        elif pd.api.types.is_datetime64_any_dtype(display_df[col]):
            display_df[col] = display_df[col].dt.strftime('%Y-%m-%d %H:%M')
    
    # Display table
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )
    
    # Pagination controls
    if page_info['total_pages'] > 1:
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if st.button("← Previous", disabled=(page <= 1)):
                st.session_state.current_page = max(1, page - 1)
                st.rerun()
        
        with col2:
            st.write(f"Page {page_info['current_page']} of {page_info['total_pages']} "
                    f"({page_info['start_item']}-{page_info['end_item']} of {page_info['total_items']} items)")
        
        with col3:
            if st.button("Next →", disabled=(page >= page_info['total_pages'])):
                st.session_state.current_page = min(page_info['total_pages'], page + 1)
                st.rerun()

def create_analysis_prompt(df: pd.DataFrame) -> str:
    """Create a prompt for LLM analysis of the data"""
    summary = data_loader.get_data_summary(df)
    column_info = data_loader.get_column_info(df)
    
    prompt = f"""
Please analyze this CSV dataset:

Dataset Overview:
- Total Rows: {summary['total_rows']:,}
- Total Columns: {summary['total_columns']}
- Numeric Columns: {summary['numeric_columns']}
- Text Columns: {summary['text_columns']}
- Date Columns: {summary['date_columns']}

Column Details:
"""
    
    for col_name, info in column_info.items():
        prompt += f"\n{col_name} ({info['dtype']}):"
        prompt += f" {info['non_null_count']:,} non-null values, {info['unique_count']:,} unique"
        
        if 'min' in info and 'max' in info:
            prompt += f", range: {info['min']:.2f} to {info['max']:.2f}"
        elif 'min_date' in info and 'max_date' in info:
            prompt += f", date range: {info['min_date']} to {info['max_date']}"
        elif 'unique_values' in info:
            top_values = list(info['unique_values'].keys())[:5]
            prompt += f", top values: {', '.join(map(str, top_values))}"
    
    prompt += f"""

Sample Data (first 5 rows):
{df.head().to_string()}

Please provide:
1. What type of data this appears to be (e.g., sales, survey, scientific, etc.)
2. Key insights about the data structure and quality
3. Interesting patterns or anomalies you notice
4. Suggestions for analysis or visualization
5. Data quality issues and recommendations

Keep the analysis concise and actionable.
"""
    
    return prompt

def main():
    st.title("📊 CSV Data Explorer")
    st.markdown("Upload any CSV file and explore your data with powerful search, filtering, and AI-powered insights.")
    
    # Sidebar for file upload and settings
    with st.sidebar:
        st.header("📁 Data Management")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV File",
            type=['csv'],
            help="Upload any CSV file to explore and analyze your data"
        )
        
        if uploaded_file is not None:
            with st.spinner("Loading data..."):
                df = data_loader.load_csv(uploaded_file)
                
                if not df.empty:
                    st.session_state.csv_df = df
                    st.session_state.filtered_df = df.copy()
                    reset_page()
                    st.success(f"✅ Loaded {len(df):,} rows successfully!")
                    
                    # Show a preview of the loaded data
                    with st.expander("📊 Data Preview", expanded=False):
                        st.dataframe(df.head(), use_container_width=True)
        
        # Clear data button
        if not st.session_state.csv_df.empty:
            if st.button("🗑️ Clear Data", help="Remove all loaded data and reset"):
                st.session_state.csv_df = pd.DataFrame()
                st.session_state.filtered_df = pd.DataFrame()
                st.rerun()
        
        # Settings
        st.header("⚙️ Settings")
        items_per_page = st.selectbox(
            "Items per Page",
            [25, 50, 100, 200],
            index=1,
            help="Number of items to display per page"
        )
    
    # Main content area
    if st.session_state.csv_df.empty:
        st.info("👆 Please upload a CSV file to get started!")
        
        st.subheader("🎯 What You Can Do")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **📊 Data Exploration:**
            - View data summary and statistics
            - Search across all text columns
            - Filter by column values
            - Sort by any column
            - Export filtered results
            """)
        
        with col2:
            st.markdown("""
            **🤖 AI-Powered Analysis:**
            - Get intelligent data insights
            - Identify patterns and anomalies
            - Receive analysis suggestions
            - Data quality assessment
            """)
        
        st.subheader("📋 Supported Data Types")
        st.markdown("""
        - **Any CSV format** - No specific column requirements
        - **Automatic data type detection** - Numbers, dates, text
        - **Smart data cleaning** - Handles common formatting issues
        - **Large files supported** - Efficient pagination and filtering
        """)
        
        return
    
    # Display data summary
    display_data_summary(st.session_state.csv_df)
    
    st.divider()
    
    # Search and filter section
    st.subheader("🔍 Search & Filter")
    
    columns_list = st.session_state.csv_df.columns.tolist()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_term = st.text_input(
            "Search Data",
            placeholder="Enter search term...",
            help="Search across all text columns"
        )
    
    with col2:
        # Column selection for filtering
        filter_column = st.selectbox(
            "Filter by Column",
            ["-- No Filter --"] + columns_list,
            help="Select a column to filter by specific values"
        )
    
    with col3:
        # Sort options
        sort_column = st.selectbox(
            "Sort by Column", 
            ["-- No Sort --"] + columns_list,
            help="Select a column to sort the data"
        )
    
    # Additional filter controls based on selected column
    filter_value = None
    filter_operation = 'equals'
    
    if filter_column != "-- No Filter --":
        col_info = data_loader.get_column_info(st.session_state.csv_df)
        col_dtype = col_info[filter_column]['dtype']
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'int' in col_dtype or 'float' in col_dtype:
                # Numeric column
                filter_operation = st.selectbox(
                    "Operation",
                    ['equals', 'greater_than', 'less_than', 'greater_equal', 'less_equal', 'not_equals']
                )
                min_val = float(st.session_state.csv_df[filter_column].min())
                max_val = float(st.session_state.csv_df[filter_column].max())
                filter_value = st.number_input(f"Value for {filter_column}", 
                                             min_value=min_val, max_value=max_val, 
                                             value=min_val)
            else:
                # Text or other column
                filter_operation = st.selectbox(
                    "Operation",
                    ['equals', 'contains', 'not_equals']
                )
                unique_values = data_loader.get_unique_values(st.session_state.csv_df, filter_column, 50)
                if len(unique_values) <= 20:
                    filter_value = st.selectbox(f"Value for {filter_column}", unique_values)
                else:
                    filter_value = st.text_input(f"Value for {filter_column}")
    
    # Filter buttons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔍 Apply Filters", type="primary"):
            df = st.session_state.csv_df.copy()
            
            # Apply search
            if search_term:
                df = data_loader.search_data(df, search_term)
            
            # Apply column filter
            if filter_column != "-- No Filter --" and filter_value is not None:
                df = data_loader.filter_by_column_value(df, filter_column, filter_value, filter_operation)
            
            # Apply sorting
            if sort_column != "-- No Sort --":
                df = data_loader.sort_dataframe(df, sort_column)
            
            st.session_state.filtered_df = df
            reset_page()
    
    with col2:
        if st.button("🔄 Reset Filters"):
            st.session_state.filtered_df = st.session_state.csv_df.copy()
            reset_page()
    
    with col3:
        # Show column info
        if st.button("📊 Column Info"):
            st.session_state.show_column_info = True
    
    with col4:
        # Quick stats
        if st.button("📈 Quick Stats"):
            st.session_state.show_quick_stats = True
    
    # Display column info if requested
    if hasattr(st.session_state, 'show_column_info') and st.session_state.show_column_info:
        st.subheader("📊 Column Information")
        column_info = data_loader.get_column_info(st.session_state.csv_df)
        
        for col_name, info in column_info.items():
            with st.expander(f"{col_name} ({info['dtype']})"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Non-null count:** {info['non_null_count']:,}")
                    st.write(f"**Unique values:** {info['unique_count']:,}")
                
                with col2:
                    if 'min' in info:
                        st.write(f"**Range:** {info['min']:.2f} to {info['max']:.2f}")
                        st.write(f"**Mean:** {info['mean']:.2f}")
                    elif 'unique_values' in info:
                        st.write("**Top values:**")
                        for val, count in list(info['unique_values'].items())[:5]:
                            st.write(f"• {val}: {count}")
        
        st.session_state.show_column_info = False
    
    # Display quick stats if requested
    if hasattr(st.session_state, 'show_quick_stats') and st.session_state.show_quick_stats:
        st.subheader("📈 Quick Statistics")
        
        numeric_cols = st.session_state.csv_df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            st.write("**Numeric Columns Summary:**")
            st.dataframe(st.session_state.csv_df[numeric_cols].describe())
        
        text_cols = st.session_state.csv_df.select_dtypes(include=['object']).columns
        if len(text_cols) > 0:
            st.write("**Text Columns Summary:**")
            st.dataframe(st.session_state.csv_df[text_cols].describe())
        
        st.session_state.show_quick_stats = False
    
    st.divider()
    
    # Display results
    result_count = len(st.session_state.filtered_df)
    st.subheader(f"📊 Results ({result_count:,} rows)")
    
    if result_count > 0:
        # Export button
        col1, col2 = st.columns([1, 4])
        with col1:
            csv_data = data_loader.export_to_csv(st.session_state.filtered_df)
            st.download_button(
                label="📥 Export CSV",
                data=csv_data,
                file_name=f"filtered_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        # Display table
        display_data_table(st.session_state.filtered_df, st.session_state.current_page, items_per_page)
    
    st.divider()
    
    # AI Analysis section
    st.subheader("🤖 AI-Powered Data Analysis")
    
    if llm_recommender.is_available():
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎯 Analyze Dataset", type="primary"):
                with st.spinner("Analyzing your data..."):
                    analysis_prompt = create_analysis_prompt(st.session_state.csv_df)
                    analysis = llm_recommender.analyze_data(analysis_prompt)
                    st.session_state.data_analysis = analysis
        
        with col2:
            if st.button("📊 Column Insights"):
                selected_col = st.selectbox("Select column to analyze", st.session_state.csv_df.columns, key="col_insight")
                if selected_col:
                    with st.spinner(f"Analyzing {selected_col} column..."):
                        col_analysis = llm_recommender.analyze_column_insights(st.session_state.csv_df, selected_col)
                        st.session_state.column_analysis = col_analysis
        
        # Display analysis results
        if hasattr(st.session_state, 'data_analysis'):
            st.subheader("💡 Data Analysis Results")
            st.markdown(st.session_state.data_analysis)
        
        if hasattr(st.session_state, 'column_analysis'):
            st.subheader("📊 Column Analysis Results")
            st.markdown(st.session_state.column_analysis)
    
    else:
        st.info("🔧 AI analysis is not available. Please set OPENAI_API_KEY or GEMINI_API_KEY in your .env file to enable AI-powered features.")

if __name__ == "__main__":
    main()