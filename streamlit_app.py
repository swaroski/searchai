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
    page_title="Inventory Search Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "inventory_df" not in st.session_state:
    st.session_state.inventory_df = pd.DataFrame()

if "filtered_df" not in st.session_state:
    st.session_state.filtered_df = pd.DataFrame()

if "current_page" not in st.session_state:
    st.session_state.current_page = 1

def reset_page():
    """Reset pagination to first page"""
    st.session_state.current_page = 1

def display_inventory_summary(df):
    """Display inventory summary statistics"""
    if df.empty:
        return
    
    summary = data_loader.get_inventory_summary(df)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Products", summary['total_products'])
        st.metric("Total Stock", f"{summary['total_stock']:,}")
    
    with col2:
        st.metric("Total Value", f"${summary['total_value']:,.2f}")
        st.metric("Categories", summary['categories_count'])
    
    with col3:
        st.metric("Low Stock Items", summary['low_stock_count'])
        st.metric("Expiring Soon", summary['expiring_count'])
    
    with col4:
        st.metric("Already Expired", summary['expired_count'])
        st.metric("Avg Price", f"${summary['avg_price']:.2f}")

def display_data_table(df, page=1, items_per_page=50):
    """Display paginated data table"""
    if df.empty:
        st.info("No data to display.")
        return
    
    # Paginate data
    paginated_df, page_info = data_loader.paginate_dataframe(df, page, items_per_page)
    
    # Format data for display
    display_df = paginated_df.copy()
    display_df['Price'] = display_df['Price'].apply(lambda x: f"${x:.2f}")
    display_df['Expiry Date'] = display_df['Expiry Date'].dt.strftime('%Y-%m-%d')
    display_df['Stock'] = display_df['Stock'].astype(str)
    
    # Display table
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Product Name": st.column_config.TextColumn("Product Name", width="medium"),
            "Category": st.column_config.TextColumn("Category", width="small"),
            "Stock": st.column_config.TextColumn("Stock", width="small"),
            "Price": st.column_config.TextColumn("Price", width="small"),
            "Expiry Date": st.column_config.TextColumn("Expiry Date", width="small")
        }
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

def main():
    st.title("📦 Inventory Search Dashboard")
    st.markdown("Upload and manage your inventory data with smart search and AI-powered recommendations.")
    
    # Sidebar for file upload and settings
    with st.sidebar:
        st.header("📁 Data Management")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Inventory CSV",
            type=['csv'],
            help="Upload a CSV file with columns: Product Name, Category, Stock, Price, Expiry Date"
        )
        
        if uploaded_file is not None:
            with st.spinner("Loading inventory data..."):
                df = data_loader.load_csv(uploaded_file)
                if not df.empty:
                    st.session_state.inventory_df = df
                    st.session_state.filtered_df = df.copy()
                    reset_page()
                    st.success(f"Loaded {len(df)} products successfully!")
        
        # Settings
        st.header("⚙️ Settings")
        low_stock_threshold = st.number_input(
            "Low Stock Threshold",
            min_value=1,
            max_value=100,
            value=config.LOW_STOCK_THRESHOLD,
            help="Products with stock below this number are considered low stock"
        )
        
        expiry_warning_days = st.number_input(
            "Expiry Warning Days",
            min_value=1,
            max_value=365,
            value=config.EXPIRY_WARNING_DAYS,
            help="Show products expiring within this many days"
        )
        
        items_per_page = st.selectbox(
            "Items per Page",
            [25, 50, 100, 200],
            index=1,
            help="Number of items to display per page"
        )
    
    # Main content area
    if st.session_state.inventory_df.empty:
        st.info("👆 Please upload an inventory CSV file to get started.")
        
        # Show sample CSV format
        st.subheader("📋 Required CSV Format")
        sample_data = {
            'Product Name': ['Apple iPhone 13', 'Samsung Galaxy S21', 'Dell Laptop', 'Office Chair'],
            'Category': ['Electronics', 'Electronics', 'Electronics', 'Furniture'],
            'Stock': [25, 8, 15, 3],
            'Price': [999.99, 799.99, 1299.99, 299.99],
            'Expiry Date': ['2025-12-31', '2025-06-30', '2026-01-15', '2027-01-01']
        }
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df, use_container_width=True)
        
        return
    
    # Display inventory summary
    display_inventory_summary(st.session_state.inventory_df)
    
    st.divider()
    
    # Search and filter section
    st.subheader("🔍 Search & Filter")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_name = st.text_input(
            "Search by Product Name",
            placeholder="Enter product name...",
            key="search_name"
        )
    
    with col2:
        categories = data_loader.get_categories(st.session_state.inventory_df)
        search_category = st.selectbox(
            "Filter by Category",
            ["All Categories"] + categories,
            key="search_category"
        )
    
    with col3:
        sort_options = {
            "Product Name": "Product Name",
            "Category": "Category", 
            "Stock (Low to High)": "Stock",
            "Stock (High to Low)": "Stock_desc",
            "Price (Low to High)": "Price",
            "Price (High to Low)": "Price_desc",
            "Expiry Date": "Expiry Date"
        }
        sort_by = st.selectbox("Sort by", list(sort_options.keys()))
    
    # Filter buttons
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("🔍 Apply Search", type="primary"):
            df = st.session_state.inventory_df.copy()
            
            # Apply search filters
            category_filter = None if search_category == "All Categories" else search_category
            df = data_loader.search_inventory(df, search_name, category_filter)
            
            # Apply sorting
            sort_column = sort_options[sort_by]
            if sort_column.endswith("_desc"):
                sort_column = sort_column[:-5]
                df = data_loader.sort_dataframe(df, sort_column, ascending=False)
            else:
                df = data_loader.sort_dataframe(df, sort_column, ascending=True)
            
            st.session_state.filtered_df = df
            reset_page()
    
    with col2:
        if st.button("📉 Low Stock"):
            df = data_loader.filter_low_stock(st.session_state.inventory_df, low_stock_threshold)
            st.session_state.filtered_df = df
            reset_page()
    
    with col3:
        if st.button("⏰ Expiring Soon"):
            df = data_loader.filter_expiring(st.session_state.inventory_df, expiry_warning_days)
            st.session_state.filtered_df = df
            reset_page()
    
    with col4:
        if st.button("❌ Expired"):
            df = data_loader.get_expired_products(st.session_state.inventory_df)
            st.session_state.filtered_df = df
            reset_page()
    
    with col5:
        if st.button("🔄 Reset"):
            st.session_state.filtered_df = st.session_state.inventory_df.copy()
            reset_page()
    
    st.divider()
    
    # Display results
    result_count = len(st.session_state.filtered_df)
    st.subheader(f"📊 Results ({result_count} items)")
    
    if result_count > 0:
        # Export button
        col1, col2 = st.columns([1, 4])
        with col1:
            csv_data = data_loader.export_to_csv(st.session_state.filtered_df)
            st.download_button(
                label="📥 Export CSV",
                data=csv_data,
                file_name=f"inventory_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        # Display table
        display_data_table(st.session_state.filtered_df, st.session_state.current_page, items_per_page)
    
    st.divider()
    
    # AI Recommendations section
    st.subheader("🤖 AI-Powered Recommendations")
    
    if llm_recommender.is_available():
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎯 Generate Reorder Recommendations", type="primary"):
                with st.spinner("Analyzing inventory and generating recommendations..."):
                    recommendations = llm_recommender.generate_reorder_recommendations(
                        st.session_state.inventory_df, 
                        low_stock_threshold
                    )
                    st.session_state.recommendations = recommendations
        
        with col2:
            categories = data_loader.get_categories(st.session_state.inventory_df)
            if categories:
                selected_category = st.selectbox("Analyze Category", categories, key="category_analysis")
                if st.button("📈 Category Analysis"):
                    with st.spinner(f"Analyzing {selected_category} category..."):
                        analysis = llm_recommender.analyze_category_trends(
                            st.session_state.inventory_df, 
                            selected_category
                        )
                        st.session_state.category_analysis = analysis
        
        # Display recommendations
        if hasattr(st.session_state, 'recommendations'):
            st.subheader("💡 Reorder Recommendations")
            st.markdown(st.session_state.recommendations)
        
        if hasattr(st.session_state, 'category_analysis'):
            st.subheader("📊 Category Analysis")
            st.markdown(st.session_state.category_analysis)
    
    else:
        st.info("🔧 AI recommendations are not available. Please set OPENAI_API_KEY or GEMINI_API_KEY in your .env file to enable AI-powered features.")
        st.markdown("""
        **To enable AI recommendations:**
        1. Get an API key from [OpenAI](https://platform.openai.com/api-keys) or [Google AI Studio](https://makersuite.google.com/app/apikey)
        2. Create a `.env` file in the project root
        3. Add: `OPENAI_API_KEY=your_key_here` or `GEMINI_API_KEY=your_key_here`
        4. Restart the application
        """)

if __name__ == "__main__":
    main()