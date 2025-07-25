import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from typing import Optional, List, Tuple
import io

class InventoryDataLoader:
    def __init__(self):
        self.required_columns = ['Product Name', 'Category', 'Stock', 'Price', 'Expiry Date']
    
    def load_csv(self, file) -> pd.DataFrame:
        """Load CSV file and return pandas DataFrame"""
        try:
            # Read CSV file
            if hasattr(file, 'read'):
                # Streamlit uploaded file
                df = pd.read_csv(file)
            else:
                # File path
                df = pd.read_csv(file)
            
            # Validate required columns
            missing_columns = [col for col in self.required_columns if col not in df.columns]
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
                st.info(f"Required columns: {', '.join(self.required_columns)}")
                return pd.DataFrame()
            
            # Clean and process the data
            df = self._clean_data(df)
            
            return df
            
        except Exception as e:
            st.error(f"Error loading CSV: {str(e)}")
            return pd.DataFrame()
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and process the inventory data"""
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Clean Product Name and Category
        df['Product Name'] = df['Product Name'].astype(str).str.strip()
        df['Category'] = df['Category'].astype(str).str.strip()
        
        # Convert Stock to numeric
        df['Stock'] = pd.to_numeric(df['Stock'], errors='coerce').fillna(0).astype(int)
        
        # Convert Price to numeric
        df['Price'] = pd.to_numeric(df['Price'].astype(str).str.replace('$', '').str.replace(',', ''), 
                                   errors='coerce').fillna(0.0)
        
        # Convert Expiry Date to datetime
        df['Expiry Date'] = pd.to_datetime(df['Expiry Date'], errors='coerce')
        
        # Remove rows with invalid data
        df = df.dropna(subset=['Product Name', 'Category'])
        df = df[df['Product Name'] != 'nan']
        df = df[df['Category'] != 'nan']
        
        # Sort by Product Name
        df = df.sort_values('Product Name').reset_index(drop=True)
        
        return df
    
    def search_inventory(self, df: pd.DataFrame, name: Optional[str] = None, 
                        category: Optional[str] = None) -> pd.DataFrame:
        """Search inventory by product name or category"""
        if df.empty:
            return df
        
        filtered_df = df.copy()
        
        if name and name.strip():
            # Case-insensitive substring search in product name
            filtered_df = filtered_df[
                filtered_df['Product Name'].str.contains(name.strip(), case=False, na=False)
            ]
        
        if category and category.strip():
            # Case-insensitive substring search in category
            filtered_df = filtered_df[
                filtered_df['Category'].str.contains(category.strip(), case=False, na=False)
            ]
        
        return filtered_df
    
    def filter_low_stock(self, df: pd.DataFrame, threshold: int = 10) -> pd.DataFrame:
        """Filter products with low stock"""
        if df.empty:
            return df
        
        return df[df['Stock'] <= threshold].copy()
    
    def filter_expiring(self, df: pd.DataFrame, days: int = 30) -> pd.DataFrame:
        """Filter products expiring within specified days"""
        if df.empty:
            return df
        
        # Calculate cutoff date
        cutoff_date = datetime.now() + timedelta(days=days)
        
        # Filter products with valid expiry dates that are within the cutoff
        filtered_df = df[
            (df['Expiry Date'].notna()) & 
            (df['Expiry Date'] <= cutoff_date) &
            (df['Expiry Date'] >= datetime.now())  # Not already expired
        ].copy()
        
        return filtered_df
    
    def get_expired_products(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get products that have already expired"""
        if df.empty:
            return df
        
        today = datetime.now()
        
        return df[
            (df['Expiry Date'].notna()) & 
            (df['Expiry Date'] < today)
        ].copy()
    
    def get_categories(self, df: pd.DataFrame) -> List[str]:
        """Get unique categories from the dataframe"""
        if df.empty:
            return []
        
        return sorted(df['Category'].unique().tolist())
    
    def get_inventory_summary(self, df: pd.DataFrame) -> dict:
        """Get summary statistics of the inventory"""
        if df.empty:
            return {}
        
        total_products = len(df)
        total_stock = df['Stock'].sum()
        total_value = (df['Stock'] * df['Price']).sum()
        low_stock_count = len(self.filter_low_stock(df))
        expiring_count = len(self.filter_expiring(df))
        expired_count = len(self.get_expired_products(df))
        categories_count = len(self.get_categories(df))
        
        return {
            'total_products': total_products,
            'total_stock': total_stock,
            'total_value': total_value,
            'low_stock_count': low_stock_count,
            'expiring_count': expiring_count,
            'expired_count': expired_count,
            'categories_count': categories_count,
            'avg_price': df['Price'].mean() if total_products > 0 else 0
        }
    
    def sort_dataframe(self, df: pd.DataFrame, sort_by: str, ascending: bool = True) -> pd.DataFrame:
        """Sort dataframe by specified column"""
        if df.empty or sort_by not in df.columns:
            return df
        
        return df.sort_values(sort_by, ascending=ascending).reset_index(drop=True)
    
    def paginate_dataframe(self, df: pd.DataFrame, page: int = 1, items_per_page: int = 50) -> Tuple[pd.DataFrame, dict]:
        """Paginate dataframe and return page info"""
        if df.empty:
            return df, {'total_pages': 0, 'current_page': 1, 'total_items': 0}
        
        total_items = len(df)
        total_pages = (total_items - 1) // items_per_page + 1
        
        # Ensure page is within bounds
        page = max(1, min(page, total_pages))
        
        start_idx = (page - 1) * items_per_page
        end_idx = start_idx + items_per_page
        
        paginated_df = df.iloc[start_idx:end_idx].copy()
        
        page_info = {
            'total_pages': total_pages,
            'current_page': page,
            'total_items': total_items,
            'start_item': start_idx + 1,
            'end_item': min(end_idx, total_items)
        }
        
        return paginated_df, page_info
    
    def export_to_csv(self, df: pd.DataFrame) -> str:
        """Export dataframe to CSV string"""
        if df.empty:
            return ""
        
        # Create a copy and format for export
        export_df = df.copy()
        
        # Format price column
        export_df['Price'] = export_df['Price'].apply(lambda x: f"${x:.2f}")
        
        # Format expiry date
        export_df['Expiry Date'] = export_df['Expiry Date'].dt.strftime('%Y-%m-%d')
        
        return export_df.to_csv(index=False)

# Create global instance
data_loader = InventoryDataLoader()