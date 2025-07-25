import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Any
import io

class CSVDataLoader:
    def __init__(self):
        # No required columns - accept any CSV structure
        pass
    
    def load_csv(self, file) -> pd.DataFrame:
        """Load any CSV file and return pandas DataFrame"""
        try:
            # Read CSV file
            if hasattr(file, 'read'):
                df = pd.read_csv(file)
            else:
                df = pd.read_csv(file)
            
            if df.empty:
                st.error("The uploaded CSV file is empty.")
                return pd.DataFrame()
            
            # Basic data cleaning
            df = self._clean_data(df)
            
            return df
            
        except Exception as e:
            st.error(f"Error loading CSV: {str(e)}")
            return pd.DataFrame()
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic data cleaning for any CSV"""
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Remove completely empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Clean column names (remove leading/trailing spaces)
        df.columns = df.columns.str.strip()
        
        # Try to infer better data types
        for col in df.columns:
            # Try to convert to numeric if possible
            if df[col].dtype == 'object':
                # Remove common currency symbols and separators for numeric detection
                test_series = df[col].astype(str).str.replace(r'[$,€£¥%]', '', regex=True)
                numeric_series = pd.to_numeric(test_series, errors='coerce')
                
                # If more than 70% of values can be converted to numeric, use numeric
                if numeric_series.notna().sum() / len(df) > 0.7:
                    df[col] = numeric_series
                else:
                    # Try to convert to datetime
                    try:
                        datetime_series = pd.to_datetime(df[col], errors='coerce')
                        if datetime_series.notna().sum() / len(df) > 0.7:
                            df[col] = datetime_series
                    except:
                        pass
        
        return df
    
    def search_data(self, df: pd.DataFrame, search_term: str, columns: List[str] = None) -> pd.DataFrame:
        """Search data across specified columns or all text columns"""
        if df.empty or not search_term.strip():
            return df
        
        filtered_df = df.copy()
        search_term = search_term.strip().lower()
        
        # If no columns specified, search all text/object columns
        if columns is None:
            columns = [col for col in df.columns if df[col].dtype == 'object']
        
        # Create a mask for rows that contain the search term in any specified column
        mask = pd.Series(False, index=df.index)
        
        for col in columns:
            if col in df.columns:
                # Convert to string and search (case-insensitive)
                col_mask = df[col].astype(str).str.lower().str.contains(search_term, na=False, regex=False)
                mask = mask | col_mask
        
        return filtered_df[mask]
    
    def filter_by_column_value(self, df: pd.DataFrame, column: str, value: Any, 
                              operation: str = 'equals') -> pd.DataFrame:
        """Filter data by column value with different operations"""
        if df.empty or column not in df.columns:
            return df
        
        filtered_df = df.copy()
        
        try:
            if operation == 'equals':
                mask = filtered_df[column] == value
            elif operation == 'not_equals':
                mask = filtered_df[column] != value
            elif operation == 'greater_than':
                mask = filtered_df[column] > value
            elif operation == 'less_than':
                mask = filtered_df[column] < value
            elif operation == 'greater_equal':
                mask = filtered_df[column] >= value
            elif operation == 'less_equal':
                mask = filtered_df[column] <= value
            elif operation == 'contains':
                mask = filtered_df[column].astype(str).str.contains(str(value), case=False, na=False)
            else:
                mask = filtered_df[column] == value
            
            return filtered_df[mask]
            
        except Exception as e:
            st.error(f"Error filtering by {column}: {e}")
            return df
    
    def get_column_info(self, df: pd.DataFrame) -> dict:
        """Get information about columns in the dataframe"""
        if df.empty:
            return {}
        
        info = {}
        for col in df.columns:
            col_info = {
                'dtype': str(df[col].dtype),
                'non_null_count': df[col].count(),
                'null_count': df[col].isnull().sum(),
                'unique_count': df[col].nunique()
            }
            
            # Add type-specific info
            if pd.api.types.is_numeric_dtype(df[col]):
                col_info.update({
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'mean': df[col].mean()
                })
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                col_info.update({
                    'min_date': df[col].min(),
                    'max_date': df[col].max()
                })
            else:
                # Text columns
                if col_info['unique_count'] <= 20:  # Show unique values for categorical data
                    col_info['unique_values'] = df[col].value_counts().head(10).to_dict()
            
            info[col] = col_info
        
        return info
    
    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """Get summary statistics of the data"""
        if df.empty:
            return {}
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        text_cols = df.select_dtypes(include=['object']).columns.tolist()
        date_cols = df.select_dtypes(include=['datetime']).columns.tolist()
        
        return {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'numeric_columns': len(numeric_cols),
            'text_columns': len(text_cols),
            'date_columns': len(date_cols),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'column_names': df.columns.tolist(),
            'numeric_column_names': numeric_cols,
            'text_column_names': text_cols,
            'date_column_names': date_cols
        }
    
    def sort_dataframe(self, df: pd.DataFrame, sort_by: str, ascending: bool = True) -> pd.DataFrame:
        """Sort dataframe by specified column"""
        if df.empty or sort_by not in df.columns:
            return df
        
        try:
            return df.sort_values(sort_by, ascending=ascending).reset_index(drop=True)
        except Exception as e:
            st.error(f"Error sorting by {sort_by}: {e}")
            return df
    
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
        
        return df.to_csv(index=False)
    
    def get_unique_values(self, df: pd.DataFrame, column: str, limit: int = 100) -> List[str]:
        """Get unique values from a column (for dropdowns, etc.)"""
        if df.empty or column not in df.columns:
            return []
        
        unique_vals = df[column].dropna().unique()
        
        # Convert to string and limit
        unique_strings = [str(val) for val in unique_vals][:limit]
        
        return sorted(unique_strings)

# Create global instance
data_loader = CSVDataLoader()