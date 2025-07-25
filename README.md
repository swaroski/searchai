# 📊 CSV Data Explorer

A powerful and intuitive data analysis dashboard built with Streamlit. Upload any CSV file, explore your data with smart search and filtering, and get AI-powered insights.

## ✨ Features

- **📁 Universal CSV Upload**: Upload any CSV file - no format restrictions
- **🔍 Smart Search**: Search across all text columns with real-time filtering  
- **📊 Advanced Filtering**: Filter by column values with multiple operations (equals, greater than, contains, etc.)
- **📈 Flexible Sorting**: Sort by any column (numeric, text, or date)
- **📋 Pagination**: Handle large datasets with configurable page sizes
- **💾 Export**: Download filtered results as CSV
- **🤖 AI-Powered Analysis**: Get intelligent insights about your data using OpenAI or Gemini
- **📊 Data Analytics**: View data summaries, column statistics, and type detection
- **⚙️ Interactive Tools**: Column info, quick stats, and data quality insights

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas
- **AI Integration**: OpenAI GPT / Google Gemini (optional)
- **Python**: 3.10+ (tested with 3.13.2)

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd searchai
pip install -r requirements.txt
```

### 2. Configure Environment (Optional)

For AI-powered recommendations, you can configure API keys in two ways:

**Option A: Local Development (.env file)**
```bash
cp .env.example .env
# Edit .env and add your API keys
```

**Option B: Streamlit Cloud (secrets.toml)**
```bash
# Configure .streamlit/secrets.toml for cloud deployment
# See deployment section below for details
```

Get API keys from:
- [OpenAI Platform](https://platform.openai.com/api-keys)
- [Google AI Studio](https://makersuite.google.com/app/apikey)

### 3. Run the Application

```bash
streamlit run streamlit_app.py
```

### 4. Access the Dashboard

Open your browser to: http://localhost:8501

## 📋 Flexible CSV Support

**🎉 Upload ANY CSV file!** The app automatically detects and maps your columns to inventory fields.

### Supported Column Types:

| Field Type | Recognized Column Names |
|------------|--------------------------|
| **Product/Item** | product_name, product, name, item, item_name, title |
| **Category** | category, type, group, class, department |
| **Stock/Quantity** | stock, quantity, qty, amount, inventory, units |
| **Price/Cost** | price, cost, value, unit_price, rate |
| **Expiry Date** | expiry_date, expiration, expire, best_before, exp_date |

### Sample CSV Formats:
**Option 1: Standard Format**
```csv
Product Name,Category,Stock,Price,Expiry Date
Apple iPhone 14,Electronics,25,999.99,2025-12-31
Samsung Galaxy S22,Electronics,8,849.99,2025-06-30
```

**Option 2: Alternative Format** (automatically detected!)
```csv
item_name,type,qty,cost,expire
iPhone 14,Electronics,25,999.99,2025-12-31
Galaxy S22,Electronics,8,849.99,2025-06-30
```

## 🎯 How to Use

### 1. Upload Inventory
- Click "Browse files" or drag-and-drop ANY CSV file
- The app will automatically detect your column structure
- If detection fails, use the interactive column mapping tool
- View the summary statistics dashboard

### 2. Search & Filter
- **Search by Name**: Type product name for instant filtering
- **Filter by Category**: Select from available categories
- **Quick Filters**: Use buttons for low stock, expiring, or expired items
- **Sort Options**: Order by price, stock, expiry date, or alphabetically

### 3. View Results
- Browse paginated results with configurable page sizes
- Export filtered data as CSV
- View detailed product information

### 4. AI Recommendations (Optional)
- Click "Generate Reorder Recommendations" for intelligent suggestions
- Get category-specific analysis and trends
- Receive prioritized reorder lists with quantities

## ⚙️ Configuration

### Environment Variables (.env file)

```bash
# API Keys (optional)
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here

# Thresholds
LOW_STOCK_THRESHOLD=10        # Products below this are "low stock"
EXPIRY_WARNING_DAYS=30        # Show items expiring within X days
ITEMS_PER_PAGE=50            # Default pagination size
```

### Streamlit Secrets (.streamlit/secrets.toml)

```toml
[general]
# API Keys (optional)
OPENAI_API_KEY = "your_openai_api_key_here"
GEMINI_API_KEY = "your_gemini_api_key_here"

# Thresholds  
LOW_STOCK_THRESHOLD = 10
EXPIRY_WARNING_DAYS = 30
ITEMS_PER_PAGE = 50
```

### Sidebar Settings
- **Low Stock Threshold**: Adjust what counts as "low stock"
- **Expiry Warning Days**: Set expiration warning period
- **Items per Page**: Choose table pagination size

## 📁 Project Structure

```
searchai/
├── streamlit_app.py          # Main Streamlit application
├── app/
│   ├── config.py             # Configuration management
│   ├── data_loader.py        # CSV processing and filtering
│   └── llm_recommender.py    # AI recommendation engine
├── .streamlit/
│   └── secrets.toml          # Streamlit secrets (for cloud deployment)
├── requirements.txt          # Python dependencies
├── sample_inventory.csv      # Example data file
├── .env.example             # Environment template
├── .gitignore               # Git ignore rules
└── README.md               # This file
```

## 🤖 AI Features

### Reorder Recommendations
The AI analyzes your inventory and provides:
- **Priority Items**: Most urgent restocking needs
- **Suggested Quantities**: Based on stock levels and trends
- **Risk Assessment**: Impact of stockouts
- **Category Insights**: Patterns across product categories

### Category Analysis
Get detailed insights about specific categories:
- Stock level trends
- Price analysis
- Performance metrics
- Optimization suggestions

## 🚀 Deployment Options

### Streamlit Cloud
1. Fork this repository
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy `streamlit_app.py`
4. Configure secrets in your Streamlit Cloud dashboard:
   ```toml
   [general]
   OPENAI_API_KEY = "your_openai_api_key_here"
   GEMINI_API_KEY = "your_gemini_api_key_here"
   LOW_STOCK_THRESHOLD = 10
   EXPIRY_WARNING_DAYS = 30
   ITEMS_PER_PAGE = 50
   ```

### Local Development
```bash
streamlit run streamlit_app.py
```

### Docker
```dockerfile
FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 📊 Sample Data

The included `sample_inventory.csv` contains 40 sample products across categories:
- Electronics (phones, laptops, accessories)
- Furniture (chairs, desks, lighting)
- Appliances (kitchen, cleaning)
- Health & Fitness (supplements, equipment)
- Home Decor

Use this to test all features before uploading your own data.

## 🔧 Troubleshooting

### Common Issues

1. **CSV Upload Fails**
   - Ensure all required columns are present
   - Check for special characters in data
   - Verify date format (YYYY-MM-DD recommended)

2. **AI Recommendations Not Working**
   - Verify API keys in `.env` file or Streamlit secrets
   - Check API key permissions and quotas
   - Ensure internet connection for API calls

3. **Performance Issues**
   - Reduce items per page for large datasets
   - Consider filtering data before operations
   - Check available system memory

### Data Format Tips
- Use consistent date formats (YYYY-MM-DD)
- Avoid special characters in product names
- Ensure numeric fields contain valid numbers
- Use UTF-8 encoding for international characters

## 🛡️ Privacy & Security

- All data processing happens locally
- CSV files are stored in memory only
- API keys are used securely for AI features
- No data is transmitted except to chosen AI services
- Streamlit secrets are properly git-ignored

## 📈 Future Enhancements

Potential additions for future versions:
- SQLite database integration
- Multi-file inventory management
- Advanced analytics and reporting
- Barcode scanning integration
- Automated reorder workflows
- Email/SMS alerts for low stock

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Data processing powered by [Pandas](https://pandas.pydata.org/)
- AI features using [OpenAI](https://openai.com/) and [Google Gemini](https://deepmind.google/technologies/gemini/)

---

**Made with ❤️ for efficient inventory management**