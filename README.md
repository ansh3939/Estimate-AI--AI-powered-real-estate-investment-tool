# EstiMate AI

EstiMate AI is a professional property analytics platform with ML-powered predictions, portfolio tracking, investment analysis, and EMI calculations for real estate in India.

## Features
- **Property Price Prediction:** Predict property prices using advanced machine learning models (XGBoost, Random Forest, Decision Tree).
- **Portfolio Tracker:** Track and analyze your real estate investments.
- **Investment Analyzer:** Evaluate potential returns and investment metrics.
- **EMI Calculator:** Calculate home loan EMIs with flexible options.
- **Appreciation Analyzer:** Analyze property appreciation trends (if enabled).

## Getting Started

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation & Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/ansh3939/Estimate-AI.git
   cd Estimate-AI
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the web application:**
   ```bash
   streamlit run main.py
   ```
   The app will be available at [http://localhost:8501](http://localhost:8501) by default.

### Notes
- Make sure you have an internet connection for the first run (to download dependencies).
- If you want to use a different port, add `--server.port <port>` to the run command.
- If you encounter any issues, ensure all dependencies are installed and you are using a compatible Python version.

## Project Structure
- `main.py` — Main Streamlit app
- `database.py` — Database management
- `fast_ml_model.py` — ML models for price prediction
- `emi_calculator.py` — EMI calculation logic
- `investment_analyzer.py` — Investment analysis
- `portfolio_analyzer.py` — Portfolio tracking
- `appreciation_analyzer.py` — Appreciation analysis (if present)
- `complete_property_dataset.csv` — Sample property data

## License
This project is open source and available under the MIT License.

---

For questions or contributions, open an issue or pull request on [GitHub](https://github.com/ansh3939/Estimate-AI). 