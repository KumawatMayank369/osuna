# LoanIQ — ML Loan Approval Predictor

A complete ML web app: Random Forest model served via FastAPI with a clean HTML frontend.

## Project Structure
```
ml_app/
├── main.py             # FastAPI backend
├── train_model.py      # Train model & save pickle
├── model.pkl           # Trained model (auto-generated)
├── loan_dataset.csv    # Dataset (auto-generated)
├── requirements.txt    # Python dependencies
├── static/
│   └── index.html      # Web frontend
└── README.md
```

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model (creates model.pkl)
```bash
python train_model.py
```

### 3. Start the FastAPI server
```bash
uvicorn main:app --reload
```

### 4. Open the web app
Open your browser and go to:
```
http://localhost:8000/app
```

## API Endpoints
| Method | URL        | Description          |
|--------|------------|----------------------|
| GET    | /          | Health check         |
| POST   | /predict   | Get loan prediction  |
| GET    | /app       | Web interface        |

## Sample API Call
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 30,
    "income": 60000,
    "credit_score": 720,
    "loan_amount": 15000,
    "employment_years": 5,
    "num_accounts": 3,
    "num_loans": 1,
    "monthly_expenses": 1500,
    "savings": 20000,
    "debt_ratio": 0.3
  }'
```

## Model Info
- **Algorithm:** Random Forest Classifier (100 trees)
- **Dataset:** 1000 synthetic loan applications, 10 features
- **Accuracy:** ~96.5%
- **Features:** age, income, credit score, loan amount, employment years, accounts, loans, expenses, savings, debt ratio
