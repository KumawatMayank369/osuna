from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI(title="Loan Approval Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    raise RuntimeError("model.pkl not found. Run train_model.py first.")


class LoanInput(BaseModel):
    age: float
    income: float
    credit_score: float
    loan_amount: float
    employment_years: float
    num_accounts: float
    num_loans: float
    monthly_expenses: float
    savings: float
    debt_ratio: float


@app.get("/")
def root():
    return {"status": "Loan Approval API is running"}


@app.post("/predict")
def predict(data: LoanInput):
    try:
        features = np.array([[
            data.age, data.income, data.credit_score, data.loan_amount,
            data.employment_years, data.num_accounts, data.num_loans,
            data.monthly_expenses, data.savings, data.debt_ratio
        ]])
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        return {
            "approved": int(prediction),
            "result": "Approved ✅" if prediction == 1 else "Rejected ❌",
            "confidence": round(float(max(probability)) * 100, 1)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Serve static HTML — must be LAST
app.mount("/app", StaticFiles(directory="static", html=True), name="static")
