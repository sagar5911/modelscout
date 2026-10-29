"""
make_sample.py
Generate a synthetic customer-churn dataset (~1500 rows) and save it to
data/sample_churn.csv.  Run from the project root:
    python scripts/make_sample.py
"""

import pathlib
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 1_500

# ------------------------------------------------------------------
# Feature generation
# ------------------------------------------------------------------

tenure_months = RNG.integers(1, 73, size=N).astype(float)
# Inject ~3 % missing
tenure_months[RNG.choice(N, size=int(N * 0.03), replace=False)] = np.nan

monthly_charges = np.round(
    20 + 70 * RNG.beta(2, 5, size=N) + RNG.normal(0, 3, size=N), 2
)
monthly_charges = np.clip(monthly_charges, 10, 110)

total_charges = np.where(
    tenure_months != np.nan,
    monthly_charges * tenure_months + RNG.normal(0, 20, size=N),
    np.nan,
)
total_charges = np.round(total_charges, 2)
# ~5 % missing
total_charges[RNG.choice(N, size=int(N * 0.05), replace=False)] = np.nan

num_support_calls = RNG.integers(0, 10, size=N).astype(float)
num_support_calls[RNG.choice(N, size=int(N * 0.02), replace=False)] = np.nan

contract_type = RNG.choice(
    ["Month-to-month", "One year", "Two year"],
    size=N,
    p=[0.55, 0.25, 0.20],
)
payment_method = RNG.choice(
    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
    size=N,
    p=[0.34, 0.22, 0.22, 0.22],
)
internet_service = RNG.choice(
    ["Fiber optic", "DSL", "No"],
    size=N,
    p=[0.44, 0.34, 0.22],
)
gender = RNG.choice(["Male", "Female"], size=N)
senior_citizen = RNG.choice([0, 1], size=N, p=[0.84, 0.16])

# Derived binary features
online_security = RNG.choice(["Yes", "No", "No internet service"], size=N,
                              p=[0.29, 0.49, 0.22])
tech_support = RNG.choice(["Yes", "No", "No internet service"], size=N,
                           p=[0.29, 0.49, 0.22])
paperless_billing = RNG.choice(["Yes", "No"], size=N, p=[0.59, 0.41])

# Synthetic churn probability (not purely random — uses real-ish relationships)
churn_logit = (
    -2.0
    + 0.04 * (monthly_charges - 50)
    - 0.03 * np.where(np.isnan(tenure_months), 12, tenure_months)
    + 0.5 * (contract_type == "Month-to-month").astype(float)
    - 0.4 * (contract_type == "Two year").astype(float)
    + 0.3 * (internet_service == "Fiber optic").astype(float)
    + 0.2 * (online_security == "No").astype(float)
    + 0.15 * np.where(np.isnan(num_support_calls), 3, num_support_calls)
    + 0.3 * senior_citizen
    + RNG.normal(0, 0.5, size=N)
)
churn_prob = 1 / (1 + np.exp(-churn_logit))
churn = (RNG.random(size=N) < churn_prob).astype(int)
# Mild imbalance: ~27 % churn
churn_label = np.where(churn == 1, "Yes", "No")

customer_id = [f"CUST-{i:05d}" for i in range(1, N + 1)]

df = pd.DataFrame(
    {
        "customer_id": customer_id,
        "gender": gender,
        "senior_citizen": senior_citizen,
        "tenure_months": tenure_months,
        "contract_type": contract_type,
        "payment_method": payment_method,
        "internet_service": internet_service,
        "online_security": online_security,
        "tech_support": tech_support,
        "paperless_billing": paperless_billing,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "num_support_calls": num_support_calls,
        "churn": churn_label,
    }
)

out_path = pathlib.Path(__file__).parent.parent / "data" / "sample_churn.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)
print(f"Saved {len(df):,} rows to {out_path}")
print(f"Churn rate: {df['churn'].value_counts(normalize=True)['Yes']:.1%}")
