import pandas as pd
import numpy as np
import string
import random
import plotly.express as px
import streamlit as st
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

st.set_page_config(page_title="Customer Eligibility Dashboard", layout="wide")

st.title("Customer Eligibility Dashboard")
st.caption("XGBoost + constraint-based grouping for term deposit targeting")

DATA_PATH = "data/Bank-term-deposit.csv"

@st.cache_data
def generate_customer_ids(n: int):
    ids = []
    letters = string.ascii_uppercase
    for i in range(n):
        num = i % 1000
        a = letters[(i // 1000) % 26]
        b = letters[(i // (1000 * 26)) % 26]
        ids.append(f"{num:03d}{a}{b}")
    return ids

@st.cache_data
def generate_unique_contacts(n: int, seed: int = 42):
    rng = random.Random(seed)
    contacts = set()
    while len(contacts) < n:
        contacts.add(str(rng.randint(1000000000, 9999999999)))
    return list(contacts)

@st.cache_data
def build_dataset(path: str):
    df = pd.read_csv(path)
    df = df.drop(columns=["duration"], errors="ignore")
    df = df.rename(columns={"nremployed": "salary"})
    df = df.replace("unknown", "missing")

    df["customer_id"] = generate_customer_ids(len(df))
    df["contact_number"] = generate_unique_contacts(len(df))

    y = df["y"].map({"no": 0, "yes": 1})
    X_raw = df.drop(columns=["y", "customer_id", "contact_number"], errors="ignore")
    X = pd.get_dummies(X_raw, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        scale_pos_weight=(len(y_train[y_train == 0]) / len(y_train[y_train == 1])),
        random_state=42,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)

    df["prediction"] = model.predict(X)
    df["probability"] = model.predict_proba(X)[:, 1]

    df["passed_conditions"] = (
        df["Age"].between(21, 45).astype(int)
        + (df["Loan"].str.lower() == "no").astype(int)
        + (df["housing"].str.lower() == "no").astype(int)
        + df["salary"].between(5000, 5050).astype(int)
        + (df["Marital"].str.lower() == "married").astype(int)
    )

    df["group"] = np.select(
        [
            (df["prediction"] == 1) & (df["passed_conditions"] == 5),
            (df["prediction"] == 1) & (df["passed_conditions"] >= 3),
        ],
        [
            "Group 1 - Eligible",
            "Group 2 - Waiting List",
        ],
        default="Group 3 - Not Eligible",
    )

    return df

try:
    df = build_dataset(DATA_PATH)
except Exception as e:
    st.error(f"Failed to load/build dashboard data: {e}")
    st.stop()

st.sidebar.header("Filters")
group_filter = st.sidebar.multiselect(
    "Select group(s)",
    options=sorted(df["group"].unique().tolist()),
    default=sorted(df["group"].unique().tolist()),
)

job_options = sorted(df["Job"].dropna().unique().tolist())
job_filter = st.sidebar.multiselect(
    "Select job(s)",
    options=job_options,
    default=job_options
)

age_range = st.sidebar.slider(
    "Age range",
    min_value=int(df["Age"].min()),
    max_value=int(df["Age"].max()),
    value=(int(df["Age"].min()), int(df["Age"].max())),
)

prob_range = st.sidebar.slider(
    "Probability range",
    min_value=0.0,
    max_value=1.0,
    value=(0.0, 1.0),
    step=0.01,
)

filtered = df[
    (df["group"].isin(group_filter))
    & (df["Job"].isin(job_filter))
    & (df["Age"].between(age_range[0], age_range[1]))
    & (df["probability"].between(prob_range[0], prob_range[1]))
].copy()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Customers", f"{len(filtered):,}")
c2.metric("Eligible", f"{(filtered['group'] == 'Group 1 - Eligible').sum():,}")
c3.metric("Waiting List", f"{(filtered['group'] == 'Group 2 - Waiting List').sum():,}")
c4.metric("Not Eligible", f"{(filtered['group'] == 'Group 3 - Not Eligible').sum():,}")

c5, c6, c7 = st.columns(3)
c5.metric("Avg Probability", f"{filtered['probability'].mean():.3f}")
c6.metric("Avg Passed Conditions", f"{filtered['passed_conditions'].mean():.2f}")
c7.metric("Actual Yes Rate", f"{(filtered['y'] == 'yes').mean():.2%}")

left, right = st.columns(2)

with left:
    group_counts = filtered["group"].value_counts().reset_index()
    group_counts.columns = ["group", "count"]
    fig_group = px.bar(group_counts, x="group", y="count", title="Customer Count by Group")
    st.plotly_chart(fig_group, use_container_width=True)

with right:
    job_group = filtered.groupby(["Job", "group"]).size().reset_index(name="count")
    fig_job = px.bar(job_group, x="Job", y="count", color="group", title="Job Distribution by Group")
    st.plotly_chart(fig_job, use_container_width=True)

left2, right2 = st.columns(2)

with left2:
    fig_age = px.histogram(filtered, x="Age", color="group", nbins=25, barmode="overlay", title="Age Distribution by Group")
    st.plotly_chart(fig_age, use_container_width=True)

with right2:
    fig_prob = px.box(filtered, x="group", y="probability", color="group", title="Prediction Probability by Group")
    st.plotly_chart(fig_prob, use_container_width=True)

st.subheader("Customer Tables")
selected_group = st.selectbox(
    "View customer list for group",
    options=["Group 1 - Eligible", "Group 2 - Waiting List", "Group 3 - Not Eligible"],
)

show_cols = [
    "customer_id", "contact_number", "group", "probability", "passed_conditions",
    "Age", "Job", "Marital", "housing", "Loan", "salary", "y"
]

group_df = filtered[filtered["group"] == selected_group][show_cols].sort_values(
    by=["probability", "passed_conditions"], ascending=[False, False]
)

st.dataframe(group_df, use_container_width=True, hide_index=True)

csv = group_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label=f"Download {selected_group} CSV",
    data=csv,
    file_name=selected_group.lower().replace(" ", "_").replace("-", "") + ".csv",
    mime="text/csv",
)

st.subheader("Top Candidates")
top_n = st.slider("Number of top customers to show", 5, 50, 10)
top_candidates = filtered.sort_values(by="probability", ascending=False)[show_cols].head(top_n)
st.dataframe(top_candidates, use_container_width=True, hide_index=True)