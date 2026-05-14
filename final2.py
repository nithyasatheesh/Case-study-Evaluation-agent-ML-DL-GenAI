import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
from docx import Document
from PyPDF2 import PdfReader
import zipfile
import io
import json
import re
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# -------------------------
# INIT
# -------------------------
load_dotenv()
client = OpenAI()

st.set_page_config(page_title="AI Evaluator", layout="wide")
st.title("📊 AI Case Study Evaluator (Final Version )")

# -------------------------
# FILE UPLOADS
# -------------------------
problem_file = st.file_uploader("Upload Problem", type=["docx", "pdf"])
rubric_file = st.file_uploader("Upload Rubric", type=["xlsx"])
zip_file = st.file_uploader("Upload Submissions (ZIP)", type=["zip"])
dataset_file = st.file_uploader("Upload Dataset (Optional)", type=["csv"])

# -------------------------
# HELPERS
# -------------------------
def read_docx(file):
    return "\n".join([p.text for p in Document(file).paragraphs])

def read_pdf(file):
    reader = PdfReader(file)
    return "\n".join([p.extract_text() or "" for p in reader.pages])

def read_rubric(file):
    return pd.read_excel(file)

def rubric_to_text(df):
    return "\n".join(
        f"{row['Criterion']} (Max {row['Max Score']}): {row['Description']}"
        for _, row in df.iterrows()
    )

def read_submission(name, content):
    text = content.decode("utf-8", errors="ignore")

    if name.endswith(".html"):
        soup = BeautifulSoup(text, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text()

    keywords = [
        "model", "fit", "predict", "accuracy", "loss",
        "r2", "r2_score", "precision", "recall",
        "ann", "neural"
    ]

    important = [
        line for line in text.split("\n")
        if any(k in line.lower() for k in keywords)
    ]

    return "\n".join(important[:300]) + "\n" + text[:4000]

def extract_zip(zip_file):
    files = []
    with zipfile.ZipFile(io.BytesIO(zip_file.read()), 'r') as z:
        for name in z.namelist():
            if name.endswith((".txt", ".html")):
                files.append((name, z.read(name)))
    return files

def safe_parse(raw):
    try:
        return json.loads(raw)
    except:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            return json.loads(match.group())
    return {"scores": {}, "strengths": "", "improvements": ""}

# -------------------------
# DATASET SUMMARY
# -------------------------
def get_dataset_summary(file):
    try:
        df = pd.read_csv(file)
        return f"""
Rows: {df.shape[0]}
Columns: {list(df.columns)}
Sample:
{df.head(3).to_string(index=False)}
"""
    except:
        return ""

# -------------------------
# METRIC EXTRACTION
# -------------------------
def extract_metrics(text):
    metrics = {}

    acc = re.findall(r'accuracy\s*[:=]\s*([0-9\.]+)', text, re.IGNORECASE)
    if acc:
        metrics["accuracy"] = acc[0]

    r2 = re.findall(r'r2[_ ]?score\s*[:=]\s*([0-9\.]+)', text, re.IGNORECASE)
    if r2:
        metrics["r2_score"] = r2[0]

    return metrics

# -------------------------
# OPENAI CALL
# -------------------------
def evaluate(prompt):
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a strict evaluator. Use full scoring range and ensure score matches feedback."
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return res.choices[0].message.content

# -------------------------
# MAIN
# -------------------------
if st.button("🚀 Evaluate Submissions"):

    if not (problem_file and rubric_file and zip_file):
        st.error("Upload Problem, Rubric, and Submissions")
        st.stop()

    problem = read_docx(problem_file) if problem_file.name.endswith(".docx") else read_pdf(problem_file)
    rubric_df = read_rubric(rubric_file)
    rubric_text = rubric_to_text(rubric_df)
    criteria_list = rubric_df["Criterion"].tolist()

    dataset_summary = get_dataset_summary(dataset_file) if dataset_file else ""

    files = extract_zip(zip_file)
    st.write(f"📂 Submissions: {len(files)}")

    def process_file(file_data):
        name, content = file_data
        submission_text = read_submission(name, content)
        metrics = extract_metrics(submission_text)

        prompt = f"""
You are an expert evaluator.

STEP 1: ANALYZE
- Identify core solution (model / logic / dashboard)
- Check completeness
- Check dataset usage

STEP 2: METRIC ANALYSIS
Student Metrics:
{metrics}

- High accuracy/R2 → reward
- Low accuracy → reduce score
- Missing metrics → do NOT penalize heavily unless required

STEP 3: SCORE

RULES:
- Use full range (0–10)
- Do NOT give same scores
- Score must reflect quality of work

CONSISTENCY RULE:
- Score > 90 → only minor improvements
- Many improvements → reduce score

STEP 4: FEEDBACK

STRENGTHS:
- Mention actual work done

IMPROVEMENTS:
- Mention clear missing parts

DATASET:
{dataset_summary}

RUBRIC:
{rubric_text}

PROBLEM:
{problem[:500]}

SUBMISSION:
{submission_text}

Return ONLY JSON:
{{
"scores": {{
{', '.join([f'"{c}": number' for c in criteria_list])}
}},
"strengths": "specific strengths",
"improvements": "specific improvements"
}}
"""

        raw = evaluate(prompt)
        parsed = safe_parse(raw)

        scores = parsed.get("scores", {})

        total = 0
        for _, row in rubric_df.iterrows():
            c = row["Criterion"]
            max_s = row["Max Score"]
            val = max(0, min(10, float(scores.get(c, 0))))
            total += (val / 10) * max_s

        total = min(100, total)

        return {
            "file": name,
            "score_100": round(total, 2),
            "strengths": parsed.get("strengths", ""),
            "improvements": parsed.get("improvements", "")
        }

    with st.spinner("Evaluating..."):
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(process_file, files))

    df_out = pd.DataFrame(results)

    st.success("✅ Evaluation Complete")
    st.dataframe(df_out, use_container_width=True)

    buffer = io.BytesIO()
    df_out.to_excel(buffer, index=False)

    st.download_button("Download Excel", buffer, "evaluation_results.xlsx")