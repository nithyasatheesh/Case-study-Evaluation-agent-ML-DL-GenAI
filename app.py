import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
from docx import Document
from PyPDF2 import PdfReader
import zipfile
import io
import json
import re
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

# -------------------------
# STREAMLIT CONFIG
# -------------------------
st.set_page_config(
    page_title="AI Case Study Evaluator",
    layout="wide"
)

st.title("📊 AI Case Study Evaluator")

# -------------------------
# OPENAI CLIENT
# -------------------------
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception:
    st.error("❌ OpenAI API Key not found.")
    st.stop()

# -------------------------
# FILE UPLOADS
# -------------------------
problem_file = st.file_uploader(
    "Upload Problem Statement",
    type=["docx", "pdf"]
)

rubric_file = st.file_uploader(
    "Upload Rubric",
    type=["xlsx"]
)

zip_file = st.file_uploader(
    "Upload Student Submissions ZIP",
    type=["zip"]
)

dataset_file = st.file_uploader(
    "Upload Dataset (Optional)",
    type=["csv"]
)

sample_file = st.file_uploader(
    "Upload Sample Submission (Optional)",
    type=["html", "txt"]
)

custom_instructions = st.text_area(
    "📝 Additional Evaluation Instructions"
)

# -------------------------
# HELPERS
# -------------------------
def read_docx(file):
    doc = Document(file)
    return "\n".join([p.text for p in doc.paragraphs])

def read_pdf(file):
    reader = PdfReader(file)
    text = []

    for page in reader.pages:
        text.append(page.extract_text() or "")

    return "\n".join(text)

def read_rubric(file):
    return pd.read_excel(file)

def rubric_to_text(df):
    return "\n".join(
        f"{row['Criterion']} "
        f"(Max {row['Max Score']}): "
        f"{row['Description']}"
        for _, row in df.iterrows()
    )

def read_submission(name, content):

    text = content.decode("utf-8", errors="ignore")

    if name.endswith(".html"):
        soup = BeautifulSoup(text, "html.parser")

        for tag in soup(["script", "style"]):
            tag.decompose()

        text = soup.get_text(separator=" ")

    return text[:8000]

def extract_zip(zip_uploaded):

    files = []

    with zipfile.ZipFile(io.BytesIO(zip_uploaded.read()), "r") as z:

        for name in z.namelist():

            if name.endswith((".txt", ".html")):
                files.append((name, z.read(name)))

    return files

def safe_parse(raw):

    try:
        return json.loads(raw)

    except Exception:

        try:
            match = re.search(r"\{.*\}", raw, re.DOTALL)

            if match:
                return json.loads(match.group())

        except Exception:
            pass

    return {
        "scores": {},
        "strengths": "Could not parse response",
        "improvements": "Try again"
    }

# -------------------------
# DATASET SUMMARY
# -------------------------
def get_dataset_summary(file):

    try:
        df = pd.read_csv(file)

        return f"""
Rows: {df.shape[0]}

Columns:
{list(df.columns)}

Sample:
{df.head(3).to_string(index=False)}
"""

    except Exception:
        return ""

# -------------------------
# METRICS EXTRACTION
# -------------------------
def extract_metrics(text):

    metrics = {}

    acc = re.findall(
        r'accuracy\s*[:=]\s*([0-9\.]+)',
        text,
        re.IGNORECASE
    )

    if acc:
        metrics["accuracy"] = acc[0]

    r2 = re.findall(
        r'r2[_ ]?score\s*[:=]\s*([0-9\.]+)',
        text,
        re.IGNORECASE
    )

    if r2:
        metrics["r2_score"] = r2[0]

    return metrics

# -------------------------
# OPENAI EVALUATION
# -------------------------
def evaluate_submission(prompt):

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict academic evaluator. "
                    "Return only valid JSON."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content

# -------------------------
# MAIN BUTTON
# -------------------------
if st.button("🚀 Evaluate Submissions"):

    if not (problem_file and rubric_file and (zip_file or sample_file)):
        st.error(
            "Please upload Problem, Rubric, and at least one Submission."
        )
        st.stop()

    # -------------------------
    # READ INPUTS
    # -------------------------
    if problem_file.name.endswith(".docx"):
        problem = read_docx(problem_file)
    else:
        problem = read_pdf(problem_file)

    rubric_df = read_rubric(rubric_file)

    rubric_text = rubric_to_text(rubric_df)

    criteria_list = rubric_df["Criterion"].tolist()

    dataset_summary = (
        get_dataset_summary(dataset_file)
        if dataset_file else ""
    )

    files = []

    if zip_file:
        files.extend(extract_zip(zip_file))

    if sample_file:
        files.append(
            (sample_file.name, sample_file.read())
        )

    st.info(f"📂 Total Submissions: {len(files)}")

    # -------------------------
    # PROCESS FILE
    # -------------------------
    def process_file(file_data):

        name, content = file_data

        submission_text = read_submission(name, content)

        metrics = extract_metrics(submission_text)

        prompt = f"""
You are a STRICT evaluator.

EVALUATION STRATEGY:
- Follow rubric strictly
- Start from full marks and deduct
- Identify weak areas
- Avoid giving identical scores

METRICS FOUND:
{metrics}

CUSTOM INSTRUCTIONS:
{custom_instructions}

DATASET:
{dataset_summary}

RUBRIC:
{rubric_text}

PROBLEM:
{problem[:1500]}

SUBMISSION:
{submission_text[:6000]}

OUTPUT FORMAT:
{{
    "scores": {{
        {', '.join([f'"{c}": number' for c in criteria_list])}
    }},
    "strengths": "specific strengths",
    "improvements": "specific improvements"
}}
"""

        raw = evaluate_submission(prompt)

        parsed = safe_parse(raw)

        scores = parsed.get("scores", {})

        total = 0

        for _, row in rubric_df.iterrows():

            criterion = row["Criterion"]

            max_score = row["Max Score"]

            value = float(scores.get(criterion, 0))

            value = max(0, min(10, value))

            total += (value / 10) * max_score

        return {
            "File": name,
            "Final Score": round(total, 2),
            "Strengths": parsed.get("strengths", ""),
            "Improvements": parsed.get("improvements", "")
        }

    # -------------------------
    # RUN PARALLEL EVALUATION
    # -------------------------
    with st.spinner("Evaluating submissions..."):

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(
                executor.map(process_file, files)
            )

    df_out = pd.DataFrame(results)

    st.success("✅ Evaluation Complete")

    st.dataframe(
        df_out,
        use_container_width=True
    )

    # -------------------------
    # DOWNLOAD
    # -------------------------
    output = io.BytesIO()

    df_out.to_excel(
        output,
        index=False,
        engine="openpyxl"
    )

    st.download_button(
        "📥 Download Results",
        output.getvalue(),
        file_name="evaluation_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
