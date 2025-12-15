from app.nlp.similarity import compute_similarity

job = """Senior Machine Learning Engineer with Python, PyTorch,
experience in model deployment and data pipelines."""

resumes = [
    ("ML Engineer", "Machine learning engineer with PyTorch, Python, deployed models"),
    ("Data Analyst", "Data analyst using SQL, dashboards, basic ML"),
    ("Backend Dev", "Backend engineer building APIs in Python and FastAPI"),
    ("Sales", "Sales manager handling retail operations and clients"),
    ("HR", "HR professional managing recruitment and payroll"),
]

for label, text in resumes:
    score = compute_similarity(job, text)
    print(f"{label:15s} -> {score:.3f}")