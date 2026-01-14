# ML-Powered Candidate Ranking API

## Overview
Hiring teams often need to compare multiple candidates for a single job opening across factors such as experience, skills, interview performance, and salary expectations. Manual comparison is subjective, time-consuming, and does not scale.

This project implements a **machine learning–powered backend API** that ranks candidates for a given job using a learned scoring function. The system is designed to be modular, explainable, and production-ready.

---

## System Architecture

The application follows a clean separation of concerns:

- **API Layer (FastAPI)**  
  Handles request validation, routing, and response formatting.

- **Business Logic Layer**  
  Performs feature normalization, scoring, and deterministic ranking.

- **Machine Learning Layer**  
  Trains regression models and serves them for inference.
- ---

## Data Model

Each candidate is represented using the following features:

- Years of experience
- Skill match score
- Interview performance score
- Salary expectation

Candidates are submitted per job, and ranking is performed **within the context of a single job**, not globally.

Pydantic schemas are used to enforce strict input and output validation.

---

## Scoring and Ranking Approach

### Feature Processing
- Numerical features are normalized per request using min–max normalization.
- Salary expectation is treated as a negative signal relative to other features.

### Model-Based Scoring
A regression model predicts a final suitability score for each candidate.

Two models were implemented and compared:
- **Linear Regression** – interpretable baseline model  
- **Gradient Boosting Regressor** – higher-capacity non-linear model

The backend supports configurable model selection without modifying business logic.

### Ranking
Candidates are ranked by predicted score in descending order.  
Ties are resolved deterministically to ensure stable and reproducible output.

---

## API Endpoints

### Rank Candidates
**POST** `/rank_candidates`

Accepts a job ID and a list of candidates, returns ranked candidates with predicted scores.

### Health Check
**GET** `/health`

Returns service and model status.  
Used for readiness checks and operational monitoring.

---

## Design Decisions

- **FastAPI** was chosen for its performance, type safety, and automatic schema documentation.
- **Regression models** allow continuous scoring rather than hard classification.
- **Model inference is decoupled from API logic**, enabling easy upgrades and experimentation.
- **Per-request normalization** ensures fair comparison among candidates for the same job.

---

---

## Future Improvements

- Model versioning and A/B testing
- Online learning using hiring feedback
- Feature expansion (education, certifications)
- Containerized deployment and cloud hosting

---

## Tech Stack

- Python
- FastAPI
- Scikit-learn
- NumPy
- Pydantic
- Uvicorn
- Pytest

