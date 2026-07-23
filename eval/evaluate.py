"""Offline evaluation harness.

Answers the interview question "how do you know it works?" with numbers instead
of vibes: run the ranker over a hand-labeled golden set and compare its order to
the human order using two metrics.

  - Kendall's tau: rank correlation in [-1, 1] (1 = identical order).
  - Top-3 overlap: fraction of the human top-3 the system also puts in its top-3.

Run:  python -m eval.evaluate     (from the candidate-ranking-api/ directory)
"""

import json
import os

from scipy.stats import kendalltau

from app.schemas import Candidate
from app.services.ranking import rank_candidates_logic

GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_set.json")


def load_golden(path: str = GOLDEN_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def evaluate(golden: dict) -> dict:
    human_rank = {c["candidate_id"]: c["human_rank"] for c in golden["candidates"]}
    candidates = [
        Candidate(**{k: v for k, v in c.items() if k != "human_rank"})
        for c in golden["candidates"]
    ]

    ranked = rank_candidates_logic(candidates, golden["job_description"])
    predicted_rank = {r.candidate_id: r.rank for r in ranked}

    # Aligned rank vectors (same candidate order) for correlation.
    ids = sorted(human_rank)
    human_vec = [human_rank[i] for i in ids]
    pred_vec = [predicted_rank[i] for i in ids]
    tau, _ = kendalltau(human_vec, pred_vec)

    human_top3 = {i for i in ids if human_rank[i] <= 3}
    pred_top3 = {r.candidate_id for r in ranked if r.rank <= 3}
    top3_overlap = len(human_top3 & pred_top3) / 3.0

    return {
        "kendall_tau": tau,
        "top3_overlap": top3_overlap,
        "ranking": ranked,
        "human_rank": human_rank,
    }


def main() -> None:
    result = evaluate(load_golden())

    print("Predicted vs. human ranking")
    print("-" * 52)
    print(f"{'cand':<6}{'pred':<6}{'human':<7}{'final':<9}{'judge':<7}")
    for r in result["ranking"]:
        judge = "-" if r.judge_score is None else f"{r.judge_score:.0f}"
        print(f"{r.candidate_id:<6}{r.rank:<6}{result['human_rank'][r.candidate_id]:<7}"
              f"{r.final_score:<9.3f}{judge:<7}")

    print("-" * 52)
    print(f"Kendall's tau : {result['kendall_tau']:.3f}   (1.0 = perfect order)")
    print(f"Top-3 overlap : {result['top3_overlap']:.0%}")


if __name__ == "__main__":
    main()
