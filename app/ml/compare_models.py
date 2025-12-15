import joblib
import numpy as np

linear_model = joblib.load("models/linear.pkl")
gb_model=joblib.load("models/gboost.pkl")

candidates = np.array([
    [0.2, 0.8, 0.7, 0.4],
    [0.6, 0.6, 0.6, 0.3],
    [0.9, 0.4, 0.8, 0.6],
    [0.3, 0.9, 0.9, 0.2],
])

linear_scores = linear_model.predict(candidates)
gb_scores = gb_model.predict(candidates)

linear_rank = np.argsort(-linear_scores)
gb_rank = np.argsort(-gb_scores)


print("Candidate | Linear Score | GB Score")
for i in range(len(candidates)):
    print(
        f"{i:9} | {linear_scores[i]:.3f}      | {gb_scores[i]:.3f}"
    )

print("\nLinear ranking:", linear_rank)
print("GB ranking:    ", gb_rank)
