import numpy as np

def generate_training_data(n=500):
    X = []
    y = []

    for _ in range(n):
        exp = np.random.uniform(0, 1)
        skill = np.random.uniform(0, 1)
        interview = np.random.uniform(0, 1)
        salary = np.random.uniform(0, 1)

        score = (
            0.3 * exp
            + 0.4 * skill
            + 0.3 * interview
            - 0.2 * salary
        )

        X.append([exp, skill, interview, salary])
        y.append(score)

    return np.array(X), np.array(y)