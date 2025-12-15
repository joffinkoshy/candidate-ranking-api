import joblib
from sklearn.ensemble import GradientBoostingRegressor
from app.ml.data import generate_training_data

X, y = generate_training_data()

model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

model.fit(X, y)

joblib.dump(model, "models/gboost.pkl")