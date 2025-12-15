import joblib

from sklearn.linear_model import LinearRegression
from app.ml.data import generate_training_data

X,y=generate_training_data()

model=LinearRegression()

model.fit(X,y)
joblib.dump(model, 'models/linear.pkl')