import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


matches = pd.read_csv("archive/matches_updated_ipl_upto_2025.csv")

matches = matches[['team1', 'team2', 'toss_winner',
                   'toss_decision', 'venue', 'winner']]

matches['toss_match_win'] = (
    matches['toss_winner'] == matches['winner']
).astype(int)


matches.dropna(inplace=True)

encoder = LabelEncoder()

encoders = {}

for column in matches.columns:
    le = LabelEncoder()
    matches[column] = le.fit_transform(matches[column])
    encoders[column] = le

    X = matches.drop('winner', axis=1)

y = matches['winner']


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier()

model.fit(X_train, y_train)
joblib.dump(model, "models/random_forest_model.pkl")
joblib.dump(encoders, "models/encoders.pkl")


predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)

matches['toss_match_win'] = (
    matches['toss_winner'] == matches['winner']
).astype(int)


