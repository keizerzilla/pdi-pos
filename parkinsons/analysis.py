import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

df = pd.read_csv("parkinsons.csv")
print("#atributos:", len(df.columns))

scores = pd.read_csv("scores.csv")
f1s = pd.read_csv("f1s.csv")
mccs = pd.read_csv("mccs.csv")
voting = pd.read_csv("voting.csv")

print(scores.mean())
print(f1s.mean())
print(mccs.mean())

score_vote = accuracy_score(voting["true"], voting["voted"])
f1_vote = f1_score(voting["true"], voting["voted"])
print("voting:", round(score_vote, 2), round(f1_vote, 2))

