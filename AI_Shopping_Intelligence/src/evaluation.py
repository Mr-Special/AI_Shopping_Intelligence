import pandas as pd
import os
import joblib
import json
from sklearn.model_selection import train_test_split
from preprocessor import preprocessor
from sklearn.metrics import(
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

model=joblib.load(r"C:\Users\LENOVO\Desktop\Shopping\AI_Shopping_Intelligence\model\model.pkl")
vectorizer=joblib.load(r"C:\Users\LENOVO\Desktop\Shopping\AI_Shopping_Intelligence\model\vectorizer.pkl")


if not os.path.exists(r"C:\Users\LENOVO\Desktop\Shopping\AI_Shopping_Intelligence\reports"):
    os.makedirs(r"C:\Users\LENOVO\Desktop\Shopping\AI_Shopping_Intelligence\reports")

# df=pd.read_csv(r"C:\Users\LENOVO\Desktop\Shopping\AI_Shopping_Intelligence\data\raw\Main_review.csv")
df=preprocessor(r"C:\Users\LENOVO\Desktop\Shopping\AI_Shopping_Intelligence\data\raw\Main_review.csv")
x=df["text"]
y=df["label"]
x=vectorizer.transform(x)
x_train,x_test,y_train,y_test=train_test_split(
    x,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

prediction=model.predict(x_test)

accuracy=accuracy_score(y_test,prediction)
f1=f1_score(y_test,prediction)

metrics={
    "accuracy":accuracy,
    "f1_score":f1,
    "classification_report":classification_report(y_test,prediction, output_dict=True),
    "confusion_matrix":confusion_matrix(y_test,prediction).tolist(),
}

with open (r"C:\Users\LENOVO\Desktop\Shopping\AI_Shopping_Intelligence\reports\metrics.json","w") as f:
    json.dump(metrics,f,indent=4)
