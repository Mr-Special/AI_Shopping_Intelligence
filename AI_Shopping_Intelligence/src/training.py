from flaml import AutoML
import joblib
from FeatureExtraction import Vectorize
from sklearn.model_selection import train_test_split
from preprocessor import preprocessor
from pathlib import Path


# PATH="data/raw/Main_review.csv"
# PATH=r"C:\Users\LENOVO\Desktop\Shopping\AI_Shopping_Intelligence\data\raw\Main_review.csv"
path=Path(r"C:\Users\LENOVO\Desktop\Shopping\AI_Shopping_Intelligence\data\raw\Main_review.csv")
df=preprocessor(path)
print(df)

def training(x,y):
    x, vectorizer = Vectorize(x)
    x_train,x_test,y_train,y_test=train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )
    model=AutoML()
    model.fit(
        X_train=x_train,
        y_train=y_train,
        task="classification",
        time_budget=60
    )
    
    # Save both the model and the vectorizer
    model_path=r"C:\Users\LENOVO\Desktop\Shopping\AI_Shopping_Intelligence\model\model.pkl"
    # vectorizer_path=r"C:\Users\LENOVO\Desktop\Shopping\AI_Shopping_Intelligence\model\vectorizer.pkl"
    joblib.dump(model, model_path)
    # joblib.dump(vectorizer, vectorizer_path)
    print("Model and Vectorizer trained and saved successfully")
    print(f"The Best Model found is {model.best_estimator}")


x=df['text']
y=df['label']
training(x,y)

