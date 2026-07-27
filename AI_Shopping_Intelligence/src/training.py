from flaml import AutoML
from sklearn.model_selection import train_test_split
from preprocessor import preprocessor

PATH="data/raw/Main_review.csv"
df=preprocessor(PATH)
print(df)

    