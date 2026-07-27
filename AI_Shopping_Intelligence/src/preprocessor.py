import pandas as pd
import numpy as np 

if __name__ == "__main__":
    PATH=r"C:\Users\LENOVO\Desktop\Shopping\AI_Shopping_Intelligence\data\raw\Main_review.csv"
    df=pd.read_csv(PATH)

    df.rename(columns={"text_":"text"},inplace=True)
    df["label"]=(df["label"]=='CG').astype(int)

    df.to_csv(r"C:\Users\LENOVO\Desktop\Shopping\AI_Shopping_Intelligence\data\processed\processed_review.csv",index=False)


def preprocessor(path):
    df=pd.read_csv(path)
    df.rename(columns={"text_":"text"},inplace=True)
    df["label"]=(df["label"]=='CG').astype(int)

    # df.to_csv("data/processed/processed_review.csv",index=False)
    return df