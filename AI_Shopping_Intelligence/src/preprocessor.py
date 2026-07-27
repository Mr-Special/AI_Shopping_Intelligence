import pandas as pd
import numpy as np 




def preprocessor(path):
    df=pd.read_csv(path)
    df.rename(columns={"text_":"text"},inplace=True)
    df["label"]=(df["label"]=='CG').astype(int)

    # df.to_csv("data/processed/processed_review.csv",index=False)
    return df

