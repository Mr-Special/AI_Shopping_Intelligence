from sklearn.feature_extraction.text import TfidfVectorizer

def Vectorize(x):
    vectorizer=TfidfVectorizer(max_features=1000)
    x_vec=vectorizer.fit_transform(x)
    return x_vec, vectorizer

    