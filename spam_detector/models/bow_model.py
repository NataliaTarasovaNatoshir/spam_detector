import os
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from joblib import dump, load


class BoWModel():
    def __init__(self, model_name='', vectorizer_params={}, classifier_params={}, load_from_file=False, filepath=None):
        if load_from_file:
            self.model_name = filepath.split('.')[-1]
            self.pipeline = load(filepath)
        else:
            self.model_name = model_name
            self.pipeline = Pipeline(
                [
                    ("vectorization", TfidfVectorizer(**vectorizer_params)),
                    ("classification", SGDClassifier(**classifier_params))
                ]
        )

    def train(self, X, y):
        X_train_joined = X.copy()
        # fill empty values
        X_train_joined['text'].fillna('', inplace=True)
        X_train_joined['subject'].fillna('', inplace=True)
        # join subject and mail body
        X_train_joined['joined_text'] = X_train_joined[['subject', 'text']].apply(
            lambda x: str(x[0]) + ' ' + str(x[1]), axis=1)
        # train the model
        self.pipeline.fit(X_train_joined['joined_text'].values, y)

    def predict(self, X):
        X_valid_joined = X.copy()
        # fill empty values
        X_valid_joined['text'].fillna('', inplace=True)
        X_valid_joined['subject'].fillna('', inplace=True)
        # join subject and mail body
        X_valid_joined['joined_text'] = X_valid_joined[['subject', 'text']].apply(
            lambda x: str(x[0]) + ' ' + str(x[1]), axis=1)
        return self.pipeline.predict_proba(X_valid_joined['joined_text'].values)[:,1]

    def dump(self, folder):
        dump(self.pipeline, os.path.join(folder, self.model_name + '.joblib'))


