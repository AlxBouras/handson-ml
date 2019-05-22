from utility import tool_q1 as util_q1
from sklearn import datasets, pipeline, model_selection
from sklearn.externals import joblib
from sklearn.ensemble import GradientBoostingClassifier

def main():
    books = datasets.load_files("data/Book/", shuffle=True, encoding="ISO-8859-1", random_state=1337)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(books.data, books.target,
                                                                        test_size=0.10)

    model = pipeline.Pipeline([
        ('union', pipeline.FeatureUnion(
            transformer_list=[
                ('other_features', util_q1.AddOtherFeatures(feature_to_add="pos_neg_count")),
                ('text_data', pipeline.Pipeline([
                    ('remove_words', util_q1.RemoveWords(words_to_remove="none")),
                    ("normalisation", util_q1.NormaliseWords(normalise_type="lemmatize")),
                    ("preprocess", util_q1.PreprocessData(attribute="frequency_filtering",
                                                          attribute_values="tf-idf")),
                ]))
            ]
        )),
        ("classifier", GradientBoostingClassifier(n_estimators=60, max_features="sqrt", subsample=0.8))
    ])

    scoring = {"accuracy": "accuracy", "recall": "recall", "precision": "precision"}

    grid_search_model = model_selection.GridSearchCV(
        model,
        {
            "classifier__max_depth": range(5, 11, 5),
            "classifier__min_samples_split": range(5, 11, 5)
        },
        n_jobs=-1, verbose=10, scoring=scoring, refit=False
    )

    grid_search_model.fit(X_train, y_train)
    joblib.dump(grid_search_model, "outputs/gridsearch_xgboost_aws.pkl")


if __name__ == "__main__":
    main()