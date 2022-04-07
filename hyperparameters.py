# CHANTRE Honorine  CHAH2807
# THOMAS Eliott THOE2303


from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, LogisticRegression, Perceptron
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

import numpy as np


pipeline_sgd = Pipeline([('tfidf', TfidfVectorizer()),
                     ('clf', SGDClassifier()),
])

pipeline_gb = Pipeline([('tfidf', TfidfVectorizer()),
                     ('clf', GradientBoostingClassifier()),
])

pipeline_rf = Pipeline([('tfidf', TfidfVectorizer()),
                     ('clf', RandomForestClassifier()),
])

pipeline_lr = Pipeline([('tfidf', TfidfVectorizer()),
                     ('clf', LogisticRegression(max_iter=1000)),
])

pipeline_per = Pipeline([('tfidf', TfidfVectorizer()),
                     ('clf', Perceptron()),
])

pipeline_svc = Pipeline([('tfidf', TfidfVectorizer()),
                     ('clf', LinearSVC(max_iter=10000)),
])



MODELS_AND_PARAMS = {
    "SGD" : { "model" : pipeline_sgd,
              "params" : {
                    'clf__penalty': ['l2', 'l1', 'elasticnet'],
                    'clf__alpha': np.linspace(1e-10, 1e-4, 40),  
                        }
    },
    "GB" : { "model" : pipeline_gb,
              "params" : {
                    "clf__learning_rate": [0.1, 0.2,0.4,0.8],
                    "clf__n_estimators":[1600]#,3200]
                        }
    },
    "RF" : { "model" : pipeline_rf,
              "params" : {
                    'clf__n_estimators': [100, 200],   
                    'clf__max_depth': [300,600,None],
                    'clf__min_samples_leaf': [1, 2, 3],   
                    'clf__min_samples_split': [16, 32, 64],
                    'clf__max_features': ['log2', 'sqrt'],
                    'clf__criterion': ['gini', 'entropy']
                        }
    },
    "LR" : { "model" : pipeline_lr,
              "params" : {
                    'clf__C': [20,10, 1],
                    'clf__tol': np.linspace(1e-12,1e-6,20)
                        }
    },
    "PER" : { "model" : pipeline_per,
              "params" : {
                    'clf__penalty': ['l2', 'l1', 'elasticnet'],
                    'clf__alpha': np.linspace(1e-8, 1e-4, 100),
                        }
    },
    "SVC" : { "model" : pipeline_svc,
              "params" : {
                    'clf__penalty': ['l2', 'l1', 'elasticnet'],
                    'clf__loss': ['hinge', 'squared_hinge'],
                    'clf__dual' : [False,True]
                        }
    }
    

}

def grid_Search(model_name, X_train, y_train, subset=-1):

    model = MODELS_AND_PARAMS[model_name]["model"]
    parameters = MODELS_AND_PARAMS[model_name]["params"]

    grid_clf = GridSearchCV(model, parameters,  scoring='accuracy', verbose=1 ,n_jobs=-1)
    
    if subset==-1:
        grid_clf.fit(X_train, y_train)
    else:
        grid_clf.fit(X_train[:subset], y_train[:subset])


    print("Best Score: ", grid_clf.best_score_)
    print("Best Params: ", grid_clf.best_params_)

    return grid_clf