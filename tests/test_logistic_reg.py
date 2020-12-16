from pyfit.logistic_reg import LogisticReg
from pyfit.metrics import accuracy_score
from pyfit.data import make_classification, train_test_split
import numpy as np
from sklearn.linear_model import SGDClassifier

def test_logistic_2():
    x_base, y_base = make_classification(
        nb_samples=1000,
        nb_features=2,
        nb_class=2,
    )
    x_train, x_test, y_train, y_test = train_test_split(x_base,y_base)
    model = LogisticReg(x_train,y_train)
    sk_model = SGDClassifier(loss="log")
    sk_model.fit(x_train, y_train)
    # On test avec l'accuracy
    y_pred = model.predict(x_test)
    y_pred_sk = sk_model.predict(x_test)
    acc = accuracy_score(y_test,y_pred)
    acc_sk = accuracy_score(y_test, y_pred_sk)
    assert acc_sk > 0.9
    assert acc > 0.9


def test_logistic_multiple():
    x_base, y_base = make_classification(
    nb_samples = 1000,
    nb_features = 4,
    nb_class = 2,
)
    x_train, x_test, y_train, y_test = train_test_split(x_base, y_base)
    model = LogisticReg(x_train, y_train)
    # On test avec l'accuracy
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    assert acc > 0.9