from pyfit.logistic_reg import LogisticReg
from pyfit.metrics import accuracy_score
from pyfit.data import make_classification, train_test_split
import numpy as np

def test_logistic_2_classes():
    x_base, y_base = make_classification(
    nb_samples=1000,
    nb_features=2,
    nb_class=2,
)
    x_train, x_test, y_train, y_test = train_test_split(x_base,y_base)
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    model = LogisticReg(x_train,y_train)
    # On test avec l'accuracy
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test,y_pred)
    assert acc >0.9
