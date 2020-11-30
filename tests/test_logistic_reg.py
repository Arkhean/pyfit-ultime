from pyfit.logistic_reg import LogisticReg
from metrics import accuracy_score
from data import make_classification
import numpy as np

def test_logistic_2_classes():
    x_base, y_base = make_classification(
    n_samples=1000,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    random_state=26,
    n_clusters_per_class=1,
)
    sets = train_test_split(x_base,y_base)
    model = LogisticReg(x_train,y_train)
    # On test avec l'accuracy

    y_pred = mosel.predict(x_test)
    acc = accuracy_score(y_test,_pred)
