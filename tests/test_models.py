import numpy as np

from molml.models import build_model


def test_model_inference_schema():
    x = np.array([[0,0], [0,1], [1,0], [1,1], [0,0], [1,1]])
    y = np.array([0,0,0,1,0,1])
    model = build_model("logistic_regression")
    model.fit(x, y)
    pred = model.predict(x)
    prob = model.predict_proba(x)
    assert pred.shape == (6,)
    assert prob.shape == (6, 2)
