from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_get():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"greeting": "Welcome User!"}


def test_post_model_below():
    data_test = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }

    r = client.post("/adult/", json=data_test)
    assert r.status_code == 200
    assert r.json() == {"fetch": "The estimated salary is <=50K"}


def test_post_model_above():
    data_test = {
        "age": 37,
        "workclass": "Private",
        "fnlgt": 280464,
        "education": " Some-college",
        "education-num": 10,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "Black",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 80,
        "native-country": "United-States"
    }

    r = client.post("/adult/", json=data_test)
    assert r.status_code == 200
    assert r.json() == {"fetch": "The estimated salary is >50K"}
