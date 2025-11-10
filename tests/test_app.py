from fastapi.testclient import TestClient
from app.app import app

client = TestClient(app)

def test_correct_response_structure():
    data = {
    "input_texts" : 
            "Today I am feeling very happy!!"
    }

    response = client.post("/predict", json = data)
    response_json = response.json()
    assert response.status_code == 200
    assert  "status" in response_json.keys()
    assert  "response_body" in response_json.keys()

    response_body = response_json["response_body"][0]
        
    assert  "input_text" in response_body.keys()
    assert  "prediction" in response_body.keys()
    assert  "scores" in response_body.keys()


def test_incorrect_response():
    data = {
    "input_texts" : 
            5
    }
    response = client.post("/predict", json = data)
    response_json = response.json()
    assert response.status_code == 422 # validation error by pedantic


def test_single_prediction():
    input_text = "Today I am feeling very happy!!"
    data = {
        "input_texts" : input_text        
    }

    response = client.post("/predict", json = data)
    response_json = response.json()
    assert response.status_code == 200
    assert response_json["status"] == "successful"

    response_body = response_json["response_body"]
    assert len(response_body) == 1

    response_body = response_body[0]
    assert response_body["input_text"] == input_text
    assert response_body["prediction"] in ["positive", "negative", "neutral"]
    scores = response_body["scores"]
    assert type(scores)==dict
    assert len(scores)==3
    assert list(scores.keys()) == ["negative", "neutral", "positive"]
    for sentiment in scores.keys():
        assert type(scores[sentiment])==float


def test_multiple_predictions():
    input_texts = ["Today I am feeling very happy!!",
                    "Today I am not feeling very happy at all!!",
                    "Today I am feeling no particular mood."]
    data = {
        "input_texts" : input_texts
    }

    response = client.post("/predict", json = data)
    response_json = response.json()
    assert response.status_code == 200
    assert response_json["status"] == "successful"

    response_body = response_json["response_body"]
    assert len(response_body) == len(input_texts)

    for i in range(len(response_body)):
        single_response = response_body[i]
        assert single_response["input_text"] == input_texts[i]
        assert single_response["prediction"] in ["positive", "negative", "neutral"]
        scores = single_response["scores"]
        assert type(scores)==dict
        assert len(scores)==3
        assert list(scores.keys()) == ["negative", "neutral", "positive"]
        for sentiment in scores.keys():
            assert type(scores[sentiment])==float


    

