import requests

url = "http://127.0.0.1:8000/predict"

data = {
    "input_texts" : [
            "Today I am feeling very happy!!",
            "Today I am not feeling very happy at all!!",
            "Today I am feeling no particular mood."]
}


response = requests.post(url, json=data)

if response.status_code == 200:
    response_json = response.json()
    print(response_json["status"])
    for message in response_json["response_body"]:
        print(message)
        
else:
    print(f"error: {response.status_code} - {response.json()}")