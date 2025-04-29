import requests

url = "http://127.0.0.1:8000/predict"

payload = {
    "odometer": 50000,
    "age": 26 
}

headers = {
    "accept": "application/json",
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)

print("Status code:", response.status_code)
print("Response:", response.json())
