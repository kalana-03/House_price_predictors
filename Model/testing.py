import requests
import json

# Test data
test_data = {
    'overallQual': 7,
    'grLivArea': 2000,
    'garageCars': 2,
    'totalBsmtSF': 1200
}

# Test the API
try:
    response = requests.post(
        'http://localhost:5000/predict',
        json=test_data,
        headers={'Content-Type': 'application/json'}
    )
    
    print("Status Code:", response.status_code)
    print("Response:", json.dumps(response.json(), indent=2))
    
except Exception as e:
    print(f"Error: {e}")