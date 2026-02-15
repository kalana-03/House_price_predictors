from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
# Allows requests from other ports
CORS(app)  

# Load the model
try:
    model = joblib.load('Model/house_price_predictor.pkl')
    print("Model loaded successfully!....")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/predict', methods=['POST'])
def predict_price():
    if model is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded. Please check if house_price_predictor.pkl exists.'
        }), 500
    
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Extract features in the EXACT order the model was trained
        # Order: OverallQual, GrLivArea, GarageCars, TotalBsmtSF
        overall_qual = float(data['overallQual'])
        gr_liv_area = float(data['grLivArea'])
        garage_cars = float(data['garageCars'])
        total_bsmt_sf = float(data['totalBsmtSF'])
        
        # Validate inputs
        if not (1 <= overall_qual <= 10):
            raise ValueError("OverallQual must be between 1 and 10")
        if gr_liv_area < 0:
            raise ValueError("Living area cannot be negative")
        if garage_cars < 0:
            raise ValueError("Garage cars cannot be negative")
        if total_bsmt_sf < 0:
            raise ValueError("Basement area cannot be negative")
        
        # Create input array in correct order
        input_data = np.array([[overall_qual, gr_liv_area, garage_cars, total_bsmt_sf]])
        
        print(f"Input data: {input_data}")  # Debug logging
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction = round(prediction, 2)
        
        print(f"Prediction: ${prediction}")  # Debug logging
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'inputs': {
                'overallQual': overall_qual,
                'grLivArea': gr_liv_area,
                'garageCars': garage_cars,
                'totalBsmtSF': total_bsmt_sf
            }
        })
        
    except KeyError as e:
        return jsonify({
            'success': False,
            'error': f'Missing required field: {str(e)}'
        }), 400
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Prediction error: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)