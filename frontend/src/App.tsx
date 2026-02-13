// src/components/HousePricePrediction.tsx
import React, { useState } from 'react';

const HousePricePrediction = () => {
  // State for form inputs
  const [formData, setFormData] = useState({
    overallQual: 5,
    grLivArea: '',
    garageCars: '',
    totalBsmtSF: '',
  });

  const [loading, setLoading] = useState(false);
  const [predictedPrice, setPredictedPrice] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;

    if (name === 'overallQual') {
    setFormData(prev => ({
      ...prev,
      [name]: Number(value),
    }));
  } else {
    setFormData(prev => ({
      ...prev,
      [name]: value === '' ? '' : value, //allow empty string 
    })); 
  }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      const data = await response.json();

      if (data.success) {
        setPredictedPrice(data.prediction);
      } else {
        setError(data.error || 'Prediction failed');
      }
    } catch (err) {
      setError('Failed to connect to the server. Make sure Flask is running. (app.py)');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-linear-to-br from-gray-900 to-gray-800 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        {/* Title */}
        <h1 className="text-4xl md:text-5xl font-bold text-center text-white mb-10 tracking-tight">
          House Price Prediction
        </h1>

        {/* Card */}
        <div className="bg-linear-to-b from-emerald-800 to-emerald-950 rounded-2xl shadow-2xl shadow-emerald-950/40 overflow-hidden border border-emerald-700/40">
          <div className="p-8 md:p-10">
            <form onSubmit={handleSubmit} className="space-y-8">
              {/* Overall Quality */}
              <div className="space-y-2">
                <label className="block text-emerald-100 font-medium">
                  Overall Quality (1-10)
                </label>
                <input
                  type="range"
                  name="overallQual"
                  min="1"
                  max="10"
                  value={formData.overallQual}
                  onChange={handleChange}
                  className="w-full h-2 bg-emerald-900 rounded-lg appearance-none cursor-pointer accent-emerald-400"
                />
                <div className="text-center text-emerald-300 font-semibold text-xl">
                  {formData.overallQual}
                </div>
              </div>

              {/* Living Area */}
              <div className="space-y-2">
                <label className="block text-emerald-100 font-medium">
                  Living Area (sqft)
                </label>
                <input
                  type="number"
                  name="grLivArea"
                  min="300"
                  max="8000"
                  value={formData.grLivArea}
                  onChange={handleChange}
                  className="w-full px-4 py-3 bg-emerald-950 border border-emerald-700 rounded-lg text-white placeholder-emerald-400 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent transition-all"
                  placeholder="Enter living area in square feet"
                />
              </div>

              {/* Garage Cars */}
              <div className="space-y-2">
                <label className="block text-emerald-100 font-medium">
                  Garage Cars
                </label>
                <input
                  type="number"
                  name="garageCars"
                  min="0"
                  max="4"
                  step="1"
                  value={formData.garageCars}
                  onChange={handleChange}
                  className="w-full px-4 py-3 bg-emerald-950 border border-emerald-700 rounded-lg text-white placeholder-emerald-400 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent transition-all"
                  placeholder="Number of cars that fit in garage"
                />
              </div>

              {/* Total Basement Area */}
              <div className="space-y-2">
                <label className="block text-emerald-100 font-medium">
                  Total Basement Area (sqft)
                </label>
                <input
                  type="number"
                  name="totalBsmtSF"
                  min="0"
                  max="6000"
                  value={formData.totalBsmtSF}
                  onChange={handleChange}
                  className="w-full px-4 py-3 bg-emerald-950 border border-emerald-700 rounded-lg text-white placeholder-emerald-400 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent transition-all"
                  placeholder="Total basement area in square feet"
                />
              </div>

              {/* Prediction Result */}
              {predictedPrice !== null && (
                <div className="bg-emerald-900/50 border border-emerald-600 rounded-xl p-6 text-center">
                  <p className="text-emerald-200 text-sm mb-2">Predicted Price</p>
                  <p className="text-3xl font-bold text-emerald-300">
                    ${predictedPrice.toLocaleString()}
                  </p>
                </div>
              )}

              {/* Error Message */}
              {error && (
                <div className="bg-red-900/50 border border-red-600 rounded-xl p-4 text-center">
                  <p className="text-red-200 text-sm">{error}</p>
                </div>
              )}

              {/* Submit Button */}
              <div className="pt-6">
                <button
                  type="submit"
                  disabled={loading}
                  className={`
                    w-full py-4 px-6 rounded-xl font-bold text-lg
                    transition-all duration-300 flex items-center justify-center gap-3
                    ${
                      loading
                        ? 'bg-emerald-700 cursor-not-allowed'
                        : 'bg-emerald-500 hover:bg-emerald-400 active:bg-emerald-600 shadow-lg shadow-emerald-700/30 hover:shadow-emerald-500/50'
                    }
                    text-gray-900
                  `}
                >
                  {loading ? (
                    <>
                      <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                      </svg>
                      Predicting...
                    </>
                  ) : (
                    <>
                      <span>Predict Price</span>
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                      </svg>
                    </>
                  )}
                </button>
              </div>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HousePricePrediction;