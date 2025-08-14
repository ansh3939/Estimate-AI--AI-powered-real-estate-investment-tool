import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import pickle
import os
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

class FastRealEstatePredictor:
    def __init__(self):
        # Initialize all three allowed models
        self.models = {
            'decision_tree': DecisionTreeRegressor(
                random_state=42,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features=None
            ),
            'random_forest': RandomForestRegressor(
                random_state=42, 
                n_estimators=100,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features=None,
                bootstrap=True,
                n_jobs=-1
            ),
            'xgboost': xgb.XGBRegressor(
                random_state=42,
                n_estimators=100,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1
            )
        }
        
        # Best performing model will be selected after training
        self.best_model = None
        self.best_model_name = None
        self.model_scores = {}
        
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = [
            'City', 'District', 'Sub_District', 'Area_SqFt', 
            'BHK', 'Property_Type', 'Furnishing'
        ]
        self.model_trained = False
        self.cache_file = 'fast_model_cache.pkl'
        
    def _encode_categorical_features(self, data: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Encode categorical features using label encoders"""
        encoded_data = data.copy()
        categorical_columns = ['City', 'District', 'Sub_District', 'Property_Type', 'Furnishing']
        
        for column in categorical_columns:
            if column in encoded_data.columns:
                try:
                    if fit:
                        if column not in self.label_encoders:
                            self.label_encoders[column] = LabelEncoder()
                        encoded_data[column] = self.label_encoders[column].fit_transform(encoded_data[column].astype(str))
                    else:
                        if column in self.label_encoders:
                            # Handle unseen categories
                            known_categories = set(self.label_encoders[column].classes_)
                            encoded_data[column] = encoded_data[column].astype(str).apply(
                                lambda x: x if x in known_categories else 'Unknown'
                            )
                            # Add 'Unknown' to encoder if not present
                            if 'Unknown' not in known_categories:
                                self.label_encoders[column].classes_ = np.append(self.label_encoders[column].classes_, 'Unknown')
                            encoded_data[column] = self.label_encoders[column].transform(encoded_data[column])
                        else:
                            # If no encoder exists, use default value
                            print(f"Warning: No encoder found for {column}, using default value 0")
                            encoded_data[column] = 0
                except Exception as e:
                    print(f"Error encoding {column}: {e}")
                    # Set default value on error
                    encoded_data[column] = 0
        return encoded_data
    
    def _create_simple_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create basic features for faster processing"""
        enhanced_data = data.copy()
        
        # Check if required columns exist, if not skip feature creation
        if 'Area_SqFt' in enhanced_data.columns and 'BHK' in enhanced_data.columns:
            # Only essential features for speed
            enhanced_data['Area_Per_Room'] = enhanced_data['Area_SqFt'] / enhanced_data['BHK'].replace(0, 1)
            enhanced_data['Area_Squared'] = enhanced_data['Area_SqFt'] ** 2
        else:
            # Add default features if columns are missing
            enhanced_data['Area_Per_Room'] = 0
            enhanced_data['Area_Squared'] = 0
        
        return enhanced_data
    
    def load_cached_model(self) -> bool:
        """Load cached model if available"""
        # Disable caching for now to ensure fresh training
        return False
    
    def save_model_cache(self):
        """Save trained model to cache"""
        try:
            cache_data = {
                'best_model': self.best_model,
                'best_model_name': self.best_model_name,
                'model_scores': self.model_scores,
                'encoders': self.label_encoders,
                'scaler': self.scaler
            }
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
        except:
            pass
    
    def train_model(self, data: pd.DataFrame) -> Dict[str, float]:
        """Train all three models and select the best performer"""
        # Try to load cached model first
        if self.load_cached_model():
            return {"mae": 0, "r2_score": 0.9, "cached": True, "best_model": self.best_model_name}
        
        print("Training all models: Decision Tree, Random Forest, and XGBoost...")
        
        # Map column names to expected format
        column_mapping = {
            'city': 'City',
            'district': 'District', 
            'sub_district': 'Sub_District',
            'area_sqft': 'Area_SqFt',
            'bhk': 'BHK',
            'property_type': 'Property_Type',
            'furnishing': 'Furnishing',
            'price_inr': 'Price_INR'
        }
        
        # Rename columns in training data
        mapped_data = data.copy()
        for old_name, new_name in column_mapping.items():
            if old_name in mapped_data.columns:
                mapped_data = mapped_data.rename(columns={old_name: new_name})
        
        # Prepare features
        enhanced_data = self._create_simple_features(mapped_data)
        enhanced_data = self._encode_categorical_features(enhanced_data, fit=True)
        
        # Select features
        X = enhanced_data[self.feature_columns + ['Area_Per_Room', 'Area_Squared']]
        y = enhanced_data['Price_INR']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features for tree-based models (helps with consistency)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train and evaluate all models
        best_score = -float('inf')
        
        for model_name, model in self.models.items():
            print(f"Training {model_name}...")
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Store scores
            self.model_scores[model_name] = {
                'mae': mae,
                'r2_score': r2
            }
            
            print(f"{model_name} - R²: {r2:.3f}, MAE: ₹{mae:,.0f}")
            
            # Select best model based on R² score
            if r2 > best_score:
                best_score = r2
                self.best_model = model
                self.best_model_name = model_name
        
        print(f"\nBest performing model: {self.best_model_name}")
        print(f"Best R² Score: {best_score:.3f}")
        
        self.model_trained = True
        self.save_model_cache()
        
        return {
            'mae': self.model_scores[self.best_model_name]['mae'],
            'r2_score': self.model_scores[self.best_model_name]['r2_score'],
            'best_model': self.best_model_name,
            'all_scores': self.model_scores,
            'cached': False
        }
    
    def predict(self, input_data: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """Make prediction using the best trained model"""
        if not self.model_trained or self.best_model is None:
            raise ValueError("Models not trained yet!")
        
        # Map lowercase column names to expected format
        column_mapping = {
            'city': 'City',
            'district': 'District', 
            'sub_district': 'Sub_District',
            'area_sqft': 'Area_SqFt',
            'bhk': 'BHK',
            'property_type': 'Property_Type',
            'furnishing': 'Furnishing',
            'price_inr': 'Price_INR'
        }
        
        # Convert input data to expected column names
        mapped_data = {}
        for key, value in input_data.items():
            mapped_key = column_mapping.get(key, key)
            mapped_data[mapped_key] = value
        
        # Create DataFrame from mapped input
        input_df = pd.DataFrame([mapped_data])
        
        # Debug: Print column names
        print(f"Debug - Input columns after mapping: {list(input_df.columns)}")
        
        # Create features
        enhanced_input = self._create_simple_features(input_df)
        print(f"Debug - Enhanced columns: {list(enhanced_input.columns)}")
        
        encoded_input = self._encode_categorical_features(enhanced_input, fit=False)
        print(f"Debug - Encoded columns: {list(encoded_input.columns)}")
        print(f"Debug - Expected feature columns: {self.feature_columns + ['Area_Per_Room', 'Area_Squared']}")
        
        # Select features
        X = encoded_input[self.feature_columns + ['Area_Per_Room', 'Area_Squared']]
        
        # Scale and predict
        X_scaled = self.scaler.transform(X)
        prediction = self.best_model.predict(X_scaled)[0]
        
        # Get predictions from all models for comparison
        all_predictions = {}
        for model_name, model in self.models.items():
            try:
                pred = model.predict(X_scaled)[0]
                all_predictions[model_name] = float(pred)
            except:
                all_predictions[model_name] = float(prediction)
        
        all_predictions['best_model'] = self.best_model_name
        
        return float(prediction), all_predictions
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the best trained model"""
        if not self.model_trained or self.best_model is None:
            return {}
        
        try:
            feature_names = self.feature_columns + ['Area_Per_Room', 'Area_Squared']
            importance_dict = dict(zip(feature_names, self.best_model.feature_importances_))
            return dict(sorted(importance_dict.items(), key=lambda x: float(x[1]), reverse=True))
        except:
            return {}