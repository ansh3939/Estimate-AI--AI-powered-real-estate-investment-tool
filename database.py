import os
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime
from typing import Dict, List, Optional
import json

# Database setup
DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable not set")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Property(Base):
    __tablename__ = "properties"
    
    id = Column(Integer, primary_key=True, index=True)
    city = Column(String(100), nullable=False)
    district = Column(String(100), nullable=False)
    sub_district = Column(String(100), nullable=False)
    area_sqft = Column(Float, nullable=False)
    bhk = Column(Integer, nullable=False)
    property_type = Column(String(50), nullable=False)
    furnishing = Column(String(50), nullable=False)
    price_inr = Column(Float, nullable=False)
    price_per_sqft = Column(Float, nullable=False)
    source = Column(String(50), default='Manual')
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)

class PredictionHistory(Base):
    __tablename__ = "prediction_history"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(50), nullable=False)
    city = Column(String(100), nullable=False)
    district = Column(String(100), nullable=False)
    sub_district = Column(String(100), nullable=False)
    area_sqft = Column(Float, nullable=False)
    bhk = Column(Integer, nullable=False)
    property_type = Column(String(50), nullable=False)
    furnishing = Column(String(50), nullable=False)
    predicted_price = Column(Float, nullable=False)
    model_used = Column(String(50), nullable=False)
    investment_score = Column(Integer, nullable=True)
    all_predictions = Column(Text, nullable=True)  # JSON string
    created_at = Column(DateTime, default=datetime.utcnow)

class UserPreferences(Base):
    __tablename__ = "user_preferences"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(50), nullable=False, unique=True)
    preferred_cities = Column(Text, nullable=True)  # JSON array
    preferred_budget_min = Column(Float, nullable=True)
    preferred_budget_max = Column(Float, nullable=True)
    preferred_bhk = Column(Integer, nullable=True)
    preferred_property_type = Column(String(50), nullable=True)
    email_notifications = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)



class DatabaseManager:
    def __init__(self):
        self.engine = engine
        self.SessionLocal = SessionLocal
        
    def create_tables(self):
        """Create all database tables"""
        Base.metadata.create_all(bind=self.engine)
        
    def get_session(self):
        """Get database session"""
        db = self.SessionLocal()
        try:
            return db
        except Exception as e:
            db.close()
            raise e
    
    def close_session(self, db):
        """Close database session"""
        db.close()
    
    def import_csv_data(self, csv_data: pd.DataFrame):
        """Import CSV data into properties table"""
        db = self.get_session()
        try:
            for _, row in csv_data.iterrows():
                property_obj = Property(
                    city=row['City'],
                    district=row['District'],
                    sub_district=row['Sub_District'],
                    area_sqft=row['Area_SqFt'],
                    bhk=row['BHK'],
                    property_type=row['Property_Type'],
                    furnishing=row['Furnishing'],
                    price_inr=row['Price_INR'],
                    price_per_sqft=row.get('Price_per_SqFt', row['Price_INR'] / row['Area_SqFt']),
                    source='CSV_Import'
                )
                db.add(property_obj)
            db.commit()
        except Exception as e:
            db.rollback()
            raise e
        finally:
            self.close_session(db)
    
    def save_prediction(self, session_id: str, input_data: Dict, prediction_result: Dict):
        """Save prediction to history"""
        db = self.get_session()
        try:
            prediction = PredictionHistory(
                session_id=session_id,
                city=input_data['City'],
                district=input_data['District'],
                sub_district=input_data['Sub_District'],
                area_sqft=input_data['Area_SqFt'],
                bhk=input_data['BHK'],
                property_type=input_data['Property_Type'],
                furnishing=input_data['Furnishing'],
                predicted_price=prediction_result['prediction'],
                model_used=prediction_result.get('model_used', 'Unknown'),
                investment_score=prediction_result.get('investment_score'),
                all_predictions=json.dumps(prediction_result.get('all_predictions', {}))
            )
            db.add(prediction)
            db.commit()
            return prediction.id
        except Exception as e:
            db.rollback()
            raise e
        finally:
            self.close_session(db)
    
    def get_prediction_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        """Get prediction history for a session"""
        db = self.get_session()
        try:
            predictions = db.query(PredictionHistory)\
                .filter(PredictionHistory.session_id == session_id)\
                .order_by(PredictionHistory.created_at.desc())\
                .limit(limit).all()
            
            result = []
            for pred in predictions:
                result.append({
                    'id': pred.id,
                    'city': pred.city,
                    'district': pred.district,
                    'sub_district': pred.sub_district,
                    'area_sqft': pred.area_sqft,
                    'bhk': pred.bhk,
                    'property_type': pred.property_type,
                    'furnishing': pred.furnishing,
                    'predicted_price': pred.predicted_price,
                    'model_used': pred.model_used,
                    'investment_score': pred.investment_score,
                    'created_at': pred.created_at
                })
            return result
        finally:
            self.close_session(db)
    

    
    def get_properties_from_db(self, city: str = None, limit: int = 1000) -> pd.DataFrame:
        """Get properties from database"""
        db = self.get_session()
        try:
            query = db.query(Property).filter(Property.is_active == True)
            if city:
                query = query.filter(Property.city == city)
            
            properties = query.limit(limit).all()
            
            data = []
            for prop in properties:
                data.append({
                    'City': str(prop.city),
                    'District': str(prop.district),
                    'Sub_District': str(prop.sub_district),
                    'Area_SqFt': float(prop.area_sqft),
                    'BHK': int(prop.bhk),
                    'Property_Type': str(prop.property_type),
                    'Furnishing': str(prop.furnishing),
                    'Price_INR': float(prop.price_inr),
                    'Price_per_SqFt': float(prop.price_per_sqft),
                    'Source': str(prop.source)
                })
            
            df = pd.DataFrame(data)
            
            # Ensure proper data types after database retrieval
            if not df.empty:
                # Ensure numeric columns are properly typed
                numeric_columns = ['Area_SqFt', 'BHK', 'Price_INR', 'Price_per_SqFt']
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Ensure categorical columns are strings
                categorical_columns = ['City', 'District', 'Sub_District', 'Property_Type', 'Furnishing', 'Source']
                for col in categorical_columns:
                    if col in df.columns:
                        df[col] = df[col].astype(str)
            
            return df
        finally:
            self.close_session(db)
    

    
    def save_user_preferences(self, session_id: str, preferences: Dict):
        """Save user preferences"""
        db = self.get_session()
        try:
            existing_pref = db.query(UserPreferences)\
                .filter(UserPreferences.session_id == session_id)\
                .first()
            
            if existing_pref:
                # Update existing preferences
                existing_pref.preferred_cities = json.dumps(preferences.get('cities', []))
                existing_pref.preferred_budget_min = preferences.get('budget_min')
                existing_pref.preferred_budget_max = preferences.get('budget_max')
                existing_pref.preferred_bhk = preferences.get('bhk')
                existing_pref.preferred_property_type = preferences.get('property_type')
                existing_pref.email_notifications = preferences.get('email_notifications', False)
                existing_pref.updated_at = datetime.utcnow()
            else:
                # Create new preferences
                user_pref = UserPreferences(
                    session_id=session_id,
                    preferred_cities=json.dumps(preferences.get('cities', [])),
                    preferred_budget_min=preferences.get('budget_min'),
                    preferred_budget_max=preferences.get('budget_max'),
                    preferred_bhk=preferences.get('bhk'),
                    preferred_property_type=preferences.get('property_type'),
                    email_notifications=preferences.get('email_notifications', False)
                )
                db.add(user_pref)
            
            db.commit()
        except Exception as e:
            db.rollback()
            raise e
        finally:
            self.close_session(db)
    
    def get_user_preferences(self, session_id: str) -> Dict:
        """Get user preferences"""
        db = self.get_session()
        try:
            pref = db.query(UserPreferences)\
                .filter(UserPreferences.session_id == session_id)\
                .first()
            
            if pref:
                return {
                    'cities': json.loads(pref.preferred_cities) if pref.preferred_cities else [],
                    'budget_min': pref.preferred_budget_min,
                    'budget_max': pref.preferred_budget_max,
                    'bhk': pref.preferred_bhk,
                    'property_type': pref.preferred_property_type,
                    'email_notifications': pref.email_notifications
                }
            return {}
        finally:
            self.close_session(db)
    
    def get_analytics_data(self) -> Dict:
        """Get analytics data for dashboard"""
        db = self.get_session()
        try:
            # Total properties
            total_properties = db.query(Property).filter(Property.is_active == True).count()
            
            # Total predictions
            total_predictions = db.query(PredictionHistory).count()
            
            # Active users (unique sessions in last 30 days)
            from datetime import timedelta
            thirty_days_ago = datetime.utcnow() - timedelta(days=30)
            active_users = db.query(PredictionHistory.session_id)\
                .filter(PredictionHistory.created_at >= thirty_days_ago)\
                .distinct().count()
            
            # Popular cities
            from sqlalchemy import func
            popular_cities = db.query(
                PredictionHistory.city,
                func.count(PredictionHistory.id).label('prediction_count')
            ).group_by(PredictionHistory.city)\
             .order_by(func.count(PredictionHistory.id).desc())\
             .limit(5).all()
            
            return {
                'total_properties': total_properties,
                'total_predictions': total_predictions,
                'active_users': active_users,
                'popular_cities': [(city, count) for city, count in popular_cities]
            }
        finally:
            self.close_session(db)

# Initialize database manager
db_manager = DatabaseManager()