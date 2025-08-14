import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any

class InvestmentAnalyzer:
    def __init__(self):
        # Market growth rates by city (annual %)
        self.city_growth_rates = {
            'Mumbai': 0.08,
            'Delhi': 0.07, 
            'Gurugram': 0.12,
            'Noida': 0.10,
            'Bangalore': 0.11
        }
        
        # Property type appreciation factors
        self.property_factors = {
            'Apartment': 1.0,
            'Builder Floor': 1.1,
            'Independent House': 1.2
        }
        
        # Location factors (multipliers for investment score)
        self.location_factors = {
            'Mumbai': {'Bandra': 1.5, 'Andheri': 1.3, 'Powai': 1.4, 'Thane': 1.2},
            'Delhi': {'Connaught Place': 1.5, 'Karol Bagh': 1.3, 'Lajpat Nagar': 1.2},
            'Gurugram': {'DLF City': 1.4, 'Sector 14': 1.3, 'Golf Course Road': 1.5},
            'Noida': {'Sector 18': 1.3, 'Sector 62': 1.4, 'Greater Noida': 1.2},
            'Bangalore': {'Koramangala': 1.5, 'Whitefield': 1.4, 'Electronic City': 1.3}
        }
    
    def analyze(self, property_data: Dict[str, Any], predicted_price: float) -> Tuple[int, str]:
        """
        Analyze investment potential of a property
        Returns: (investment_score, recommendation_text)
        """
        score = 5  # Base score
        recommendation_points = []
        
        city = property_data['City']
        district = property_data.get('District', '')
        sub_district = property_data.get('Sub_District', '')
        property_type = property_data['Property_Type']
        area_sqft = property_data['Area_SqFt']
        bhk = property_data['BHK']
        
        # Price per square foot analysis
        price_per_sqft = predicted_price / area_sqft
        
        # City growth potential
        if city in self.city_growth_rates:
            growth_rate = self.city_growth_rates[city]
            if growth_rate >= 0.10:
                score += 2
                recommendation_points.append(f"Excellent growth potential in {city} ({growth_rate*100:.0f}% annually)")
            elif growth_rate >= 0.08:
                score += 1
                recommendation_points.append(f"Good growth potential in {city} ({growth_rate*100:.0f}% annually)")
        
        # Property type factor
        if property_type in self.property_factors:
            factor = self.property_factors[property_type]
            if factor >= 1.2:
                score += 1
                recommendation_points.append(f"{property_type} typically shows better appreciation")
            elif factor >= 1.1:
                recommendation_points.append(f"{property_type} shows moderate appreciation potential")
        
        # Location premium
        if city in self.location_factors:
            if sub_district in self.location_factors[city]:
                location_factor = self.location_factors[city][sub_district]
                if location_factor >= 1.4:
                    score += 2
                    recommendation_points.append(f"{sub_district} is a premium location with high demand")
                elif location_factor >= 1.2:
                    score += 1
                    recommendation_points.append(f"{sub_district} is a developing area with good potential")
        
        # Size and configuration analysis
        if bhk >= 3 and area_sqft >= 1200:
            score += 1
            recommendation_points.append("Larger properties typically have better resale value")
        elif bhk == 2 and 800 <= area_sqft <= 1200:
            recommendation_points.append("Good configuration for rental income")
        
        # Price point analysis
        if city == 'Mumbai' and price_per_sqft < 15000:
            score += 1
            recommendation_points.append("Below average price per sq ft for Mumbai - good value")
        elif city == 'Delhi' and price_per_sqft < 12000:
            score += 1
            recommendation_points.append("Below average price per sq ft for Delhi - good value")
        elif city in ['Gurugram', 'Noida'] and price_per_sqft < 8000:
            score += 1
            recommendation_points.append(f"Below average price per sq ft for {city} - good value")
        elif city == 'Bangalore' and price_per_sqft < 7000:
            score += 1
            recommendation_points.append("Below average price per sq ft for Bangalore - good value")
        
        # Market timing factors
        if predicted_price < 2000000:  # Under 20 lakhs
            recommendation_points.append("Entry-level pricing - good for first-time buyers")
        elif predicted_price > 10000000:  # Above 1 crore
            score -= 1
            recommendation_points.append("High-value property - requires significant capital")
        
        # Rental yield estimation
        estimated_monthly_rent = predicted_price * 0.0025  # Assuming 3% annual yield
        annual_rent = estimated_monthly_rent * 12
        rental_yield = (annual_rent / predicted_price) * 100
        
        if rental_yield >= 3.5:
            score += 1
            recommendation_points.append(f"Good rental yield potential (~{rental_yield:.1f}%)")
        elif rental_yield >= 2.5:
            recommendation_points.append(f"Moderate rental yield potential (~{rental_yield:.1f}%)")
        else:
            recommendation_points.append(f"Lower rental yield expected (~{rental_yield:.1f}%)")
        
        # Ensure score is within bounds
        score = max(1, min(10, score))
        
        # Generate recommendation text (plain text format)
        if score >= 8:
            recommendation = "Strong Buy Recommendation\n\n"
        elif score >= 6:
            recommendation = "Good Investment Opportunity\n\n"
        elif score >= 4:
            recommendation = "Consider with Caution\n\n"
        else:
            recommendation = "High Risk Investment\n\n"
        
        # Add detailed points
        recommendation += "Key Factors:\n"
        for point in recommendation_points:
            recommendation += f"• {point}\n"
        
        # Add financial metrics
        recommendation += f"\nFinancial Metrics:\n"
        recommendation += f"• Price per sq ft: ₹{price_per_sqft:,.0f}\n"
        recommendation += f"• Estimated monthly rent: ₹{estimated_monthly_rent:,.0f}\n"
        recommendation += f"• Projected 5-year value: ₹{self._calculate_future_value(predicted_price, city, 5):,.0f}\n"
        
        return score, recommendation
    
    def _calculate_future_value(self, current_price: float, city: str, years: int) -> float:
        """Calculate projected future value based on growth rates"""
        if city in self.city_growth_rates:
            growth_rate = self.city_growth_rates[city]
            return current_price * ((1 + growth_rate) ** years)
        return current_price * (1.06 ** years)  # Default 6% growth
