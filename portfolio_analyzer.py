import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, List
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class PropertyPortfolioAnalyzer:
    def __init__(self):
        # Historical growth rates by city (annual percentage)
        self.city_growth_rates = {
            'Mumbai': 8.5,
            'Delhi': 7.8,
            'Bangalore': 9.2,
            'Hyderabad': 8.9,
            'Chennai': 7.5,
            'Pune': 8.1,
            'Kolkata': 6.8,
            'Ahmedabad': 7.2,
            'Jaipur': 6.9,
            'Lucknow': 7.0,
            'Kanpur': 6.5,
            'Nagpur': 7.1,
            'Indore': 7.3,
            'Thane': 8.3,
            'Bhopal': 6.7,
            'Visakhapatnam': 7.4,
            'Pimpri-Chinchwad': 7.9,
            'Patna': 6.4,
            'Vadodara': 7.0,
            'Ghaziabad': 7.6
        }
        
        # Market cycle indicators
        self.market_sentiment = {
            'Mumbai': 'Hold', 'Delhi': 'Buy', 'Bangalore': 'Hold',
            'Hyderabad': 'Buy', 'Chennai': 'Hold', 'Pune': 'Buy'
        }
        
    def analyze_current_property_value(self, purchase_data: Dict[str, Any], current_predictor) -> Dict[str, Any]:
        """Analyze current value of owned property"""
        
        # Get current market prediction
        current_prediction, _ = current_predictor.predict(purchase_data)
        
        # Calculate appreciation
        purchase_price = purchase_data['purchase_price']
        purchase_year = purchase_data.get('purchase_year', 2020)
        current_year = datetime.now().year
        years_held = max(1, current_year - purchase_year)  # At least 1 year
        
        # Calculate actual growth
        total_growth = ((current_prediction - purchase_price) / purchase_price) * 100
        annual_growth = total_growth / years_held if years_held > 0 else 0
        
        # Market benchmark growth
        city = purchase_data.get('city', purchase_data.get('City', 'Mumbai'))
        expected_annual_growth = self.city_growth_rates.get(city, 7.5)
        expected_current_value = purchase_price * (1 + expected_annual_growth/100) ** years_held
        expected_total_growth = ((expected_current_value - purchase_price) / purchase_price) * 100
        
        # Performance analysis
        performance_vs_market = annual_growth - expected_annual_growth
        
        return {
            'purchase_price': purchase_price,
            'current_value': current_prediction,
            'total_appreciation': current_prediction - purchase_price,
            'total_growth_percent': total_growth,
            'annual_growth_percent': annual_growth,
            'annualized_return': annual_growth,  # Alias for UI compatibility
            'years_held': years_held,
            'expected_current_value': expected_current_value,
            'expected_growth_percent': expected_total_growth,
            'performance_vs_market': performance_vs_market,
            'market_benchmark': expected_annual_growth
        }
    
    def generate_hold_sell_recommendation(self, property_analysis: Dict[str, Any], 
                                        property_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate buy/sell/hold recommendation"""
        
        city = property_data.get('city', property_data.get('City', 'Mumbai'))
        annual_growth = property_analysis['annual_growth_percent']
        performance_vs_market = property_analysis['performance_vs_market']
        years_held = property_analysis['years_held']
        
        # Scoring factors
        growth_score = min(10, max(0, annual_growth))
        market_performance_score = min(5, max(-5, performance_vs_market))
        holding_period_score = min(3, years_held * 0.5)  # Bonus for longer holds
        
        # Market sentiment score
        sentiment = self.market_sentiment.get(city, 'Hold')
        sentiment_score = {'Buy': 2, 'Hold': 0, 'Sell': -2}.get(sentiment, 0)
        
        total_score = growth_score + market_performance_score + holding_period_score + sentiment_score
        
        # Generate recommendation
        if total_score >= 12:
            recommendation = "STRONG HOLD"
            reasoning = f"Excellent performance ({annual_growth:.1f}% annual growth). Property is outperforming market by {performance_vs_market:.1f}%."
        elif total_score >= 8:
            recommendation = "HOLD"
            reasoning = f"Good performance ({annual_growth:.1f}% annual growth). Continue holding for steady appreciation."
        elif total_score >= 5:
            recommendation = "CONDITIONAL HOLD"
            reasoning = f"Average performance ({annual_growth:.1f}% annual growth). Monitor market conditions closely."
        elif total_score >= 2:
            recommendation = "CONSIDER SELLING"
            reasoning = f"Below-average performance ({annual_growth:.1f}% annual growth). Consider selling if you need liquidity."
        else:
            recommendation = "SELL"
            reasoning = f"Poor performance ({annual_growth:.1f}% annual growth). Consider selling and reinvesting elsewhere."
        
        # Future projections
        next_year_value = property_analysis['current_value'] * (1 + self.city_growth_rates.get(city, 7.5)/100)
        five_year_value = property_analysis['current_value'] * (1 + self.city_growth_rates.get(city, 7.5)/100) ** 5
        
        return {
            'recommendation': recommendation,
            'reasoning': reasoning,
            'confidence_score': min(100, max(0, total_score * 8)),
            'next_year_projection': next_year_value,
            'five_year_projection': five_year_value,
            'market_sentiment': sentiment,
            'key_factors': {
                'annual_growth': annual_growth,
                'market_performance': performance_vs_market,
                'holding_period': years_held,
                'market_sentiment': sentiment
            }
        }
    
    def analyze_investment_opportunity(self, target_property: Dict[str, Any], 
                                     budget: float, current_predictor) -> Dict[str, Any]:
        """Analyze if a property at given price is a good investment"""
        
        # Get market prediction for the property
        predicted_value, all_predictions = current_predictor.predict(target_property)
        
        # Calculate confidence based on model consistency (difference between predictions)
        confidence = 85.0  # Default confidence
        try:
            if len(all_predictions) > 1:
                # Safely extract numeric predictions
                predictions_list = []
                for pred in all_predictions.values():
                    try:
                        numeric_pred = float(pred)
                        if not np.isnan(numeric_pred) and np.isfinite(numeric_pred):
                            predictions_list.append(numeric_pred)
                    except (ValueError, TypeError):
                        continue
                
                if len(predictions_list) > 1:
                    predictions_array = np.array(predictions_list, dtype=np.float64)
                    std_dev = float(np.std(predictions_array))
                    mean_pred = float(np.mean(predictions_array))
                    if mean_pred > 0:
                        confidence = float(max(0.0, 100.0 - (std_dev / mean_pred * 100)))
        except Exception:
            confidence = 85.0
        
        city = target_property['City']
        asking_price = budget
        
        # Value analysis
        value_gap = predicted_value - asking_price
        value_gap_percent = (value_gap / asking_price) * 100
        
        # Growth projections
        annual_growth_rate = self.city_growth_rates.get(city, 7.5)
        one_year_value = asking_price * (1 + annual_growth_rate/100)
        three_year_value = asking_price * (1 + annual_growth_rate/100) ** 3
        five_year_value = asking_price * (1 + annual_growth_rate/100) ** 5
        
        # Investment scoring
        value_score = min(10, max(-5, value_gap_percent))
        growth_potential_score = min(10, annual_growth_rate)
        location_score = min(5, len(target_property.get('District', ''))) # Basic location scoring
        
        # Market timing score
        sentiment = self.market_sentiment.get(city, 'Hold')
        timing_score = {'Buy': 5, 'Hold': 2, 'Sell': -3}.get(sentiment, 0)
        
        total_investment_score = value_score + growth_potential_score + location_score + timing_score
        
        # Generate recommendation
        if total_investment_score >= 20:
            investment_recommendation = "STRONG BUY"
            investment_reasoning = f"Excellent opportunity! Property is undervalued by {abs(value_gap_percent):.1f}% with strong growth potential."
        elif total_investment_score >= 15:
            investment_recommendation = "BUY"
            investment_reasoning = f"Good investment opportunity with {annual_growth_rate:.1f}% expected annual growth."
        elif total_investment_score >= 10:
            investment_recommendation = "CONDITIONAL BUY"
            investment_reasoning = f"Fair opportunity. Consider if price can be negotiated down by 5-10%."
        elif total_investment_score >= 5:
            investment_recommendation = "AVOID"
            investment_reasoning = f"Property appears overvalued by {abs(value_gap_percent):.1f}%. Wait for better opportunities."
        else:
            investment_recommendation = "STRONG AVOID"
            investment_reasoning = f"Poor investment. Property significantly overvalued with limited growth potential."
        
        # ROI calculations
        three_year_roi = ((three_year_value - asking_price) / asking_price) * 100
        five_year_roi = ((five_year_value - asking_price) / asking_price) * 100
        
        return {
            'asking_price': asking_price,
            'predicted_market_value': predicted_value,
            'fair_market_value': predicted_value,  # Alias for UI compatibility
            'value_gap': value_gap,
            'value_gap_percent': value_gap_percent,
            'budget_adequacy': min(100, max(0, (predicted_value / asking_price) * 100)),
            'projected_roi_annual': annual_growth_rate,
            'investment_attractiveness_score': min(100, max(0, total_investment_score * 5)),
            'investment_recommendation': investment_recommendation,
            'reasoning': investment_reasoning,
            'detailed_analysis': investment_reasoning,  # Alias for UI compatibility
            'risk_level': 'Medium' if total_investment_score > 10 else 'High' if total_investment_score > 5 else 'Very High',
            'confidence_score': min(100, max(0, total_investment_score * 4)),
            'growth_projections': {
                'one_year': one_year_value,
                'three_year': three_year_value,
                'five_year': five_year_value
            },
            'roi_projections': {
                'three_year_roi': three_year_roi,
                'five_year_roi': five_year_roi,
                'annual_growth_rate': annual_growth_rate
            },
            'market_sentiment': sentiment,
            'investment_score': total_investment_score
        }
    
    def create_portfolio_dashboard(self, analysis_data: Dict[str, Any]) -> go.Figure:
        """Create comprehensive portfolio analysis dashboard"""
        
        if 'property_analysis' in analysis_data:
            # Property value tracking chart
            analysis = analysis_data['property_analysis']
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Value Growth Over Time', 'Performance vs Market', 
                               'Future Projections', 'Investment Score'),
                specs=[[{"secondary_y": False}, {"type": "indicator"}],
                       [{"secondary_y": False}, {"type": "indicator"}]]
            )
            
            # Value growth timeline
            years = np.arange(0, analysis['years_held'] + 5, 0.5)
            purchase_price = analysis['purchase_price']
            growth_rate = analysis['annual_growth_percent'] / 100
            
            actual_values = [purchase_price * (1 + growth_rate) ** year for year in years[:int(analysis['years_held']*2)+1]]
            projected_values = [analysis['current_value'] * (1 + growth_rate) ** (year - analysis['years_held']) 
                              for year in years[int(analysis['years_held']*2):]]
            
            # Actual growth line
            fig.add_trace(go.Scatter(
                x=years[:len(actual_values)], 
                y=actual_values,
                mode='lines+markers',
                name='Actual Value',
                line=dict(color='#2E7D32', width=3)
            ), row=1, col=1)
            
            # Projected growth line
            if len(projected_values) > 0:
                fig.add_trace(go.Scatter(
                    x=years[int(analysis['years_held']*2):], 
                    y=projected_values,
                    mode='lines',
                    name='Projected Value',
                    line=dict(color='#4CAF50', width=2, dash='dash')
                ), row=1, col=1)
            
            # Performance indicator
            fig.add_trace(go.Indicator(
                mode="gauge+number+delta",
                value=analysis['performance_vs_market'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "vs Market (%)"},
                delta={'reference': 0},
                gauge={'axis': {'range': [-5, 10]},
                       'bar': {'color': "#2E7D32"},
                       'steps': [{'range': [-5, 0], 'color': "lightgray"},
                                {'range': [0, 5], 'color': "lightgreen"},
                                {'range': [5, 10], 'color': "green"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 0}}
            ), row=1, col=2)
            
            return fig
        
        return go.Figure()
    
    def generate_market_timing_analysis(self, city: str) -> Dict[str, Any]:
        """Generate market timing analysis for a city"""
        
        current_growth = self.city_growth_rates.get(city, 7.5)
        sentiment = self.market_sentiment.get(city, 'Hold')
        
        # Market cycle analysis
        if current_growth > 8.5:
            cycle_phase = "Growth Phase"
            timing_advice = "Good time to buy before prices peak"
        elif current_growth > 7.0:
            cycle_phase = "Stable Phase"
            timing_advice = "Balanced market - suitable for both buying and selling"
        else:
            cycle_phase = "Correction Phase"
            timing_advice = "Wait for market stabilization before major investments"
        
        return {
            'city': city,
            'current_growth_rate': current_growth,
            'market_sentiment': sentiment,
            'cycle_phase': cycle_phase,
            'market_phase': cycle_phase,  # Alias for UI compatibility
            'timing_advice': timing_advice,
            'phase_description': timing_advice,  # Alias for UI compatibility  
            'recommended_action': sentiment.upper(),
            'action_reason': timing_advice,
            'recommendation_confidence': 75
        }