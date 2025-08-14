import math
from typing import Dict

class EMICalculator:
    def __init__(self):
        pass
    
    def calculate_emi(self, principal: float, annual_rate: float, tenure_years: int) -> Dict[str, float]:
        """
        Calculate EMI and related details
        
        Args:
            principal: Loan amount in INR
            annual_rate: Annual interest rate in percentage
            tenure_years: Loan tenure in years
            
        Returns:
            Dictionary with EMI details
        """
        # Convert annual rate to monthly rate
        monthly_rate = annual_rate / (12 * 100)
        
        # Convert years to months
        tenure_months = tenure_years * 12
        
        # Calculate EMI using formula: EMI = P * r * (1+r)^n / ((1+r)^n - 1)
        if monthly_rate == 0:  # Handle zero interest rate
            emi = principal / tenure_months
        else:
            emi = principal * monthly_rate * (1 + monthly_rate) ** tenure_months / \
                  ((1 + monthly_rate) ** tenure_months - 1)
        
        # Calculate totals
        total_amount = emi * tenure_months
        total_interest = total_amount - principal
        
        return {
            'emi': round(emi, 2),
            'total_amount': round(total_amount, 2),
            'total_interest': round(total_interest, 2),
            'principal': principal,
            'monthly_rate': monthly_rate * 100,
            'tenure_months': tenure_months
        }
    
    def generate_amortization_schedule(self, principal: float, annual_rate: float, 
                                     tenure_years: int, num_periods: int = 12) -> list:
        """
        Generate amortization schedule for the first num_periods
        
        Args:
            principal: Loan amount
            annual_rate: Annual interest rate in percentage
            tenure_years: Loan tenure in years
            num_periods: Number of periods to show (default: 12 months)
            
        Returns:
            List of dictionaries with payment details
        """
        emi_details = self.calculate_emi(principal, annual_rate, tenure_years)
        emi = emi_details['emi']
        monthly_rate = annual_rate / (12 * 100)
        
        schedule = []
        outstanding_balance = principal
        
        for month in range(1, min(num_periods + 1, tenure_years * 12 + 1)):
            # Calculate interest for current month
            interest_payment = outstanding_balance * monthly_rate
            
            # Calculate principal payment
            principal_payment = emi - interest_payment
            
            # Update outstanding balance
            outstanding_balance -= principal_payment
            
            schedule.append({
                'month': month,
                'emi': round(emi, 2),
                'principal': round(principal_payment, 2),
                'interest': round(interest_payment, 2),
                'outstanding': round(max(0, outstanding_balance), 2)
            })
            
            # Break if loan is fully paid
            if outstanding_balance <= 0:
                break
        
        return schedule
    
    def calculate_prepayment_savings(self, principal: float, annual_rate: float, 
                                   tenure_years: int, prepayment_amount: float, 
                                   prepayment_month: int) -> Dict[str, float]:
        """
        Calculate savings from prepayment
        
        Args:
            principal: Original loan amount
            annual_rate: Annual interest rate
            tenure_years: Original tenure in years
            prepayment_amount: Additional payment amount
            prepayment_month: Month when prepayment is made
            
        Returns:
            Dictionary with savings details
        """
        # Original EMI calculation
        original_emi_details = self.calculate_emi(principal, annual_rate, tenure_years)
        original_emi = original_emi_details['emi']
        original_total_interest = original_emi_details['total_interest']
        
        # Calculate remaining balance after prepayment month
        monthly_rate = annual_rate / (12 * 100)
        outstanding = principal
        
        # Reduce outstanding balance till prepayment month
        for month in range(prepayment_month):
            interest_payment = outstanding * monthly_rate
            principal_payment = original_emi - interest_payment
            outstanding -= principal_payment
        
        # Apply prepayment
        outstanding_after_prepayment = outstanding - prepayment_amount
        
        # Calculate new tenure for remaining amount
        if outstanding_after_prepayment <= 0:
            new_tenure_months = prepayment_month
            interest_saved = original_total_interest
        else:
            # Calculate remaining months with same EMI
            remaining_months = math.log(1 + (outstanding_after_prepayment * monthly_rate / original_emi)) / \
                             math.log(1 + monthly_rate)
            new_tenure_months = prepayment_month + math.ceil(remaining_months)
            
            # Calculate new total interest
            new_total_payment = (original_emi * new_tenure_months) + prepayment_amount
            new_total_interest = new_total_payment - principal
            interest_saved = original_total_interest - new_total_interest
        
        tenure_saved_months = (tenure_years * 12) - new_tenure_months
        
        return {
            'interest_saved': round(max(0, interest_saved), 2),
            'tenure_reduction': max(0, tenure_saved_months),
            'new_emi': original_emi,
            'new_tenure_months': new_tenure_months,
            'outstanding_after_prepayment': round(max(0, outstanding_after_prepayment), 2)
        }
