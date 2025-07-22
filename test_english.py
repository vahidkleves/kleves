#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive Test for English Thermal Analysis System
Interactive Test for Thermal Analysis System
"""

from thermal_analyzer_english import ThermalAnalyzer

def interactive_test_english():
    """Interactive test of English program"""
    print("=== Interactive Test - Thermal Analysis System ===")
    
    analyzer = ThermalAnalyzer()
    
    # Train model with existing data
    print("Training model with existing data...")
    data_list = analyzer.database.get_all_data()
    
    if len(data_list) >= 3:
        analyzer.predictor.train_model(data_list)
        print("Model is ready for prediction!")
        
        # Test prediction
        print("\nTesting prediction:")
        test_result = analyzer.predict_insulation_temperature(
            200, 2.5, 15, "pipe", "polyurethane"
        )
        
        if test_result:
            print(f"For pipe at 200°C: Predicted insulation temperature = {test_result:.1f}°C")
            print(f"Temperature reduction: {200 - test_result:.1f}°C")
            print(f"Efficiency: {((200 - test_result)/200)*100:.1f}%")
        
    else:
        print("Insufficient data for training.")
        print("Please run demo_english.py first.")

if __name__ == "__main__":
    interactive_test_english()
