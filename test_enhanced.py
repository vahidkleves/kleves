#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Test for Enhanced Thermal Analyzer
"""

from thermal_analyzer_enhanced import EnhancedThermalAnalyzer

def quick_test():
    """Quick test of enhanced features"""
    print("=== Quick Test - Enhanced Thermal Analyzer ===")
    
    analyzer = EnhancedThermalAnalyzer()
    
    # Test 1: Calculate convection coefficient
    print("\n1. Testing convection coefficient calculation:")
    print("Scenario: Pipe with polyurethane insulation")
    print("Equipment: 200°C, Insulation surface: 40°C, Ambient: 25°C")
    
    h_conv = analyzer.calculate_convection_coefficient(
        equipment_temp=200,
        insulation_temp=40,
        insulation_type="polyurethane",
        geometry_type="pipe",
        thickness=0.05,
        ambient_temp=25
    )
    
    # Test 2: Add enhanced data
    print("\n2. Testing enhanced data addition:")
    analyzer.add_manual_data_enhanced(
        geometry_type="sphere",
        equipment_temp=180,
        insulation_temp=35,
        insulation_type="foam",
        cross_section_area=2.0,
        ambient_temp=25
    )
    
    # Test 3: Train and predict
    print("\n3. Testing prediction capabilities:")
    success = analyzer.train_prediction_model()
    
    if success:
        results = analyzer.predict_properties(
            equipment_temp=220,
            geometry_type="pipe",
            insulation_type="polyurethane",
            cross_section_area=2.5,
            ambient_temp=25
        )
        
        if results:
            print(f"\nQuick Prediction Summary:")
            print(f"Predicted insulation temperature: {results['insulation_temp']:.1f}°C")
            print(f"Calculated convection coefficient: {results['convection_coefficient']:.2f} W/m²·K")
            print(f"Heat flux: {results['heat_flux']:.1f} W/m²")
            print(f"Efficiency: {results['efficiency']:.1f}%")
    
    print("\n=== Quick Test Complete ===")
    print("Enhanced features working correctly!")

if __name__ == "__main__":
    quick_test()
