#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
English Demo for Thermal Insulation Analysis System
Simple Demo for Thermal Insulation Analysis System
"""

from thermal_analyzer_english import ThermalAnalyzer

def run_english_demo():
    """Run English demo of the program"""
    print("=== Thermal Insulation Analysis System Demo ===")
    print("No external dependencies required")
    
    # Create analyzer instance
    analyzer = ThermalAnalyzer()
    
    # 1. Import sample HTML files
    print("\n1. Importing sample HTML files...")
    analyzer.import_html_files("./html_files")
    
    # 2. Add some manual data
    print("\n2. Adding manual data...")
    sample_data = [
        ("cube", 280, 45, "polyurethane", 3.0, 16),
        ("pipe", 150, 28, "foam", 1.5, 10),
        ("surface", 200, 38, "glass wool", 2.8, 14),
        ("sphere", 190, 32, "polyurethane", 2.2, 13),
        ("pipe", 310, 52, "glass wool", 3.5, 18)
    ]
    
    for geometry, eq_temp, ins_temp, ins_type, area, coeff in sample_data:
        analyzer.add_manual_data(geometry, eq_temp, ins_temp, ins_type, area, coeff)
        print(f"Data for {geometry} added.")
    
    # 3. Train model
    print("\n3. Training prediction model...")
    success = analyzer.train_prediction_model()
    
    if success:
        # 4. Test predictions
        print("\n4. Testing predictions for new geometries...")
        test_cases = [
            ("sphere", 220, 2.0, 13, "polyurethane"),
            ("surface", 180, 3.5, 11, "foam"),
            ("pipe", 300, 1.2, 20, "glass wool"),
            ("cube", 160, 2.5, 12, "foam")
        ]
        
        print("\n" + "="*60)
        print("Prediction Results:")
        print("="*60)
        
        for geometry, eq_temp, area, coeff, ins_type in test_cases:
            prediction = analyzer.predict_insulation_temperature(
                eq_temp, area, coeff, geometry, ins_type
            )
            
            if prediction:
                temp_reduction = eq_temp - prediction
                efficiency = (temp_reduction / eq_temp) * 100
                
                print(f"\nGeometry: {geometry} | Insulation: {ins_type}")
                print(f"Equipment temperature: {eq_temp}°C")
                print(f"Predicted insulation temperature: {prediction:.1f}°C")
                print(f"Temperature reduction: {temp_reduction:.1f}°C ({efficiency:.1f}%)")
                print("-" * 40)
    
    # 5. Display data statistics
    print("\n5. Overall data statistics:")
    data_list = analyzer.database.get_all_data()
    if len(data_list) > 0:
        print(f"Total records: {len(data_list)}")
        
        geometries = list(set([d['geometry_type'] for d in data_list]))
        insulations = list(set([d['insulation_type'] for d in data_list]))
        eq_temps = [d['equipment_surface_temp'] for d in data_list]
        ins_temps = [d['insulation_surface_temp'] for d in data_list]
        
        print(f"Available geometries: {geometries}")
        print(f"Insulation types: {insulations}")
        print(f"Equipment temperature range: {min(eq_temps):.1f} - {max(eq_temps):.1f}°C")
        print(f"Insulation temperature range: {min(ins_temps):.1f} - {max(ins_temps):.1f}°C")
        
        # Calculate average efficiency
        total_efficiency = 0
        for data in data_list:
            reduction = data['equipment_surface_temp'] - data['insulation_surface_temp']
            eff = (reduction / data['equipment_surface_temp']) * 100
            total_efficiency += eff
        
        avg_efficiency = total_efficiency / len(data_list)
        print(f"Average insulation efficiency: {avg_efficiency:.1f}%")
    
    print("\n" + "="*60)
    print("=== Demo Complete ===")
    print("For full program usage:")
    print("python3 thermal_analyzer_english.py")
    print("="*60)

if __name__ == "__main__":
    run_english_demo()
