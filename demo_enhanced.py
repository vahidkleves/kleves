#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Demo for Thermal Analysis System with Heat Transfer Calculations
"""

from thermal_analyzer_enhanced import EnhancedThermalAnalyzer

def run_enhanced_demo():
    """Run enhanced demo with heat transfer calculations"""
    print("=== Enhanced Thermal Analysis System Demo ===")
    print("With Heat Transfer Coefficient Calculation")
    
    # Create enhanced analyzer
    analyzer = EnhancedThermalAnalyzer()
    
    # 1. Add sample data with calculations
    print("\n1. Adding sample data with automatic heat transfer calculations...")
    sample_cases = [
        # (geometry, equipment_temp, insulation_temp, insulation_type, area, ambient_temp)
        ("pipe", 250, 45, "polyurethane", 2.5, 25),
        ("sphere", 180, 38, "foam", 1.8, 20),
        ("surface", 320, 55, "glass wool", 4.2, 30),
        ("cube", 200, 42, "polyurethane", 3.0, 25),
        ("pipe", 150, 35, "foam", 1.5, 25)
    ]
    
    for geometry, eq_temp, ins_temp, ins_type, area, ambient in sample_cases:
        analyzer.add_manual_data_enhanced(geometry, eq_temp, ins_temp, ins_type, area, ambient)
        print(f"Added data for {geometry} with {ins_type}")
        print("-" * 40)
    
    # 2. Demonstrate convection coefficient calculation
    print("\n2. Calculating convection coefficient from known parameters...")
    print("\nExample: Pipe with polyurethane insulation")
    h_conv = analyzer.calculate_convection_coefficient(
        equipment_temp=200,
        insulation_temp=40,
        insulation_type="polyurethane",
        geometry_type="pipe",
        thickness=0.05,  # 5 cm
        ambient_temp=25
    )
    
    # 3. Train the enhanced model
    print("\n3. Training enhanced prediction model...")
    success = analyzer.train_prediction_model()
    
    if success:
        # 4. Demonstrate complete thermal analysis
        print("\n4. Complete thermal analysis predictions...")
        
        test_scenarios = [
            # (geometry, equipment_temp, insulation_type, area, ambient_temp)
            ("sphere", 220, "polyurethane", 2.0, 25),
            ("surface", 280, "foam", 3.5, 20),
            ("pipe", 160, "glass wool", 1.2, 30),
            ("cube", 300, "polyurethane", 2.8, 25)
        ]
        
        print("\n" + "="*70)
        print("COMPLETE THERMAL ANALYSIS RESULTS")
        print("="*70)
        
        for i, (geometry, eq_temp, ins_type, area, ambient) in enumerate(test_scenarios, 1):
            print(f"\nScenario {i}:")
            print("=" * 50)
            
            results = analyzer.predict_properties(
                equipment_temp=eq_temp,
                geometry_type=geometry,
                insulation_type=ins_type,
                cross_section_area=area,
                ambient_temp=ambient
            )
            
            if results:
                print(f"\nSUMMARY FOR SCENARIO {i}:")
                print(f"Configuration: {geometry} with {ins_type} insulation")
                print(f"Equipment → Insulation: {eq_temp}°C → {results['insulation_temp']:.1f}°C")
                print(f"Heat Transfer Coefficient: {results['convection_coefficient']:.2f} W/m²·K")
                print(f"Heat Flux: {results['heat_flux']:.2f} W/m²")
                print(f"Insulation Efficiency: {results['efficiency']:.1f}%")
            
            print("\n" + "-" * 70)
    
    # 5. Show comparison of different materials
    print("\n5. Material comparison for same conditions...")
    print("\nComparing insulation materials for pipe at 200°C:")
    
    materials = ["polyurethane", "foam", "glass wool"]
    base_conditions = {
        'equipment_temp': 200,
        'geometry_type': 'pipe',
        'cross_section_area': 2.0,
        'ambient_temp': 25
    }
    
    print(f"\n{'Material':<15} {'Insul.Temp':<12} {'Heat Flux':<12} {'h_conv':<12} {'Efficiency':<10}")
    print("-" * 65)
    
    for material in materials:
        results = analyzer.predict_properties(insulation_type=material, **base_conditions)
        if results:
            print(f"{material:<15} {results['insulation_temp']:<12.1f} "
                  f"{results['heat_flux']:<12.1f} {results['convection_coefficient']:<12.1f} "
                  f"{results['efficiency']:<10.1f}%")
    
    # 6. Show database statistics
    print("\n6. Enhanced database statistics:")
    data_list = analyzer.database.get_all_data()
    if len(data_list) > 0:
        print(f"Total records: {len(data_list)}")
        
        h_convs = [d['calculated_h_conv'] for d in data_list if d.get('calculated_h_conv')]
        heat_fluxes = [d['heat_flux'] for d in data_list if d.get('heat_flux')]
        thicknesses = [d['thickness'] for d in data_list if d.get('thickness')]
        
        if h_convs:
            print(f"Convection coefficient range: {min(h_convs):.1f} - {max(h_convs):.1f} W/m²·K")
            print(f"Average convection coefficient: {sum(h_convs)/len(h_convs):.1f} W/m²·K")
        
        if heat_fluxes:
            print(f"Heat flux range: {min(heat_fluxes):.1f} - {max(heat_fluxes):.1f} W/m²")
            print(f"Average heat flux: {sum(heat_fluxes)/len(heat_fluxes):.1f} W/m²")
        
        if thicknesses:
            print(f"Insulation thickness range: {min(thicknesses)*100:.1f} - {max(thicknesses)*100:.1f} cm")
    
    print("\n" + "="*70)
    print("=== Enhanced Demo Complete ===")
    print("Key Features Demonstrated:")
    print("• Automatic heat transfer coefficient calculation")
    print("• Heat flux calculation through insulation")
    print("• Material property database")
    print("• Complete thermal analysis prediction")
    print("• Material comparison capabilities")
    print("\nFor full program usage:")
    print("python3 thermal_analyzer_enhanced.py")
    print("="*70)

if __name__ == "__main__":
    run_enhanced_demo()
