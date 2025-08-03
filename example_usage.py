#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
مثال استفاده از کتابخانه تحلیل انتقال حرارت
"""

from heat_transfer_analyzer import HeatTransferAnalyzer

def example_basic_usage():
    """مثال پایه استفاده از کتابخانه"""
    print("=== مثال پایه استفاده ===")
    
    # ایجاد نمونه analyzer
    analyzer = HeatTransferAnalyzer()
    
    # ایجاد فایل‌های نمونه
    print("1. ایجاد فایل‌های نمونه...")
    analyzer.create_sample_data()
    
    # بارگذاری داده‌ها
    print("\n2. بارگذاری داده‌ها...")
    analyzer.load_colors_data(file_path='sample_colors_output.html')
    analyzer.load_insulation_data('inputdata.xlsx')
    
    # محاسبه نظری
    print("\n3. محاسبه نظری دمای سطح...")
    surface_temp, info = analyzer.calculate_theoretical_surface_temperature()
    print(f"دمای سطح: {surface_temp:.2f} °C")
    print(f"کاهش دما: {info['temperature_reduction']:.2f} °C")
    print(f"کارایی: {info['efficiency_percentage']:.1f}%")
    
    return analyzer

def example_ml_prediction():
    """مثال پیش‌بینی با یادگیری ماشین"""
    print("\n=== مثال یادگیری ماشین ===")
    
    analyzer = HeatTransferAnalyzer()
    
    # ایجاد داده‌های تست متعدد
    print("1. ایجاد داده‌های تست...")
    
    # شبیه‌سازی چندین فایل Colors با پارامترهای مختلف
    test_data = [
        {
            'inner_temperature': 500,
            'ambient_temperature': 25,
            'wind_speed': 3,
            'surface_area': 10.5,
            'equipment_type': 'horizontal_pipe',
            'surface_temperature': 65  # مقدار واقعی برای آموزش
        },
        {
            'inner_temperature': 400,
            'ambient_temperature': 30,
            'wind_speed': 2,
            'surface_area': 8.2,
            'equipment_type': 'vertical_pipe',
            'surface_temperature': 85
        },
        {
            'inner_temperature': 600,
            'ambient_temperature': 20,
            'wind_speed': 4,
            'surface_area': 15.0,
            'equipment_type': 'sphere',
            'surface_temperature': 55
        },
        {
            'inner_temperature': 350,
            'ambient_temperature': 35,
            'wind_speed': 1.5,
            'surface_area': 6.5,
            'equipment_type': 'flat_horizontal',
            'surface_temperature': 95
        },
        {
            'inner_temperature': 750,
            'ambient_temperature': 15,
            'wind_speed': 5,
            'surface_area': 20.0,
            'equipment_type': 'cube',
            'surface_temperature': 45
        }
    ]
    
    analyzer.colors_data = test_data
    analyzer.load_insulation_data('inputdata.xlsx')
    
    # آموزش مدل
    print("\n2. آموزش مدل یادگیری ماشین...")
    results = analyzer.train_ml_model()
    
    if results:
        print("نتایج آموزش:")
        for model_name, metrics in results.items():
            print(f"  {model_name}: R² = {metrics['r2']:.4f}")
    
    # پیش‌بینی برای داده جدید
    print("\n3. پیش‌بینی برای تجهیز جدید...")
    new_equipment = {
        'inner_temperature': 550,
        'ambient_temperature': 25,
        'wind_speed': 3.5,
        'surface_area': 12.0,
        'equipment_type': 'turbine'
    }
    
    # پیش‌بینی با روش‌های مختلف
    methods = ['theoretical', 'ml', 'auto']
    for method in methods:
        try:
            surface_temp, info = analyzer.predict_surface_temperature(
                colors_data=new_equipment, 
                method=method
            )
            print(f"روش {method}: {surface_temp:.2f} °C")
        except Exception as e:
            print(f"روش {method}: خطا - {str(e)}")
    
    return analyzer

def example_comprehensive_analysis():
    """مثال تحلیل کامل"""
    print("\n=== مثال تحلیل کامل ===")
    
    analyzer = HeatTransferAnalyzer()
    
    # بارگذاری داده‌ها
    analyzer.load_colors_data(file_path='sample_colors_output.html')
    analyzer.load_insulation_data('inputdata.xlsx')
    
    # تحلیل کامل تجهیز
    print("1. تحلیل کامل تجهیز...")
    equipment_data = analyzer.colors_data[0]
    results = analyzer.analyze_equipment(equipment_data)
    
    if 'error' not in results:
        print(f"دمای سطح: {results['surface_temperature']:.2f} °C")
        print(f"کارایی: {results['performance']['efficiency_percentage']:.1f}%")
        print(f"ایمن: {'بله' if results['performance']['is_safe'] else 'خیر'}")
        
        if results['recommendations']:
            print("توصیه‌ها:")
            for rec in results['recommendations']:
                print(f"  - {rec}")
    
    # تولید گزارش
    print("\n2. تولید گزارش...")
    report_file = analyzer.generate_report('example_report.json')
    
    # ذخیره مدل (اگر آموزش دیده باشد)
    if analyzer.ml_predictor.is_trained:
        print("\n3. ذخیره مدل...")
        analyzer.save_model('example_model.pkl')
    
    return analyzer

def example_custom_insulation():
    """مثال با عایق سفارشی"""
    print("\n=== مثال عایق سفارشی ===")
    
    analyzer = HeatTransferAnalyzer()
    
    # تعریف لایه‌های عایق سفارشی
    custom_insulation = [
        {
            'insulation_type': 'Cerablanket',
            'thickness': 0.030,  # 30mm
            'density': 96,
            'thermal_conductivity': 0.04
        },
        {
            'insulation_type': 'Rock Wool',
            'thickness': 0.050,  # 50mm
            'density': 100,
            'thermal_conductivity': 0.042
        },
        {
            'insulation_type': 'Silika Needeled Mat',
            'thickness': 0.025,  # 25mm
            'density': 120,
            'thermal_conductivity': 0.038
        }
    ]
    
    # تعریف تجهیز سفارشی
    custom_equipment = {
        'inner_temperature': 450,
        'ambient_temperature': 30,
        'wind_speed': 2.5,
        'surface_area': 8.0,
        'equipment_type': 'valve'
    }
    
    analyzer.insulation_data = custom_insulation
    analyzer.colors_data = [custom_equipment]
    
    # محاسبه دمای سطح
    print("محاسبه با عایق سفارشی...")
    surface_temp, info = analyzer.calculate_theoretical_surface_temperature(
        custom_equipment, custom_insulation
    )
    
    print(f"دمای سطح: {surface_temp:.2f} °C")
    print(f"ضخامت کل عایق: {sum(layer['thickness'] for layer in custom_insulation)*1000:.1f} mm")
    print(f"کاهش دما: {info['temperature_reduction']:.2f} °C")
    
    return analyzer

def example_parameter_study():
    """مثال مطالعه پارامتری"""
    print("\n=== مطالعه پارامتری ===")
    
    analyzer = HeatTransferAnalyzer()
    analyzer.load_insulation_data('inputdata.xlsx')
    
    # پارامترهای پایه
    base_params = {
        'inner_temperature': 500,
        'ambient_temperature': 25,
        'wind_speed': 3,
        'surface_area': 10,
        'equipment_type': 'horizontal_pipe'
    }
    
    # مطالعه تأثیر سرعت باد
    print("تأثیر سرعت باد بر دمای سطح:")
    wind_speeds = [1, 2, 3, 4, 5]
    
    for wind_speed in wind_speeds:
        params = base_params.copy()
        params['wind_speed'] = wind_speed
        
        surface_temp, _ = analyzer.calculate_theoretical_surface_temperature(
            params, analyzer.insulation_data
        )
        
        print(f"  سرعت باد {wind_speed} m/s: دمای سطح {surface_temp:.1f} °C")
    
    # مطالعه تأثیر دمای داخلی
    print("\nتأثیر دمای داخلی بر دمای سطح:")
    inner_temps = [300, 400, 500, 600, 700]
    
    for inner_temp in inner_temps:
        params = base_params.copy()
        params['inner_temperature'] = inner_temp
        
        surface_temp, _ = analyzer.calculate_theoretical_surface_temperature(
            params, analyzer.insulation_data
        )
        
        efficiency = ((inner_temp - surface_temp) / (inner_temp - 25)) * 100
        print(f"  دمای داخلی {inner_temp}°C: دمای سطح {surface_temp:.1f}°C (کارایی: {efficiency:.1f}%)")

def main():
    """اجرای همه مثال‌ها"""
    print("شروع مثال‌های استفاده از کتابخانه تحلیل انتقال حرارت\n")
    
    try:
        # مثال پایه
        analyzer1 = example_basic_usage()
        
        # مثال یادگیری ماشین
        analyzer2 = example_ml_prediction()
        
        # مثال تحلیل کامل
        analyzer3 = example_comprehensive_analysis()
        
        # مثال عایق سفارشی
        analyzer4 = example_custom_insulation()
        
        # مطالعه پارامتری
        example_parameter_study()
        
        print("\n=== همه مثال‌ها با موفقیت اجرا شدند! ===")
        
    except Exception as e:
        print(f"خطا در اجرای مثال‌ها: {str(e)}")

if __name__ == "__main__":
    main()