#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
دموی ساده سیستم تحلیل انتقال حرارت
Simple Demo for Thermal Insulation Analysis System
"""

from simple_thermal_analyzer import ThermalAnalyzer

def run_simple_demo():
    """اجرای دموی ساده برنامه"""
    print("=== دموی ساده سیستم تحلیل انتقال حرارت ===")
    print("بدون نیاز به پکیج‌های خارجی")
    
    # ایجاد نمونه تحلیلگر
    analyzer = ThermalAnalyzer()
    
    # 1. وارد کردن فایل‌های HTML نمونه
    print("\n1. وارد کردن فایل‌های HTML نمونه...")
    analyzer.import_html_files("./html_files")
    
    # 2. افزودن چند داده دستی
    print("\n2. افزودن داده‌های دستی...")
    sample_data = [
        ("مکعب", 280, 45, "پلی اورتان", 3.0, 16),
        ("لوله", 150, 28, "فوم", 1.5, 10),
        ("سطح", 200, 38, "پشم شیشه", 2.8, 14),
        ("کره", 190, 32, "پلی اورتان", 2.2, 13),
        ("لوله", 310, 52, "پشم شیشه", 3.5, 18)
    ]
    
    for geometry, eq_temp, ins_temp, ins_type, area, coeff in sample_data:
        analyzer.add_manual_data(geometry, eq_temp, ins_temp, ins_type, area, coeff)
        print(f"داده {geometry} اضافه شد.")
    
    # 3. آموزش مدل
    print("\n3. آموزش مدل پیش‌بینی...")
    success = analyzer.train_prediction_model()
    
    if success:
        # 4. تست پیش‌بینی
        print("\n4. تست پیش‌بینی برای هندسه‌های جدید...")
        test_cases = [
            ("کره", 220, 2.0, 13, "پلی اورتان"),
            ("سطح", 180, 3.5, 11, "فوم"),
            ("لوله", 300, 1.2, 20, "پشم شیشه"),
            ("مکعب", 160, 2.5, 12, "فوم")
        ]
        
        print("\n" + "="*60)
        print("نتایج پیش‌بینی:")
        print("="*60)
        
        for geometry, eq_temp, area, coeff, ins_type in test_cases:
            prediction = analyzer.predict_insulation_temperature(
                eq_temp, area, coeff, geometry, ins_type
            )
            
            if prediction:
                temp_reduction = eq_temp - prediction
                efficiency = (temp_reduction / eq_temp) * 100
                
                print(f"\nهندسه: {geometry} | عایق: {ins_type}")
                print(f"دمای تجهیز: {eq_temp}°C")
                print(f"دمای پیش‌بینی عایق: {prediction:.1f}°C")
                print(f"کاهش دما: {temp_reduction:.1f}°C ({efficiency:.1f}%)")
                print("-" * 40)
    
    # 5. نمایش آمار داده‌ها
    print("\n5. آمار کلی داده‌ها:")
    data_list = analyzer.database.get_all_data()
    if len(data_list) > 0:
        print(f"تعداد کل رکوردها: {len(data_list)}")
        
        geometries = list(set([d['geometry_type'] for d in data_list]))
        insulations = list(set([d['insulation_type'] for d in data_list]))
        eq_temps = [d['equipment_surface_temp'] for d in data_list]
        ins_temps = [d['insulation_surface_temp'] for d in data_list]
        
        print(f"هندسه‌های موجود: {geometries}")
        print(f"انواع عایق: {insulations}")
        print(f"محدوده دمای تجهیز: {min(eq_temps):.1f} - {max(eq_temps):.1f}°C")
        print(f"محدوده دمای عایق: {min(ins_temps):.1f} - {max(ins_temps):.1f}°C")
        
        # محاسبه میانگین بازده
        total_efficiency = 0
        for data in data_list:
            reduction = data['equipment_surface_temp'] - data['insulation_surface_temp']
            eff = (reduction / data['equipment_surface_temp']) * 100
            total_efficiency += eff
        
        avg_efficiency = total_efficiency / len(data_list)
        print(f"میانگین بازده عایق‌کاری: {avg_efficiency:.1f}%")
    
    print("\n" + "="*60)
    print("=== پایان دمو ===")
    print("برای استفاده کامل از برنامه:")
    print("python3 simple_thermal_analyzer.py")
    print("="*60)

if __name__ == "__main__":
    run_simple_demo()
