#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
دموی سیستم تحلیل انتقال حرارت
Demo for Thermal Insulation Analysis System
"""

from thermal_analyzer import ThermalAnalyzer

def run_demo():
    """اجرای دموی برنامه"""
    print("=== دموی سیستم تحلیل انتقال حرارت ===")
    
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
        ("سطح", 200, 38, "پشم شیشه", 2.8, 14)
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
            ("لوله", 300, 1.2, 20, "پشم شیشه")
        ]
        
        for geometry, eq_temp, area, coeff, ins_type in test_cases:
            prediction = analyzer.predict_insulation_temperature(
                eq_temp, area, coeff, geometry, ins_type
            )
            
            if prediction:
                temp_reduction = eq_temp - prediction
                efficiency = (temp_reduction / eq_temp) * 100
                
                print(f"\nهندسه: {geometry}")
                print(f"دمای تجهیز: {eq_temp}°C")
                print(f"دمای پیش‌بینی عایق: {prediction:.1f}°C")
                print(f"کاهش دما: {temp_reduction:.1f}°C")
                print(f"بازده عایق‌کاری: {efficiency:.1f}%")
    
    # 5. نمایش آمار داده‌ها
    print("\n5. آمار کلی داده‌ها:")
    df = analyzer.database.get_all_data()
    if len(df) > 0:
        print(f"تعداد کل رکوردها: {len(df)}")
        print(f"هندسه‌های موجود: {list(df['geometry_type'].unique())}")
        print(f"انواع عایق: {list(df['insulation_type'].unique())}")
        print(f"محدوده دمای تجهیز: {df['equipment_surface_temp'].min():.1f} - {df['equipment_surface_temp'].max():.1f}°C")
        print(f"محدوده دمای عایق: {df['insulation_surface_temp'].min():.1f} - {df['insulation_surface_temp'].max():.1f}°C")
    
    print("\n=== پایان دمو ===")
    print("برای استفاده کامل از برنامه، دستور python thermal_analyzer.py را اجرا کنید.")

if __name__ == "__main__":
    run_demo()
