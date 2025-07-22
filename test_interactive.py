#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
تست تعاملی سیستم تحلیل حرارتی
Interactive Test for Thermal Analysis System
"""

from simple_thermal_analyzer import ThermalAnalyzer

def interactive_test():
    """تست تعاملی برنامه"""
    print("=== تست تعاملی سیستم تحلیل حرارتی ===")
    
    analyzer = ThermalAnalyzer()
    
    # آموزش مدل با داده‌های موجود
    print("آموزش مدل با داده‌های موجود...")
    data_list = analyzer.database.get_all_data()
    
    if len(data_list) >= 3:
        analyzer.predictor.train_model(data_list)
        print("مدل آماده پیش‌بینی است!")
        
        # تست پیش‌بینی
        print("\nتست پیش‌بینی:")
        test_result = analyzer.predict_insulation_temperature(
            200, 2.5, 15, "لوله", "پلی اورتان"
        )
        
        if test_result:
            print(f"برای لوله با دمای 200°C: دمای عایق پیش‌بینی شده = {test_result:.1f}°C")
            print(f"کاهش دما: {200 - test_result:.1f}°C")
        
    else:
        print("داده‌های کافی برای آموزش وجود ندارد.")
        print("لطفاً ابتدا simple_demo.py را اجرا کنید.")

if __name__ == "__main__":
    interactive_test()
