#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
برنامه اصلی تحلیل انتقال حرارت و پیش‌بینی دمای سطح
"""

import sys
import os
from heat_transfer_analyzer import HeatTransferAnalyzer

def print_banner():
    """نمایش بنر برنامه"""
    print("=" * 60)
    print("   سیستم تحلیل انتقال حرارت و پیش‌بینی دمای سطح")
    print("   Heat Transfer Analysis & Surface Temperature Prediction")
    print("=" * 60)
    print()

def print_menu():
    """نمایش منوی اصلی"""
    print("\n--- منوی اصلی ---")
    print("1. ایجاد فایل‌های نمونه")
    print("2. بارگذاری داده‌های Colors (HTML)")
    print("3. بارگذاری داده‌های عایق (Excel)")
    print("4. محاسبه نظری دمای سطح")
    print("5. آموزش مدل یادگیری ماشین")
    print("6. پیش‌بینی دمای سطح")
    print("7. تحلیل کامل تجهیز")
    print("8. تولید گزارش")
    print("9. ذخیره/بارگذاری مدل")
    print("10. نمایش وضعیت سیستم")
    print("0. خروج")
    print("-" * 30)

def handle_create_samples(analyzer):
    """ایجاد فایل‌های نمونه"""
    print("\n=== ایجاد فایل‌های نمونه ===")
    analyzer.create_sample_data()
    print("\nاکنون می‌توانید از فایل‌های زیر استفاده کنید:")
    print("- inputdata.xlsx: داده‌های عایق")
    print("- sample_colors_output.html: خروجی نمونه Colors")

def handle_load_colors_data(analyzer):
    """بارگذاری داده‌های Colors"""
    print("\n=== بارگذاری داده‌های Colors ===")
    print("1. بارگذاری از یک فایل")
    print("2. بارگذاری از پوشه")
    
    choice = input("انتخاب کنید (1 یا 2): ").strip()
    
    if choice == "1":
        file_path = input("مسیر فایل HTML را وارد کنید: ").strip()
        if os.path.exists(file_path):
            analyzer.load_colors_data(file_path=file_path)
        else:
            print(f"فایل {file_path} پیدا نشد!")
    
    elif choice == "2":
        dir_path = input("مسیر پوشه را وارد کنید: ").strip()
        if os.path.exists(dir_path):
            analyzer.load_colors_data(directory_path=dir_path)
        else:
            print(f"پوشه {dir_path} پیدا نشد!")
    
    else:
        print("انتخاب نامعتبر!")

def handle_load_insulation_data(analyzer):
    """بارگذاری داده‌های عایق"""
    print("\n=== بارگذاری داده‌های عایق ===")
    file_path = input("مسیر فایل Excel را وارد کنید (پیش‌فرض: inputdata.xlsx): ").strip()
    
    if not file_path:
        file_path = "inputdata.xlsx"
    
    if os.path.exists(file_path):
        analyzer.load_insulation_data(file_path)
    else:
        print(f"فایل {file_path} پیدا نشد!")
        print("از گزینه 1 برای ایجاد فایل نمونه استفاده کنید")

def handle_theoretical_calculation(analyzer):
    """محاسبه نظری دمای سطح"""
    print("\n=== محاسبه نظری دمای سطح ===")
    
    status = analyzer.get_status()
    if not status['ready_for_analysis']:
        print("ابتدا داده‌های Colors و عایق را بارگذاری کنید!")
        return
    
    try:
        surface_temp, info = analyzer.calculate_theoretical_surface_temperature()
        
        print(f"\n✓ دمای سطح محاسبه شده: {surface_temp:.2f} °C")
        print(f"✓ کاهش دما: {info['temperature_reduction']:.2f} °C")
        print(f"✓ کارایی عایق‌کاری: {info['efficiency_percentage']:.1f}%")
        print(f"✓ جریان حرارت: {info['heat_flow']:.2f} W")
        print(f"✓ تلفات حرارتی: {info['heat_loss']:.2f} W")
        
    except Exception as e:
        print(f"خطا در محاسبه: {str(e)}")

def handle_train_ml_model(analyzer):
    """آموزش مدل یادگیری ماشین"""
    print("\n=== آموزش مدل یادگیری ماشین ===")
    
    status = analyzer.get_status()
    if not status['colors_data_loaded']:
        print("ابتدا داده‌های Colors را بارگذاری کنید!")
        return
    
    if status['colors_data_count'] < 3:
        print("برای آموزش مدل حداقل 3 نمونه داده نیاز است!")
        print(f"تعداد داده‌های فعلی: {status['colors_data_count']}")
        return
    
    try:
        results = analyzer.train_ml_model()
        
        if results:
            print("\n✓ آموزش مدل با موفقیت انجام شد!")
            best_model = max(results.keys(), key=lambda x: results[x]['r2'])
            print(f"✓ بهترین مدل: {best_model}")
        else:
            print("خطا در آموزش مدل!")
            
    except Exception as e:
        print(f"خطا در آموزش مدل: {str(e)}")

def handle_predict_temperature(analyzer):
    """پیش‌بینی دمای سطح"""
    print("\n=== پیش‌بینی دمای سطح ===")
    
    status = analyzer.get_status()
    if not status['ready_for_analysis']:
        print("ابتدا داده‌های Colors و عایق را بارگذاری کنید!")
        return
    
    print("1. محاسبه نظری")
    print("2. یادگیری ماشین")
    print("3. ترکیبی (خودکار)")
    
    method_choice = input("روش را انتخاب کنید (1-3): ").strip()
    
    method_map = {"1": "theoretical", "2": "ml", "3": "auto"}
    method = method_map.get(method_choice, "auto")
    
    try:
        surface_temp, info = analyzer.predict_surface_temperature(method=method)
        
        print(f"\n✓ دمای سطح پیش‌بینی شده: {surface_temp:.2f} °C")
        print(f"✓ روش استفاده شده: {info.get('method', method)}")
        
        if 'temperature_reduction' in info:
            print(f"✓ کاهش دما: {info['temperature_reduction']:.2f} °C")
        
        if 'efficiency_percentage' in info:
            print(f"✓ کارایی: {info['efficiency_percentage']:.1f}%")
            
    except Exception as e:
        print(f"خطا در پیش‌بینی: {str(e)}")

def handle_equipment_analysis(analyzer):
    """تحلیل کامل تجهیز"""
    print("\n=== تحلیل کامل تجهیز ===")
    
    status = analyzer.get_status()
    if not status['ready_for_analysis']:
        print("ابتدا داده‌های Colors و عایق را بارگذاری کنید!")
        return
    
    if not analyzer.colors_data:
        print("هیچ تجهیزی برای تحلیل موجود نیست!")
        return
    
    # انتخاب تجهیز
    if len(analyzer.colors_data) > 1:
        print("تجهیزات موجود:")
        for i, data in enumerate(analyzer.colors_data):
            name = data.get('file_name', f'تجهیز {i+1}')
            print(f"{i+1}. {name}")
        
        choice = input(f"تجهیز را انتخاب کنید (1-{len(analyzer.colors_data)}): ").strip()
        try:
            equipment_index = int(choice) - 1
            if 0 <= equipment_index < len(analyzer.colors_data):
                equipment_data = analyzer.colors_data[equipment_index]
            else:
                print("انتخاب نامعتبر!")
                return
        except ValueError:
            print("ورودی نامعتبر!")
            return
    else:
        equipment_data = analyzer.colors_data[0]
    
    try:
        results = analyzer.analyze_equipment(equipment_data)
        
        if 'error' in results:
            print(f"خطا در تحلیل: {results['error']}")
            return
        
        print(f"\n=== نتایج تحلیل ===")
        print(f"دمای سطح: {results['surface_temperature']:.2f} °C")
        print(f"کاهش دما: {results['performance']['temperature_reduction']:.2f} °C")
        print(f"کارایی: {results['performance']['efficiency_percentage']:.1f}%")
        print(f"ایمن: {'بله' if results['performance']['is_safe'] else 'خیر'}")
        
        if results['recommendations']:
            print("\nتوصیه‌ها:")
            for i, rec in enumerate(results['recommendations'], 1):
                print(f"{i}. {rec}")
        
    except Exception as e:
        print(f"خطا در تحلیل: {str(e)}")

def handle_generate_report(analyzer):
    """تولید گزارش"""
    print("\n=== تولید گزارش ===")
    
    status = analyzer.get_status()
    if not status['ready_for_analysis']:
        print("ابتدا داده‌های Colors و عایق را بارگذاری کنید!")
        return
    
    output_file = input("نام فایل گزارش (پیش‌فرض: analysis_report.json): ").strip()
    if not output_file:
        output_file = "analysis_report.json"
    
    try:
        report_file = analyzer.generate_report(output_file)
        print(f"✓ گزارش در فایل {report_file} ذخیره شد")
        
    except Exception as e:
        print(f"خطا در تولید گزارش: {str(e)}")

def handle_model_operations(analyzer):
    """عملیات مدل (ذخیره/بارگذاری)"""
    print("\n=== عملیات مدل ===")
    print("1. ذخیره مدل")
    print("2. بارگذاری مدل")
    
    choice = input("انتخاب کنید (1 یا 2): ").strip()
    
    if choice == "1":
        if not analyzer.ml_predictor.is_trained:
            print("هیچ مدل آموزش دیده‌ای برای ذخیره وجود ندارد!")
            return
        
        file_path = input("نام فایل برای ذخیره (پیش‌فرض: heat_transfer_model.pkl): ").strip()
        if not file_path:
            file_path = "heat_transfer_model.pkl"
        
        analyzer.save_model(file_path)
    
    elif choice == "2":
        file_path = input("مسیر فایل مدل (پیش‌فرض: heat_transfer_model.pkl): ").strip()
        if not file_path:
            file_path = "heat_transfer_model.pkl"
        
        if os.path.exists(file_path):
            analyzer.load_model(file_path)
        else:
            print(f"فایل {file_path} پیدا نشد!")
    
    else:
        print("انتخاب نامعتبر!")

def handle_show_status(analyzer):
    """نمایش وضعیت سیستم"""
    print("\n=== وضعیت سیستم ===")
    status = analyzer.get_status()
    
    print(f"داده‌های Colors بارگذاری شده: {'✓' if status['colors_data_loaded'] else '✗'}")
    if status['colors_data_loaded']:
        print(f"  تعداد فایل‌ها: {status['colors_data_count']}")
    
    print(f"داده‌های عایق بارگذاری شده: {'✓' if status['insulation_data_loaded'] else '✗'}")
    if status['insulation_data_loaded']:
        print(f"  تعداد لایه‌ها: {status['insulation_layers_count']}")
    
    print(f"مدل ML آموزش دیده: {'✓' if status['ml_model_trained'] else '✗'}")
    print(f"آماده برای تحلیل: {'✓' if status['ready_for_analysis'] else '✗'}")

def main():
    """تابع اصلی برنامه"""
    print_banner()
    
    # ایجاد نمونه analyzer
    analyzer = HeatTransferAnalyzer()
    
    # منوی اصلی
    while True:
        print_menu()
        choice = input("انتخاب خود را وارد کنید: ").strip()
        
        if choice == "0":
            print("خروج از برنامه...")
            break
        
        elif choice == "1":
            handle_create_samples(analyzer)
        
        elif choice == "2":
            handle_load_colors_data(analyzer)
        
        elif choice == "3":
            handle_load_insulation_data(analyzer)
        
        elif choice == "4":
            handle_theoretical_calculation(analyzer)
        
        elif choice == "5":
            handle_train_ml_model(analyzer)
        
        elif choice == "6":
            handle_predict_temperature(analyzer)
        
        elif choice == "7":
            handle_equipment_analysis(analyzer)
        
        elif choice == "8":
            handle_generate_report(analyzer)
        
        elif choice == "9":
            handle_model_operations(analyzer)
        
        elif choice == "10":
            handle_show_status(analyzer)
        
        else:
            print("انتخاب نامعتبر! لطفاً عددی بین 0 تا 10 وارد کنید.")
        
        # منتظر ماندن برای ورودی کاربر
        input("\nبرای ادامه Enter را فشار دهید...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nبرنامه توسط کاربر متوقف شد.")
    except Exception as e:
        print(f"\nخطای غیرمنتظره: {str(e)}")
        sys.exit(1)