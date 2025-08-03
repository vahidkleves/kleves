#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
تست جامع سیستم تحلیل انتقال حرارت
"""

import os
import sys
import traceback
from heat_transfer_analyzer import HeatTransferAnalyzer

def test_file_creation():
    """تست ایجاد فایل‌های نمونه"""
    print("=== تست ایجاد فایل‌های نمونه ===")
    
    try:
        analyzer = HeatTransferAnalyzer()
        analyzer.create_sample_data()
        
        # بررسی وجود فایل‌ها
        files_to_check = ['inputdata.xlsx', 'sample_colors_output.html']
        for file_name in files_to_check:
            if os.path.exists(file_name):
                print(f"✓ فایل {file_name} با موفقیت ایجاد شد")
            else:
                print(f"✗ فایل {file_name} ایجاد نشد")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ خطا در ایجاد فایل‌ها: {str(e)}")
        return False

def test_data_loading():
    """تست بارگذاری داده‌ها"""
    print("\n=== تست بارگذاری داده‌ها ===")
    
    try:
        analyzer = HeatTransferAnalyzer()
        
        # بارگذاری داده‌های Colors
        if os.path.exists('sample_colors_output.html'):
            colors_data = analyzer.load_colors_data(file_path='sample_colors_output.html')
            if colors_data:
                print("✓ داده‌های Colors با موفقیت بارگذاری شد")
            else:
                print("✗ خطا در بارگذاری داده‌های Colors")
                return False
        
        # بارگذاری داده‌های عایق
        if os.path.exists('inputdata.xlsx'):
            insulation_data = analyzer.load_insulation_data('inputdata.xlsx')
            if insulation_data:
                print("✓ داده‌های عایق با موفقیت بارگذاری شد")
                print(f"  تعداد لایه‌ها: {len(insulation_data)}")
            else:
                print("✗ خطا در بارگذاری داده‌های عایق")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ خطا در بارگذاری داده‌ها: {str(e)}")
        return False

def test_theoretical_calculation():
    """تست محاسبات نظری"""
    print("\n=== تست محاسبات نظری ===")
    
    try:
        analyzer = HeatTransferAnalyzer()
        analyzer.load_colors_data(file_path='sample_colors_output.html')
        analyzer.load_insulation_data('inputdata.xlsx')
        
        surface_temp, info = analyzer.calculate_theoretical_surface_temperature()
        
        # بررسی معقول بودن نتایج
        if 20 <= surface_temp <= 200:  # دمای منطقی
            print(f"✓ محاسبه نظری موفق: {surface_temp:.2f} °C")
            print(f"  کاهش دما: {info['temperature_reduction']:.2f} °C")
            print(f"  کارایی: {info['efficiency_percentage']:.1f}%")
            return True
        else:
            print(f"✗ نتیجه غیرمنطقی: {surface_temp:.2f} °C")
            return False
            
    except Exception as e:
        print(f"✗ خطا در محاسبات نظری: {str(e)}")
        return False

def test_ml_training():
    """تست آموزش مدل یادگیری ماشین"""
    print("\n=== تست آموزش مدل ML ===")
    
    try:
        analyzer = HeatTransferAnalyzer()
        
        # ایجاد داده‌های آموزشی متعدد
        test_data = []
        for i in range(10):
            data = {
                'inner_temperature': 300 + i * 50,
                'ambient_temperature': 20 + i * 2,
                'wind_speed': 1 + i * 0.5,
                'surface_area': 5 + i,
                'equipment_type': ['horizontal_pipe', 'vertical_pipe', 'sphere'][i % 3],
                'surface_temperature': 50 + i * 10  # مقدار هدف
            }
            test_data.append(data)
        
        analyzer.colors_data = test_data
        analyzer.load_insulation_data('inputdata.xlsx')
        
        # آموزش مدل
        results = analyzer.train_ml_model()
        
        if results and len(results) > 0:
            print("✓ آموزش مدل ML موفق")
            best_model = max(results.keys(), key=lambda x: results[x]['r2'])
            best_r2 = results[best_model]['r2']
            print(f"  بهترین مدل: {best_model} (R² = {best_r2:.4f})")
            
            if best_r2 > 0.5:  # R² قابل قبول
                return True
            else:
                print(f"✗ کیفیت مدل پایین: R² = {best_r2:.4f}")
                return False
        else:
            print("✗ آموزش مدل ناموفق")
            return False
            
    except Exception as e:
        print(f"✗ خطا در آموزش مدل: {str(e)}")
        return False

def test_prediction():
    """تست پیش‌بینی"""
    print("\n=== تست پیش‌بینی ===")
    
    try:
        analyzer = HeatTransferAnalyzer()
        analyzer.load_colors_data(file_path='sample_colors_output.html')
        analyzer.load_insulation_data('inputdata.xlsx')
        
        # تست پیش‌بینی نظری
        surface_temp_theoretical, _ = analyzer.predict_surface_temperature(method='theoretical')
        
        if 20 <= surface_temp_theoretical <= 200:
            print(f"✓ پیش‌بینی نظری موفق: {surface_temp_theoretical:.2f} °C")
        else:
            print(f"✗ پیش‌بینی نظری غیرمنطقی: {surface_temp_theoretical:.2f} °C")
            return False
        
        # تست پیش‌بینی خودکار
        surface_temp_auto, info_auto = analyzer.predict_surface_temperature(method='auto')
        
        if 20 <= surface_temp_auto <= 200:
            print(f"✓ پیش‌بینی خودکار موفق: {surface_temp_auto:.2f} °C")
            print(f"  روش استفاده شده: {info_auto.get('method', 'نامشخص')}")
            return True
        else:
            print(f"✗ پیش‌بینی خودکار غیرمنطقی: {surface_temp_auto:.2f} °C")
            return False
            
    except Exception as e:
        print(f"✗ خطا در پیش‌بینی: {str(e)}")
        return False

def test_equipment_analysis():
    """تست تحلیل کامل تجهیز"""
    print("\n=== تست تحلیل کامل تجهیز ===")
    
    try:
        analyzer = HeatTransferAnalyzer()
        analyzer.load_colors_data(file_path='sample_colors_output.html')
        analyzer.load_insulation_data('inputdata.xlsx')
        
        equipment_data = analyzer.colors_data[0]
        results = analyzer.analyze_equipment(equipment_data)
        
        if 'error' in results:
            print(f"✗ خطا در تحلیل تجهیز: {results['error']}")
            return False
        
        # بررسی وجود اجزای ضروری نتیجه
        required_keys = ['surface_temperature', 'performance', 'equipment_data']
        for key in required_keys:
            if key not in results:
                print(f"✗ کلید ضروری {key} در نتایج وجود ندارد")
                return False
        
        surface_temp = results['surface_temperature']
        efficiency = results['performance']['efficiency_percentage']
        
        if 20 <= surface_temp <= 200 and 0 <= efficiency <= 100:
            print(f"✓ تحلیل کامل تجهیز موفق")
            print(f"  دمای سطح: {surface_temp:.2f} °C")
            print(f"  کارایی: {efficiency:.1f}%")
            print(f"  ایمن: {'بله' if results['performance']['is_safe'] else 'خیر'}")
            return True
        else:
            print(f"✗ نتایج غیرمنطقی در تحلیل تجهیز")
            return False
            
    except Exception as e:
        print(f"✗ خطا در تحلیل تجهیز: {str(e)}")
        return False

def test_report_generation():
    """تست تولید گزارش"""
    print("\n=== تست تولید گزارش ===")
    
    try:
        analyzer = HeatTransferAnalyzer()
        analyzer.load_colors_data(file_path='sample_colors_output.html')
        analyzer.load_insulation_data('inputdata.xlsx')
        
        report_file = analyzer.generate_report('test_report.json')
        
        if os.path.exists(report_file):
            file_size = os.path.getsize(report_file)
            if file_size > 100:  # حداقل 100 بایت
                print(f"✓ گزارش با موفقیت تولید شد: {report_file}")
                print(f"  اندازه فایل: {file_size} بایت")
                return True
            else:
                print(f"✗ فایل گزارش خیلی کوچک است: {file_size} بایت")
                return False
        else:
            print("✗ فایل گزارش ایجاد نشد")
            return False
            
    except Exception as e:
        print(f"✗ خطا در تولید گزارش: {str(e)}")
        return False

def test_status_check():
    """تست بررسی وضعیت سیستم"""
    print("\n=== تست بررسی وضعیت ===")
    
    try:
        analyzer = HeatTransferAnalyzer()
        
        # وضعیت اولیه
        status = analyzer.get_status()
        if not status['colors_data_loaded'] and not status['insulation_data_loaded']:
            print("✓ وضعیت اولیه صحیح")
        else:
            print("✗ وضعیت اولیه غلط")
            return False
        
        # بعد از بارگذاری داده‌ها
        analyzer.load_colors_data(file_path='sample_colors_output.html')
        analyzer.load_insulation_data('inputdata.xlsx')
        
        status = analyzer.get_status()
        if status['colors_data_loaded'] and status['insulation_data_loaded'] and status['ready_for_analysis']:
            print("✓ وضعیت بعد از بارگذاری صحیح")
            return True
        else:
            print("✗ وضعیت بعد از بارگذاری غلط")
            return False
            
    except Exception as e:
        print(f"✗ خطا در بررسی وضعیت: {str(e)}")
        return False

def test_custom_data():
    """تست با داده‌های سفارشی"""
    print("\n=== تست داده‌های سفارشی ===")
    
    try:
        analyzer = HeatTransferAnalyzer()
        
        # داده‌های سفارشی
        custom_equipment = {
            'inner_temperature': 400,
            'ambient_temperature': 30,
            'wind_speed': 2.5,
            'surface_area': 8.0,
            'equipment_type': 'valve'
        }
        
        custom_insulation = [
            {
                'insulation_type': 'Cerablanket',
                'thickness': 0.025,
                'density': 96,
                'thermal_conductivity': 0.04
            },
            {
                'insulation_type': 'Rock Wool',
                'thickness': 0.040,
                'density': 100,
                'thermal_conductivity': 0.042
            }
        ]
        
        surface_temp, info = analyzer.calculate_theoretical_surface_temperature(
            custom_equipment, custom_insulation
        )
        
        if 20 <= surface_temp <= 200:
            print(f"✓ محاسبه با داده‌های سفارشی موفق: {surface_temp:.2f} °C")
            print(f"  کاهش دما: {info['temperature_reduction']:.2f} °C")
            return True
        else:
            print(f"✗ نتیجه غیرمنطقی با داده‌های سفارشی: {surface_temp:.2f} °C")
            return False
            
    except Exception as e:
        print(f"✗ خطا در تست داده‌های سفارشی: {str(e)}")
        return False

def cleanup_test_files():
    """پاک کردن فایل‌های تست"""
    test_files = [
        'inputdata.xlsx',
        'sample_colors_output.html',
        'test_report.json',
        'example_report.json',
        'example_model.pkl',
        'heat_transfer_model.pkl'
    ]
    
    for file_name in test_files:
        try:
            if os.path.exists(file_name):
                os.remove(file_name)
        except:
            pass

def run_comprehensive_test():
    """اجرای تست جامع"""
    print("شروع تست جامع سیستم تحلیل انتقال حرارت")
    print("=" * 60)
    
    # لیست تست‌ها
    tests = [
        ("ایجاد فایل‌های نمونه", test_file_creation),
        ("بارگذاری داده‌ها", test_data_loading),
        ("محاسبات نظری", test_theoretical_calculation),
        ("آموزش مدل ML", test_ml_training),
        ("پیش‌بینی", test_prediction),
        ("تحلیل کامل تجهیز", test_equipment_analysis),
        ("تولید گزارش", test_report_generation),
        ("بررسی وضعیت", test_status_check),
        ("داده‌های سفارشی", test_custom_data)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            print(f"\n[{passed_tests + 1}/{total_tests}] {test_name}")
            if test_func():
                passed_tests += 1
            else:
                print(f"تست {test_name} ناموفق!")
        except Exception as e:
            print(f"خطای غیرمنتظره در تست {test_name}: {str(e)}")
            traceback.print_exc()
    
    # نتایج نهایی
    print("\n" + "=" * 60)
    print("نتایج تست جامع:")
    print(f"تست‌های موفق: {passed_tests}/{total_tests}")
    print(f"درصد موفقیت: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        print("🎉 همه تست‌ها با موفقیت انجام شد!")
        status = "موفق"
    elif passed_tests >= total_tests * 0.8:
        print("⚠️ اکثر تست‌ها موفق بودند")
        status = "نسبتاً موفق"
    else:
        print("❌ تعداد زیادی از تست‌ها ناموفق بودند")
        status = "ناموفق"
    
    # پاک کردن فایل‌های تست
    print("\nپاک کردن فایل‌های تست...")
    cleanup_test_files()
    
    return status, passed_tests, total_tests

def main():
    """تابع اصلی"""
    try:
        status, passed, total = run_comprehensive_test()
        
        if status == "موفق":
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\nتست توسط کاربر متوقف شد.")
        cleanup_test_files()
        sys.exit(1)
    except Exception as e:
        print(f"\nخطای غیرمنتظره در تست: {str(e)}")
        cleanup_test_files()
        sys.exit(1)

if __name__ == "__main__":
    main()