#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
تحلیلگر انتقال حرارت و عایق‌های حرارتی - نسخه ساده
Simple Thermal Insulation Analysis System
"""

import sqlite3
import json
import os
from datetime import datetime
import pickle
import re

class ThermalData:
    """کلاس برای ذخیره اطلاعات حرارتی"""
    def __init__(self, geometry_type, equipment_surface_temp, 
                 insulation_surface_temp, insulation_type,
                 cross_section_area, convection_coefficient,
                 thermal_conductivity=None, thickness=None):
        self.geometry_type = geometry_type
        self.equipment_surface_temp = equipment_surface_temp
        self.insulation_surface_temp = insulation_surface_temp
        self.insulation_type = insulation_type
        self.cross_section_area = cross_section_area
        self.convection_coefficient = convection_coefficient
        self.thermal_conductivity = thermal_conductivity
        self.thickness = thickness
        self.timestamp = datetime.now()

class HTMLParser:
    """کلاس برای پردازش فایل‌های HTML خروجی نرم‌افزار"""
    
    def __init__(self):
        self.supported_patterns = {
            'temperature': r'(\d+\.?\d*)\s*°?[CF]',
            'geometry': r'(لوله|کره|مکعب|سطح|pipe|sphere|cube|surface)',
            'insulation': r'(پلی\s*اورتان|فوم|پشم\s*شیشه|polyurethane|foam|glass\s*wool)',
            'area': r'(\d+\.?\d*)\s*(m²|متر\s*مربع)',
            'coefficient': r'(\d+\.?\d*)\s*(W/m²\.K|وات\s*بر\s*متر\s*مربع\s*کلوین)'
        }
    
    def parse_html_file(self, file_path):
        """پردازش یک فایل HTML و استخراج اطلاعات حرارتی"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # حذف تگ‌های HTML
            text = re.sub(r'<[^>]+>', ' ', content)
            
            # استخراج اطلاعات با استفاده از regex
            extracted_data = {}
            for key, pattern in self.supported_patterns.items():
                matches = re.findall(pattern, text, re.IGNORECASE)
                extracted_data[key] = matches
            
            return self._process_extracted_data(extracted_data)
            
        except Exception as e:
            print(f"خطا در پردازش فایل {file_path}: {e}")
            return None
    
    def _process_extracted_data(self, raw_data):
        """پردازش و تمیز کردن داده‌های استخراج شده"""
        processed = {}
        
        # پردازش دماها
        if raw_data.get('temperature'):
            temps = [float(t) for t in raw_data['temperature']]
            processed['equipment_temp'] = max(temps) if temps else None
            processed['insulation_temp'] = min(temps) if len(temps) > 1 else None
        
        # پردازش نوع هندسه
        if raw_data.get('geometry'):
            processed['geometry_type'] = raw_data['geometry'][0]
        
        # پردازش نوع عایق
        if raw_data.get('insulation'):
            processed['insulation_type'] = raw_data['insulation'][0]
        
        # پردازش سطح مقطع
        if raw_data.get('area'):
            processed['cross_section_area'] = float(raw_data['area'][0])
        
        # پردازش ضریب انتقال حرارت
        if raw_data.get('coefficient'):
            processed['convection_coefficient'] = float(raw_data['coefficient'][0])
        
        return processed

class ThermalDatabase:
    """کلاس برای مدیریت پایگاه داده حرارتی"""
    
    def __init__(self, db_path="thermal_data.db"):
        self.db_path = db_path
        self._initialize_database()
    
    def _initialize_database(self):
        """ایجاد پایگاه داده و جداول مورد نیاز"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS thermal_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                geometry_type TEXT NOT NULL,
                equipment_surface_temp REAL NOT NULL,
                insulation_surface_temp REAL NOT NULL,
                insulation_type TEXT NOT NULL,
                cross_section_area REAL NOT NULL,
                convection_coefficient REAL NOT NULL,
                thermal_conductivity REAL,
                thickness REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def insert_data(self, thermal_data):
        """درج داده جدید در پایگاه داده"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO thermal_records 
            (geometry_type, equipment_surface_temp, insulation_surface_temp,
             insulation_type, cross_section_area, convection_coefficient,
             thermal_conductivity, thickness)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            thermal_data.geometry_type,
            thermal_data.equipment_surface_temp,
            thermal_data.insulation_surface_temp,
            thermal_data.insulation_type,
            thermal_data.cross_section_area,
            thermal_data.convection_coefficient,
            thermal_data.thermal_conductivity,
            thermal_data.thickness
        ))
        
        conn.commit()
        conn.close()
    
    def get_all_data(self):
        """دریافت تمام داده‌ها از پایگاه داده"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM thermal_records")
        rows = cursor.fetchall()
        
        columns = [description[0] for description in cursor.description]
        data = []
        for row in rows:
            data.append(dict(zip(columns, row)))
        
        conn.close()
        return data

class SimplePredictor:
    """کلاس ساده برای پیش‌بینی بدون استفاده از sklearn"""
    
    def __init__(self):
        self.is_trained = False
        self.training_data = []
        self.geometry_mapping = {
            'لوله': 1, 'pipe': 1,
            'کره': 2, 'sphere': 2,
            'مکعب': 3, 'cube': 3,
            'سطح': 4, 'surface': 4
        }
        self.insulation_mapping = {
            'پلی اورتان': 1, 'polyurethane': 1,
            'فوم': 2, 'foam': 2,
            'پشم شیشه': 3, 'glass wool': 3
        }
    
    def train_model(self, data_list):
        """آموزش مدل پیش‌بینی ساده"""
        if len(data_list) < 3:
            print("تعداد داده‌ها برای آموزش کافی نیست. حداقل 3 نمونه نیاز است.")
            return False
        
        self.training_data = []
        
        for record in data_list:
            geometry_code = self.geometry_mapping.get(record['geometry_type'], 0)
            insulation_code = self.insulation_mapping.get(record['insulation_type'], 0)
            
            features = [
                record['equipment_surface_temp'],
                record['cross_section_area'],
                record['convection_coefficient'],
                geometry_code,
                insulation_code
            ]
            
            self.training_data.append({
                'features': features,
                'target': record['insulation_surface_temp']
            })
        
        self.is_trained = True
        print(f"مدل با {len(self.training_data)} نمونه آموزش داده شد.")
        return True
    
    def predict(self, equipment_temp, cross_section_area, convection_coefficient, geometry_type, insulation_type):
        """پیش‌بینی دمای سطح عایق با استفاده از میانگین وزنی"""
        
        if not self.is_trained:
            raise ValueError("مدل هنوز آموزش نداده شده است.")
        
        geometry_code = self.geometry_mapping.get(geometry_type, 0)
        insulation_code = self.insulation_mapping.get(insulation_type, 0)
        
        input_features = [equipment_temp, cross_section_area, convection_coefficient, geometry_code, insulation_code]
        
        # محاسبه فاصله با هر نمونه آموزشی
        weights = []
        targets = []
        
        for sample in self.training_data:
            distance = 0
            for i, feature in enumerate(input_features):
                distance += (feature - sample['features'][i]) ** 2
            
            distance = distance ** 0.5
            
            # وزن معکوس فاصله
            weight = 1 / (distance + 0.001)
            
            weights.append(weight)
            targets.append(sample['target'])
        
        # محاسبه میانگین وزنی
        total_weight = sum(weights)
        weighted_prediction = sum(w * t for w, t in zip(weights, targets)) / total_weight
        
        return weighted_prediction

class ThermalAnalyzer:
    """کلاس اصلی برای تحلیل حرارتی"""
    
    def __init__(self, db_path="thermal_data.db"):
        self.database = ThermalDatabase(db_path)
        self.parser = HTMLParser()
        self.predictor = SimplePredictor()
    
    def import_html_files(self, directory_path):
        """وارد کردن فایل‌های HTML از یک پوشه"""
        if not os.path.exists(directory_path):
            print(f"پوشه {directory_path} وجود ندارد.")
            return
        
        imported_count = 0
        
        for filename in os.listdir(directory_path):
            if filename.lower().endswith('.html'):
                file_path = os.path.join(directory_path, filename)
                data = self.parser.parse_html_file(file_path)
                
                if data and self._validate_data(data):
                    thermal_data = ThermalData(
                        geometry_type=data.get('geometry_type', 'نامشخص'),
                        equipment_surface_temp=data.get('equipment_temp', 0),
                        insulation_surface_temp=data.get('insulation_temp', 0),
                        insulation_type=data.get('insulation_type', 'نامشخص'),
                        cross_section_area=data.get('cross_section_area', 0),
                        convection_coefficient=data.get('convection_coefficient', 0)
                    )
                    
                    self.database.insert_data(thermal_data)
                    imported_count += 1
                    print(f"فایل {filename} با موفقیت وارد شد.")
                else:
                    print(f"فایل {filename} داده‌های کافی ندارد.")
        
        print(f"تعداد {imported_count} فایل با موفقیت وارد شد.")
    
    def _validate_data(self, data):
        """اعتبارسنجی داده‌های استخراج شده"""
        required_fields = ['equipment_temp', 'insulation_temp', 'cross_section_area']
        return all(field in data and data[field] is not None for field in required_fields)
    
    def train_prediction_model(self):
        """آموزش مدل پیش‌بینی با داده‌های موجود"""
        data_list = self.database.get_all_data()
        if len(data_list) == 0:
            print("هیچ داده‌ای برای آموزش مدل وجود ندارد.")
            return False
        
        return self.predictor.train_model(data_list)
    
    def predict_insulation_temperature(self, equipment_temp, cross_section_area, convection_coefficient, geometry_type, insulation_type):
        """پیش‌بینی دمای سطح عایق برای هندسه جدید"""
        try:
            prediction = self.predictor.predict(
                equipment_temp, cross_section_area, convection_coefficient,
                geometry_type, insulation_type
            )
            return prediction
        except Exception as e:
            print(f"خطا در پیش‌بینی: {e}")
            return None
    
    def add_manual_data(self, geometry_type, equipment_temp, insulation_temp, insulation_type, cross_section_area, convection_coefficient):
        """افزودن دستی داده‌های حرارتی"""
        thermal_data = ThermalData(
            geometry_type=geometry_type,
            equipment_surface_temp=equipment_temp,
            insulation_surface_temp=insulation_temp,
            insulation_type=insulation_type,
            cross_section_area=cross_section_area,
            convection_coefficient=convection_coefficient
        )
        
        self.database.insert_data(thermal_data)
        print("داده جدید با موفقیت اضافه شد.")

def main():
    """تابع اصلی برنامه"""
    print("=== سیستم تحلیل انتقال حرارت و عایق‌های حرارتی ===")
    print("نسخه ساده بدون وابستگی‌های خارجی")
    
    analyzer = ThermalAnalyzer()
    
    while True:
        print("\n" + "="*50)
        print("گزینه‌های موجود:")
        print("1. وارد کردن فایل‌های HTML")
        print("2. افزودن داده دستی")
        print("3. آموزش مدل پیش‌بینی")
        print("4. پیش‌بینی دمای عایق برای هندسه جدید")
        print("5. مشاهده آمار داده‌ها")
        print("6. خروج")
        print("="*50)
        
        choice = input("\nانتخاب شما (1-6): ").strip()
        
        if choice == '1':
            directory = input("مسیر پوشه حاوی فایل‌های HTML (پیش‌فرض: ./html_files): ").strip()
            if not directory:
                directory = "./html_files"
            analyzer.import_html_files(directory)
        
        elif choice == '2':
            try:
                print("\nافزودن داده جدید:")
                geometry = input("نوع هندسه (لوله/کره/مکعب/سطح): ").strip()
                eq_temp = float(input("دمای سطح تجهیز (°C): "))
                ins_temp = float(input("دمای سطح عایق (°C): "))
                ins_type = input("نوع عایق (پلی اورتان/فوم/پشم شیشه): ").strip()
                area = float(input("سطح مقطع (m²): "))
                coeff = float(input("ضریب انتقال حرارت (W/m².K): "))
                
                analyzer.add_manual_data(geometry, eq_temp, ins_temp, ins_type, area, coeff)
            except ValueError:
                print("خطا: لطفاً مقادیر عددی معتبر وارد کنید.")
        
        elif choice == '3':
            print("\nشروع آموزش مدل...")
            analyzer.train_prediction_model()
        
        elif choice == '4':
            try:
                print("\nپیش‌بینی دمای عایق:")
                geometry = input("نوع هندسه (لوله/کره/مکعب/سطح): ").strip()
                eq_temp = float(input("دمای سطح تجهیز (°C): "))
                area = float(input("سطح مقطع (m²): "))
                coeff = float(input("ضریب انتقال حرارت (W/m².K): "))
                ins_type = input("نوع عایق (پلی اورتان/فوم/پشم شیشه): ").strip()
                
                prediction = analyzer.predict_insulation_temperature(
                    eq_temp, area, coeff, geometry, ins_type
                )
                
                if prediction is not None:
                    print(f"\n*** دمای پیش‌بینی شده سطح عایق: {prediction:.1f} °C ***")
                    
                    # محاسبه کاهش دما
                    temp_reduction = eq_temp - prediction
                    print(f"کاهش دما پس از عایق‌کاری: {temp_reduction:.1f} °C")
                    print(f"درصد کاهش دما: {(temp_reduction/eq_temp)*100:.1f}%")
                
            except ValueError:
                print("خطا: لطفاً مقادیر عددی معتبر وارد کنید.")
        
        elif choice == '5':
            data_list = analyzer.database.get_all_data()
            if len(data_list) > 0:
                print(f"\nآمار داده‌ها:")
                print(f"تعداد کل رکوردها: {len(data_list)}")
                
                geometries = list(set([d['geometry_type'] for d in data_list]))
                insulations = list(set([d['insulation_type'] for d in data_list]))
                eq_temps = [d['equipment_surface_temp'] for d in data_list]
                ins_temps = [d['insulation_surface_temp'] for d in data_list]
                
                print(f"هندسه‌های موجود: {geometries}")
                print(f"انواع عایق موجود: {insulations}")
                print(f"محدوده دمای تجهیز: {min(eq_temps):.1f} تا {max(eq_temps):.1f} °C")
                print(f"محدوده دمای عایق: {min(ins_temps):.1f} تا {max(ins_temps):.1f} °C")
            else:
                print("هیچ داده‌ای موجود نیست.")
        
        elif choice == '6':
            print("برنامه بسته شد. موفق باشید!")
            break
        
        else:
            print("گزینه نامعتبر. لطفاً عددی بین 1 تا 6 وارد کنید.")

if __name__ == "__main__":
    main()
