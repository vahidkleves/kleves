"""
خواننده Excel برای خواندن مشخصات عایق‌ها از فایل inputdata.xlsx
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional

class InsulationDataReader:
    """
    کلاس خواندن داده‌های عایق از فایل Excel
    """
    
    def __init__(self):
        # نام‌های ستون‌های مورد انتظار
        self.column_mappings = {
            'insulation_type': ['نوع عایق', 'Insulation Type', 'Type', 'Material'],
            'thickness': ['ضخامت', 'Thickness', 'ضخامت (mm)', 'Thickness (mm)'],
            'density': ['چگالی', 'Density', 'چگالی (kg/m3)', 'Density (kg/m3)'],
            'thermal_conductivity': ['ضریب انتقال حرارت', 'Thermal Conductivity', 'k', 'K (W/m.K)'],
            'layer_number': ['شماره لایه', 'Layer Number', 'Layer', 'No.']
        }
        
        # انواع عایق‌های قابل قبول
        self.valid_insulation_types = [
            'Cerablanket',
            'Silika Needeled Mat', 
            'Rock Wool',
            'Needeled Mat'
        ]
    
    def read_excel_file(self, file_path: str, sheet_name: str = None) -> List[Dict]:
        """
        خواندن فایل Excel و استخراج داده‌های عایق
        """
        try:
            # خواندن فایل Excel
            if sheet_name:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
            else:
                # خواندن اولین شیت
                df = pd.read_excel(file_path)
            
            # پاک کردن ردیف‌های خالی
            df = df.dropna(how='all')
            
            # شناسایی ستون‌ها
            column_map = self._identify_columns(df)
            
            # استخراج داده‌ها
            insulation_data = self._extract_insulation_data(df, column_map)
            
            # اعتبارسنجی داده‌ها
            validated_data = self._validate_insulation_data(insulation_data)
            
            return validated_data
            
        except FileNotFoundError:
            raise FileNotFoundError(f"فایل {file_path} پیدا نشد")
        except Exception as e:
            raise Exception(f"خطا در خواندن فایل Excel: {str(e)}")
    
    def _identify_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        شناسایی ستون‌های مربوط به هر پارامتر
        """
        column_map = {}
        
        for param, possible_names in self.column_mappings.items():
            for col_name in df.columns:
                col_name_clean = str(col_name).strip()
                if any(name.lower() in col_name_clean.lower() for name in possible_names):
                    column_map[param] = col_name
                    break
        
        return column_map
    
    def _extract_insulation_data(self, df: pd.DataFrame, column_map: Dict[str, str]) -> List[Dict]:
        """
        استخراج داده‌های عایق از DataFrame
        """
        insulation_data = []
        
        for index, row in df.iterrows():
            layer_data = {}
            
            # استخراج هر پارامتر
            for param, col_name in column_map.items():
                if col_name in df.columns:
                    value = row[col_name]
                    
                    # پردازش مقدار بر اساس نوع پارامتر
                    if param == 'insulation_type':
                        layer_data[param] = self._clean_insulation_type(value)
                    elif param in ['thickness', 'density', 'thermal_conductivity']:
                        layer_data[param] = self._convert_to_float(value)
                    elif param == 'layer_number':
                        layer_data[param] = self._convert_to_int(value)
                    else:
                        layer_data[param] = value
            
            # اضافه کردن لایه فقط اگر داده‌های ضروری موجود باشد
            if self._is_valid_layer(layer_data):
                insulation_data.append(layer_data)
        
        return insulation_data
    
    def _clean_insulation_type(self, value) -> str:
        """
        پاک‌سازی و استاندارد کردن نوع عایق
        """
        if pd.isna(value):
            return ''
        
        value_str = str(value).strip()
        
        # تطبیق با انواع عایق‌های قابل قبول
        for valid_type in self.valid_insulation_types:
            if valid_type.lower() in value_str.lower():
                return valid_type
        
        return value_str
    
    def _convert_to_float(self, value) -> float:
        """
        تبدیل مقدار به float
        """
        if pd.isna(value):
            return 0.0
        
        try:
            # حذف کاراکترهای غیرعددی
            if isinstance(value, str):
                value = value.replace(',', '').replace(' ', '')
            
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    def _convert_to_int(self, value) -> int:
        """
        تبدیل مقدار به int
        """
        if pd.isna(value):
            return 0
        
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return 0
    
    def _is_valid_layer(self, layer_data: Dict) -> bool:
        """
        بررسی معتبر بودن داده‌های لایه
        """
        required_fields = ['insulation_type', 'thickness']
        
        for field in required_fields:
            if field not in layer_data or not layer_data[field]:
                return False
        
        # بررسی نوع عایق
        if layer_data['insulation_type'] not in self.valid_insulation_types:
            return False
        
        # بررسی ضخامت
        if layer_data['thickness'] <= 0:
            return False
        
        return True
    
    def _validate_insulation_data(self, insulation_data: List[Dict]) -> List[Dict]:
        """
        اعتبارسنجی و تکمیل داده‌های عایق
        """
        validated_data = []
        
        for layer in insulation_data:
            validated_layer = {}
            
            # کپی داده‌های موجود
            for key, value in layer.items():
                validated_layer[key] = value
            
            # تکمیل داده‌های مفقود
            if 'layer_number' not in validated_layer:
                validated_layer['layer_number'] = len(validated_data) + 1
            
            if 'density' not in validated_layer or validated_layer['density'] <= 0:
                validated_layer['density'] = self._get_default_density(validated_layer['insulation_type'])
            
            if 'thermal_conductivity' not in validated_layer or validated_layer['thermal_conductivity'] <= 0:
                validated_layer['thermal_conductivity'] = self._get_default_thermal_conductivity(validated_layer['insulation_type'])
            
            # تبدیل ضخامت از میلی‌متر به متر (در صورت نیاز)
            if validated_layer['thickness'] > 1:  # احتمالاً به میلی‌متر است
                validated_layer['thickness'] = validated_layer['thickness'] / 1000
            
            validated_data.append(validated_layer)
        
        return validated_data
    
    def _get_default_density(self, insulation_type: str) -> float:
        """
        چگالی پیش‌فرض برای انواع مختلف عایق
        """
        defaults = {
            'Cerablanket': 96.0,
            'Silika Needeled Mat': 120.0,
            'Rock Wool': 100.0,
            'Needeled Mat': 80.0
        }
        return defaults.get(insulation_type, 100.0)
    
    def _get_default_thermal_conductivity(self, insulation_type: str) -> float:
        """
        ضریب انتقال حرارت پیش‌فرض برای انواع مختلف عایق
        """
        defaults = {
            'Cerablanket': 0.04,
            'Silika Needeled Mat': 0.038,
            'Rock Wool': 0.042,
            'Needeled Mat': 0.045
        }
        return defaults.get(insulation_type, 0.04)
    
    def create_sample_excel(self, file_path: str = 'inputdata.xlsx'):
        """
        ایجاد فایل Excel نمونه با داده‌های عایق
        """
        sample_data = {
            'شماره لایه': [1, 2, 3, 4],
            'نوع عایق': ['Cerablanket', 'Rock Wool', 'Silika Needeled Mat', 'Needeled Mat'],
            'ضخامت (mm)': [25, 50, 30, 40],
            'چگالی (kg/m3)': [96, 100, 120, 80],
            'ضریب انتقال حرارت (W/m.K)': [0.04, 0.042, 0.038, 0.045]
        }
        
        df = pd.DataFrame(sample_data)
        df.to_excel(file_path, index=False)
        print(f"فایل نمونه {file_path} ایجاد شد")
    
    def get_layer_summary(self, insulation_data: List[Dict]) -> Dict:
        """
        خلاصه‌ای از لایه‌های عایق
        """
        if not insulation_data:
            return {}
        
        total_thickness = sum(layer['thickness'] for layer in insulation_data)
        layer_types = [layer['insulation_type'] for layer in insulation_data]
        
        summary = {
            'total_layers': len(insulation_data),
            'total_thickness': total_thickness,
            'insulation_types': layer_types,
            'average_thermal_conductivity': np.mean([layer['thermal_conductivity'] for layer in insulation_data])
        }
        
        return summary