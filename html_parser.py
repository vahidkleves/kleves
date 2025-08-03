"""
پارسر HTML برای خواندن داده‌های خروجی نرم‌افزار Colors
"""

from bs4 import BeautifulSoup
import re
import os
from typing import Dict, List, Optional

class ColorsHTMLParser:
    """
    کلاس پارسر برای تحلیل فایل‌های HTML خروجی نرم‌افزار Colors
    """
    
    def __init__(self):
        # الگوهای regex برای استخراج داده‌ها
        self.temperature_patterns = {
            'inner_temperature': [
                r'دمای سطح داخلی.*?(\d+\.?\d*)',
                r'Inner Surface Temperature.*?(\d+\.?\d*)',
                r'Internal Temperature.*?(\d+\.?\d*)',
                r'Equipment Temperature.*?(\d+\.?\d*)'
            ],
            'ambient_temperature': [
                r'دمای محیط.*?(\d+\.?\d*)',
                r'Ambient Temperature.*?(\d+\.?\d*)',
                r'Environment Temperature.*?(\d+\.?\d*)'
            ],
            'surface_temperature': [
                r'دمای سطح.*?(\d+\.?\d*)',
                r'Surface Temperature.*?(\d+\.?\d*)',
                r'Outer Surface Temperature.*?(\d+\.?\d*)'
            ]
        }
        
        self.other_patterns = {
            'wind_speed': [
                r'سرعت باد.*?(\d+\.?\d*)',
                r'Wind Speed.*?(\d+\.?\d*)',
                r'Air Speed.*?(\d+\.?\d*)'
            ],
            'insulation_thickness': [
                r'ضخامت عایق.*?(\d+\.?\d*)',
                r'Insulation Thickness.*?(\d+\.?\d*)',
                r'Total Thickness.*?(\d+\.?\d*)'
            ],
            'surface_area': [
                r'مساحت سطح.*?(\d+\.?\d*)',
                r'Surface Area.*?(\d+\.?\d*)',
                r'Total Area.*?(\d+\.?\d*)'
            ]
        }
        
        self.equipment_patterns = {
            'horizontal_pipe': ['لوله افقی', 'Horizontal Pipe', 'horizontal pipe'],
            'vertical_pipe': ['لوله عمودی', 'Vertical Pipe', 'vertical pipe'],
            'flat_horizontal': ['سطح صاف افقی', 'Flat Horizontal', 'horizontal surface'],
            'flat_vertical': ['سطح صاف عمودی', 'Flat Vertical', 'vertical surface'],
            'sphere': ['کره', 'Sphere', 'sphere'],
            'cube': ['مکعب', 'Cube', 'cube'],
            'turbine': ['توربین', 'Turbine', 'turbine'],
            'valve': ['ولو', 'Valve', 'valve']
        }
    
    def parse_html_file(self, file_path: str) -> Dict:
        """
        تحلیل فایل HTML و استخراج داده‌های مورد نیاز
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            soup = BeautifulSoup(content, 'html.parser')
            
            # استخراج داده‌ها از جدول Model Summary
            data = self._extract_from_model_summary(soup)
            
            # اگر جدول Model Summary پیدا نشد، از روش عمومی استفاده کن
            if not data:
                data = self._extract_from_general_content(content)
            
            # شناسایی نوع تجهیز
            equipment_type = self._identify_equipment_type(content)
            data['equipment_type'] = equipment_type
            
            # اعتبارسنجی داده‌ها
            validated_data = self._validate_extracted_data(data)
            
            return validated_data
            
        except FileNotFoundError:
            raise FileNotFoundError(f"فایل {file_path} پیدا نشد")
        except Exception as e:
            raise Exception(f"خطا در تحلیل فایل HTML: {str(e)}")
    
    def _extract_from_model_summary(self, soup: BeautifulSoup) -> Dict:
        """
        استخراج داده‌ها از جدول Model Summary
        """
        data = {}
        
        # جستجو برای جدول Model Summary
        summary_table = None
        
        # روش 1: جستجو با عنوان
        for heading in soup.find_all(['h1', 'h2', 'h3', 'h4']):
            if 'model summary' in heading.get_text().lower() or 'خلاصه مدل' in heading.get_text():
                summary_table = heading.find_next('table')
                break
        
        # روش 2: جستجو در div های مربوطه
        if not summary_table:
            for div in soup.find_all('div', class_=['summary', 'model-summary', 'report-summary']):
                table = div.find('table')
                if table:
                    summary_table = table
                    break
        
        # روش 3: جستجو در همه جدول‌ها
        if not summary_table:
            for table in soup.find_all('table'):
                if self._is_model_summary_table(table):
                    summary_table = table
                    break
        
        if summary_table:
            data = self._extract_from_table(summary_table)
        
        return data
    
    def _is_model_summary_table(self, table) -> bool:
        """
        بررسی اینکه آیا جدول مربوط به Model Summary است یا نه
        """
        text_content = table.get_text().lower()
        keywords = ['temperature', 'speed', 'thickness', 'area', 'دما', 'سرعت', 'ضخامت', 'مساحت']
        
        found_keywords = sum(1 for keyword in keywords if keyword in text_content)
        return found_keywords >= 3
    
    def _extract_from_table(self, table) -> Dict:
        """
        استخراج داده‌ها از جدول
        """
        data = {}
        rows = table.find_all('tr')
        
        for row in rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) >= 2:
                key = cells[0].get_text().strip().lower()
                value_text = cells[1].get_text().strip()
                
                # استخراج مقدار عددی
                value_match = re.search(r'(\d+\.?\d*)', value_text)
                if value_match:
                    value = float(value_match.group(1))
                    
                    # شناسایی نوع داده بر اساس کلیدواژه
                    if any(pattern in key for pattern in ['inner', 'internal', 'داخلی']):
                        data['inner_temperature'] = value
                    elif any(pattern in key for pattern in ['ambient', 'environment', 'محیط']):
                        data['ambient_temperature'] = value
                    elif any(pattern in key for pattern in ['surface', 'outer', 'سطح']):
                        data['surface_temperature'] = value
                    elif any(pattern in key for pattern in ['wind', 'air', 'باد', 'هوا']):
                        data['wind_speed'] = value
                    elif any(pattern in key for pattern in ['thickness', 'ضخامت']):
                        data['insulation_thickness'] = value
                    elif any(pattern in key for pattern in ['area', 'مساحت']):
                        data['surface_area'] = value
        
        return data
    
    def _extract_from_general_content(self, content: str) -> Dict:
        """
        استخراج داده‌ها از کل محتوای HTML با استفاده از regex
        """
        data = {}
        
        # استخراج دماها
        for temp_type, patterns in self.temperature_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, content, re.IGNORECASE | re.UNICODE)
                if match:
                    data[temp_type] = float(match.group(1))
                    break
        
        # استخراج سایر پارامترها
        for param_type, patterns in self.other_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, content, re.IGNORECASE | re.UNICODE)
                if match:
                    data[param_type] = float(match.group(1))
                    break
        
        return data
    
    def _identify_equipment_type(self, content: str) -> str:
        """
        شناسایی نوع تجهیز از محتوای HTML
        """
        content_lower = content.lower()
        
        for equipment_type, patterns in self.equipment_patterns.items():
            for pattern in patterns:
                if pattern.lower() in content_lower:
                    return equipment_type
        
        return 'unknown'
    
    def _validate_extracted_data(self, data: Dict) -> Dict:
        """
        اعتبارسنجی داده‌های استخراج شده
        """
        validated_data = {}
        
        # بررسی وجود داده‌های ضروری
        required_fields = ['inner_temperature', 'ambient_temperature', 'wind_speed', 'surface_area']
        
        for field in required_fields:
            if field in data and isinstance(data[field], (int, float)) and data[field] > 0:
                validated_data[field] = data[field]
            else:
                # مقادیر پیش‌فرض
                defaults = {
                    'inner_temperature': 100.0,
                    'ambient_temperature': 25.0,
                    'wind_speed': 2.0,
                    'surface_area': 1.0,
                    'insulation_thickness': 0.05
                }
                validated_data[field] = defaults.get(field, 0.0)
        
        # کپی سایر فیلدها
        for key, value in data.items():
            if key not in validated_data:
                validated_data[key] = value
        
        return validated_data
    
    def parse_multiple_files(self, directory_path: str) -> List[Dict]:
        """
        تحلیل چندین فایل HTML در یک پوشه
        """
        results = []
        
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"پوشه {directory_path} پیدا نشد")
        
        html_files = [f for f in os.listdir(directory_path) if f.endswith('.html')]
        
        for file_name in html_files:
            file_path = os.path.join(directory_path, file_name)
            try:
                data = self.parse_html_file(file_path)
                data['file_name'] = file_name
                results.append(data)
                print(f"✓ فایل {file_name} با موفقیت تحلیل شد")
            except Exception as e:
                print(f"✗ خطا در تحلیل فایل {file_name}: {str(e)}")
        
        return results