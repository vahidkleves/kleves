"""
کلاس اصلی تحلیل انتقال حرارت که همه اجزا را یکپارچه می‌کند
"""

import os
import json
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from html_parser import ColorsHTMLParser
from excel_reader import InsulationDataReader
from heat_calculations import HeatTransferCalculator
from ml_model import SurfaceTemperaturePredictor

class HeatTransferAnalyzer:
    """
    کلاس اصلی تحلیل انتقال حرارت و پیش‌بینی دمای سطح
    """
    
    def __init__(self):
        self.html_parser = ColorsHTMLParser()
        self.excel_reader = InsulationDataReader()
        self.heat_calculator = HeatTransferCalculator()
        self.ml_predictor = SurfaceTemperaturePredictor()
        
        # داده‌های بارگذاری شده
        self.colors_data = []
        self.insulation_data = []
        self.training_data = []
        
        # نتایج
        self.analysis_results = {}
        
    def load_colors_data(self, file_path: str = None, directory_path: str = None) -> List[Dict]:
        """
        بارگذاری داده‌های نرم‌افزار Colors از فایل یا پوشه
        """
        try:
            if file_path:
                # بارگذاری از یک فایل
                data = self.html_parser.parse_html_file(file_path)
                self.colors_data = [data]
                print(f"✓ داده‌های Colors از {file_path} بارگذاری شد")
                
            elif directory_path:
                # بارگذاری از پوشه
                self.colors_data = self.html_parser.parse_multiple_files(directory_path)
                print(f"✓ {len(self.colors_data)} فایل از پوشه {directory_path} بارگذاری شد")
                
            else:
                raise ValueError("یکی از file_path یا directory_path باید مشخص شود")
            
            return self.colors_data
            
        except Exception as e:
            print(f"خطا در بارگذاری داده‌های Colors: {str(e)}")
            return []
    
    def load_insulation_data(self, file_path: str) -> List[Dict]:
        """
        بارگذاری داده‌های عایق از فایل Excel
        """
        try:
            self.insulation_data = self.excel_reader.read_excel_file(file_path)
            print(f"✓ {len(self.insulation_data)} لایه عایق از {file_path} بارگذاری شد")
            
            # نمایش خلاصه لایه‌ها
            summary = self.excel_reader.get_layer_summary(self.insulation_data)
            if summary:
                print(f"  - تعداد لایه‌ها: {summary['total_layers']}")
                print(f"  - ضخامت کل: {summary['total_thickness']:.3f} متر")
                print(f"  - انواع عایق: {', '.join(summary['insulation_types'])}")
            
            return self.insulation_data
            
        except Exception as e:
            print(f"خطا در بارگذاری داده‌های عایق: {str(e)}")
            return []
    
    def calculate_theoretical_surface_temperature(self, colors_data: Dict = None, 
                                                insulation_layers: List[Dict] = None) -> Tuple[float, Dict]:
        """
        محاسبه نظری دمای سطح با استفاده از معادلات انتقال حرارت
        """
        # استفاده از داده‌های پیش‌فرض اگر ارائه نشده
        if colors_data is None:
            if not self.colors_data:
                raise ValueError("داده‌های Colors بارگذاری نشده است")
            colors_data = self.colors_data[0]
        
        if insulation_layers is None:
            if not self.insulation_data:
                raise ValueError("داده‌های عایق بارگذاری نشده است")
            insulation_layers = self.insulation_data
        
        # اعتبارسنجی ورودی‌ها
        errors = self.heat_calculator.validate_inputs(
            colors_data['inner_temperature'],
            colors_data['ambient_temperature'],
            colors_data['wind_speed'],
            insulation_layers
        )
        
        if errors:
            raise ValueError("خطاهای اعتبارسنجی: " + "; ".join(errors))
        
        # محاسبه دمای سطح
        surface_temp, heat_flow = self.heat_calculator.calculate_surface_temperature(
            inner_temp=colors_data['inner_temperature'],
            ambient_temp=colors_data['ambient_temperature'],
            wind_speed=colors_data['wind_speed'],
            layers_data=insulation_layers,
            geometry_type=colors_data.get('equipment_type', 'unknown'),
            surface_area=colors_data['surface_area']
        )
        
        # محاسبه تلفات حرارتی
        heat_loss = self.heat_calculator.calculate_heat_loss(
            inner_temp=colors_data['inner_temperature'],
            surface_temp=surface_temp,
            layers_data=insulation_layers,
            geometry_type=colors_data.get('equipment_type', 'unknown'),
            surface_area=colors_data['surface_area']
        )
        
        # اطلاعات تکمیلی
        info = {
            'method': 'theoretical_calculation',
            'heat_flow': heat_flow,
            'heat_loss': heat_loss,
            'temperature_reduction': colors_data['inner_temperature'] - surface_temp,
            'efficiency_percentage': ((colors_data['inner_temperature'] - surface_temp) / 
                                    (colors_data['inner_temperature'] - colors_data['ambient_temperature'])) * 100,
            'input_parameters': {
                'inner_temperature': colors_data['inner_temperature'],
                'ambient_temperature': colors_data['ambient_temperature'],
                'wind_speed': colors_data['wind_speed'],
                'surface_area': colors_data['surface_area'],
                'equipment_type': colors_data.get('equipment_type', 'unknown'),
                'total_insulation_thickness': sum(layer['thickness'] for layer in insulation_layers),
                'layer_count': len(insulation_layers)
            }
        }
        
        return surface_temp, info
    
    def prepare_training_data(self) -> pd.DataFrame:
        """
        آماده‌سازی داده‌ها برای آموزش مدل یادگیری ماشین
        """
        if not self.colors_data:
            raise ValueError("داده‌های Colors برای آموزش موجود نیست")
        
        # برای هر فایل Colors، از همان لایه‌های عایق استفاده می‌کنیم
        insulation_layers_list = [self.insulation_data for _ in self.colors_data]
        
        # آماده‌سازی ویژگی‌ها
        df = self.ml_predictor.prepare_features(self.colors_data, insulation_layers_list)
        
        print(f"✓ داده‌ها برای آموزش آماده شد: {len(df)} نمونه با {len(df.columns)} ویژگی")
        
        return df
    
    def train_ml_model(self) -> Dict:
        """
        آموزش مدل یادگیری ماشین
        """
        # آماده‌سازی داده‌ها
        df = self.prepare_training_data()
        
        # آموزش مدل‌ها
        results = self.ml_predictor.train_models(df)
        
        if results:
            print("\n=== نتایج آموزش مدل‌ها ===")
            for model_name, metrics in results.items():
                print(f"{model_name}:")
                print(f"  R² Score: {metrics['r2']:.4f}")
                print(f"  MSE: {metrics['mse']:.4f}")
                print(f"  MAE: {metrics['mae']:.4f}")
                print(f"  CV Score: {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")
                print()
        
        return results
    
    def predict_surface_temperature(self, colors_data: Dict = None, 
                                  insulation_layers: List[Dict] = None, 
                                  method: str = 'auto') -> Tuple[float, Dict]:
        """
        پیش‌بینی دمای سطح (با انتخاب روش)
        
        Args:
            colors_data: داده‌های تجهیز
            insulation_layers: لایه‌های عایق
            method: 'theoretical', 'ml', یا 'auto'
        """
        # استفاده از داده‌های پیش‌فرض
        if colors_data is None:
            if not self.colors_data:
                raise ValueError("داده‌های Colors موجود نیست")
            colors_data = self.colors_data[0]
        
        if insulation_layers is None:
            if not self.insulation_data:
                raise ValueError("داده‌های عایق موجود نیست")
            insulation_layers = self.insulation_data
        
        if method == 'theoretical':
            return self.calculate_theoretical_surface_temperature(colors_data, insulation_layers)
        
        elif method == 'ml':
            if not self.ml_predictor.is_trained:
                raise ValueError("مدل یادگیری ماشین آموزش نداده شده است")
            return self.ml_predictor.predict(colors_data, insulation_layers)
        
        elif method == 'auto':
            # اگر مدل آموزش دیده باشد، از آن استفاده کن، وگرنه محاسبه نظری
            if self.ml_predictor.is_trained:
                ml_result = self.ml_predictor.predict(colors_data, insulation_layers)
                theoretical_result = self.calculate_theoretical_surface_temperature(colors_data, insulation_layers)
                
                # ترکیب نتایج
                combined_temp = (ml_result[0] + theoretical_result[0]) / 2
                combined_info = {
                    'method': 'combined',
                    'ml_prediction': ml_result[0],
                    'theoretical_prediction': theoretical_result[0],
                    'combined_prediction': combined_temp,
                    'ml_info': ml_result[1],
                    'theoretical_info': theoretical_result[1]
                }
                return combined_temp, combined_info
            else:
                return self.calculate_theoretical_surface_temperature(colors_data, insulation_layers)
        
        else:
            raise ValueError("روش نامعتبر. باید یکی از 'theoretical', 'ml', یا 'auto' باشد")
    
    def analyze_equipment(self, equipment_data: Dict, method: str = 'auto') -> Dict:
        """
        تحلیل کامل یک تجهیز
        """
        results = {}
        
        try:
            # پیش‌بینی دمای سطح
            surface_temp, info = self.predict_surface_temperature(equipment_data, method=method)
            
            results['surface_temperature'] = surface_temp
            results['prediction_info'] = info
            results['equipment_data'] = equipment_data
            results['insulation_summary'] = self.excel_reader.get_layer_summary(self.insulation_data)
            
            # محاسبه پارامترهای عملکرد
            temp_reduction = equipment_data['inner_temperature'] - surface_temp
            efficiency = (temp_reduction / (equipment_data['inner_temperature'] - equipment_data['ambient_temperature'])) * 100
            
            results['performance'] = {
                'temperature_reduction': temp_reduction,
                'efficiency_percentage': efficiency,
                'is_safe': surface_temp < 60,  # فرض: دمای ایمن زیر 60 درجه
                'energy_saving': temp_reduction * equipment_data.get('surface_area', 1) * 10  # تقریبی
            }
            
            # توصیه‌ها
            recommendations = []
            if surface_temp > 60:
                recommendations.append("دمای سطح بالا است. افزایش ضخامت عایق توصیه می‌شود")
            if efficiency < 80:
                recommendations.append("کارایی عایق‌کاری پایین است. بررسی نوع عایق ضروری است")
            if len(self.insulation_data) < 2:
                recommendations.append("استفاده از چند لایه عایق می‌تواند عملکرد را بهبود بخشد")
            
            results['recommendations'] = recommendations
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def generate_report(self, output_file: str = 'analysis_report.json') -> str:
        """
        تولید گزارش کامل تحلیل
        """
        report = {
            'project_info': {
                'title': 'تحلیل انتقال حرارت و پیش‌بینی دمای سطح',
                'date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_equipment': len(self.colors_data),
                'insulation_layers': len(self.insulation_data)
            },
            'data_summary': {
                'colors_data_count': len(self.colors_data),
                'insulation_layers_summary': self.excel_reader.get_layer_summary(self.insulation_data),
                'ml_model_trained': self.ml_predictor.is_trained
            },
            'analysis_results': []
        }
        
        # تحلیل هر تجهیز
        for i, equipment_data in enumerate(self.colors_data):
            equipment_results = self.analyze_equipment(equipment_data)
            equipment_results['equipment_id'] = i + 1
            equipment_results['file_name'] = equipment_data.get('file_name', f'equipment_{i+1}')
            report['analysis_results'].append(equipment_results)
        
        # اهمیت ویژگی‌ها (اگر مدل آموزش دیده باشد)
        if self.ml_predictor.is_trained:
            feature_importance = self.ml_predictor.get_feature_importance()
            report['feature_importance'] = feature_importance
        
        # ذخیره گزارش
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"✓ گزارش در فایل {output_file} ذخیره شد")
        
        return output_file
    
    def save_model(self, file_path: str = 'heat_transfer_model.pkl'):
        """
        ذخیره مدل آموزش دیده
        """
        if self.ml_predictor.is_trained:
            self.ml_predictor.save_model(file_path)
        else:
            print("مدل هنوز آموزش نداده شده است")
    
    def load_model(self, file_path: str = 'heat_transfer_model.pkl'):
        """
        بارگذاری مدل ذخیره شده
        """
        try:
            self.ml_predictor.load_model(file_path)
            print("✓ مدل با موفقیت بارگذاری شد")
        except Exception as e:
            print(f"خطا در بارگذاری مدل: {str(e)}")
    
    def create_sample_data(self):
        """
        ایجاد فایل‌های نمونه برای تست
        """
        # ایجاد فایل Excel نمونه
        self.excel_reader.create_sample_excel('inputdata.xlsx')
        
        # ایجاد فایل HTML نمونه
        sample_html = '''<!DOCTYPE html>
<html>
<head>
    <title>Colors Analysis Report</title>
    <meta charset="utf-8">
</head>
<body>
    <h2>Model Summary</h2>
    <table>
        <thead>
            <tr>
                <th>Parameter</th>
                <th>Value</th>
                <th>Unit</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Inner Surface Temperature</td>
                <td>500</td>
                <td>°C</td>
            </tr>
            <tr>
                <td>Ambient Temperature</td>
                <td>25</td>
                <td>°C</td>
            </tr>
            <tr>
                <td>Wind Speed</td>
                <td>3</td>
                <td>m/s</td>
            </tr>
            <tr>
                <td>Surface Area</td>
                <td>10.5</td>
                <td>m²</td>
            </tr>
            <tr>
                <td>Equipment Type</td>
                <td>Horizontal Pipe</td>
                <td>-</td>
            </tr>
            <tr>
                <td>Surface Temperature</td>
                <td>65</td>
                <td>°C</td>
            </tr>
        </tbody>
    </table>
</body>
</html>'''
        
        with open('sample_colors_output.html', 'w', encoding='utf-8') as f:
            f.write(sample_html)
        
        print("✓ فایل‌های نمونه ایجاد شدند:")
        print("  - inputdata.xlsx")
        print("  - sample_colors_output.html")
    
    def get_status(self) -> Dict:
        """
        وضعیت فعلی سیستم
        """
        return {
            'colors_data_loaded': len(self.colors_data) > 0,
            'insulation_data_loaded': len(self.insulation_data) > 0,
            'ml_model_trained': self.ml_predictor.is_trained,
            'colors_data_count': len(self.colors_data),
            'insulation_layers_count': len(self.insulation_data),
            'ready_for_analysis': len(self.colors_data) > 0 and len(self.insulation_data) > 0
        }