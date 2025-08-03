"""
محاسبات انتقال حرارت برای تجهیزات عایق‌کاری شده
"""

import numpy as np
import math

class HeatTransferCalculator:
    """
    کلاس محاسبه انتقال حرارت برای انواع مختلف تجهیزات
    """
    
    def __init__(self):
        # ضرایب انتقال حرارت جابجایی برای انواع مختلف عایق
        self.insulation_properties = {
            'Cerablanket': {'k': 0.04, 'density': 96},  # W/m.K
            'Silika Needeled Mat': {'k': 0.038, 'density': 120},
            'Rock Wool': {'k': 0.042, 'density': 100},
            'Needeled Mat': {'k': 0.045, 'density': 80}
        }
        
    def calculate_convection_coefficient(self, wind_speed, geometry_type='horizontal_pipe'):
        """
        محاسبه ضریب انتقال حرارت جابجایی بر اساس سرعت باد و نوع هندسه
        """
        if geometry_type in ['horizontal_pipe', 'vertical_pipe']:
            # برای لوله‌ها
            h = 5.7 + 3.8 * wind_speed
        elif geometry_type in ['flat_horizontal', 'flat_vertical']:
            # برای سطوح صاف
            h = 8.6 + 4.2 * wind_speed
        elif geometry_type == 'sphere':
            # برای کره
            h = 7.5 + 4.1 * wind_speed
        elif geometry_type == 'cube':
            # برای مکعب
            h = 6.8 + 3.9 * wind_speed
        else:
            # حالت عمومی
            h = 6.0 + 4.0 * wind_speed
            
        return h
    
    def calculate_thermal_resistance(self, layers_data, geometry_type, diameter=None):
        """
        محاسبه مقاومت حرارتی کل برای لایه‌های مختلف عایق
        """
        total_resistance = 0
        current_radius = diameter / 2 if diameter else None
        
        for layer in layers_data:
            thickness = layer['thickness']  # متر
            insulation_type = layer['type']
            k = self.insulation_properties[insulation_type]['k']
            
            if geometry_type in ['horizontal_pipe', 'vertical_pipe']:
                # مقاومت حرارتی برای لوله
                if current_radius:
                    r_inner = current_radius
                    r_outer = current_radius + thickness
                    resistance = math.log(r_outer / r_inner) / (2 * math.pi * k)
                    current_radius = r_outer
                else:
                    resistance = thickness / k
            else:
                # مقاومت حرارتی برای سطوح صاف
                resistance = thickness / k
                
            total_resistance += resistance
            
        return total_resistance
    
    def calculate_surface_temperature(self, inner_temp, ambient_temp, wind_speed, 
                                    layers_data, geometry_type, surface_area, 
                                    diameter=None):
        """
        محاسبه دمای سطح خارجی بر اساس انتقال حرارت
        """
        # محاسبه ضریب انتقال حرارت جابجایی
        h_conv = self.calculate_convection_coefficient(wind_speed, geometry_type)
        
        # محاسبه مقاومت حرارتی عایق‌ها
        R_insulation = self.calculate_thermal_resistance(layers_data, geometry_type, diameter)
        
        # مقاومت حرارتی جابجایی در سطح خارجی
        R_convection = 1 / (h_conv * surface_area)
        
        # محاسبه جریان حرارت
        total_resistance = R_insulation + R_convection
        heat_flow = (inner_temp - ambient_temp) / total_resistance
        
        # محاسبه دمای سطح
        surface_temp = ambient_temp + heat_flow * R_convection
        
        return surface_temp, heat_flow
    
    def calculate_heat_loss(self, inner_temp, surface_temp, layers_data, 
                          geometry_type, surface_area, diameter=None):
        """
        محاسبه تلفات حرارتی
        """
        R_insulation = self.calculate_thermal_resistance(layers_data, geometry_type, diameter)
        heat_loss = (inner_temp - surface_temp) / R_insulation * surface_area
        return heat_loss
    
    def validate_inputs(self, inner_temp, ambient_temp, wind_speed, layers_data):
        """
        اعتبارسنجی ورودی‌ها
        """
        errors = []
        
        if inner_temp <= ambient_temp:
            errors.append("دمای داخلی باید بیشتر از دمای محیط باشد")
        
        if wind_speed < 0:
            errors.append("سرعت باد نمی‌تواند منفی باشد")
            
        if not layers_data:
            errors.append("حداقل یک لایه عایق باید تعریف شود")
            
        for i, layer in enumerate(layers_data):
            if layer['thickness'] <= 0:
                errors.append(f"ضخامت لایه {i+1} باید مثبت باشد")
                
            if layer['type'] not in self.insulation_properties:
                errors.append(f"نوع عایق لایه {i+1} معتبر نیست")
        
        return errors