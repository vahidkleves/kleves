#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Thermal Insulation Analysis System
With Heat Transfer Coefficient Calculation and Decision Tree Prediction
"""

import sqlite3
import json
import os
from datetime import datetime
import pickle
import re
import math

class ThermalData:
    """Class for storing thermal information"""
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

class HeatTransferCalculator:
    """Class for heat transfer calculations"""
    
    def __init__(self):
        # Thermal conductivity values for common insulation materials (W/m·K)
        self.thermal_conductivities = {
            'polyurethane': 0.025,  # پلی اورتان
            'foam': 0.035,         # فوم
            'glass wool': 0.040,   # پشم شیشه
            'پلی اورتان': 0.025,
            'فوم': 0.035,
            'پشم شیشه': 0.040
        }
        
        # Typical insulation thickness values (m)
        self.typical_thickness = {
            'pipe': 0.05,      # 5 cm for pipes
            'sphere': 0.08,    # 8 cm for spheres
            'cube': 0.06,      # 6 cm for cubes
            'surface': 0.10,   # 10 cm for surfaces
            'لوله': 0.05,
            'کره': 0.08,
            'مکعب': 0.06,
            'سطح': 0.10
        }
    
    def calculate_heat_flux(self, equipment_temp, insulation_temp, 
                          thermal_conductivity, thickness):
        """Calculate heat flux through insulation (W/m²)"""
        if thickness <= 0:
            return 0
        
        heat_flux = thermal_conductivity * (equipment_temp - insulation_temp) / thickness
        return heat_flux
    
    def calculate_convection_coefficient(self, insulation_temp, ambient_temp, 
                                       heat_flux):
        """Calculate convection heat transfer coefficient (W/m²·K)"""
        temp_diff = insulation_temp - ambient_temp
        if temp_diff <= 0:
            return 0
        
        h_conv = heat_flux / temp_diff
        return h_conv
    
    def estimate_convection_coefficient(self, geometry_type, insulation_temp, 
                                      ambient_temp=25):
        """Estimate convection coefficient based on geometry and temperature"""
        temp_diff = insulation_temp - ambient_temp
        
        if temp_diff <= 0:
            return 10  # Default minimum value
        
        # Natural convection correlations
        if geometry_type in ['pipe', 'لوله']:
            # Horizontal cylinder
            h_conv = 1.32 * (temp_diff ** 0.25)
        elif geometry_type in ['sphere', 'کره']:
            # Sphere
            h_conv = 2.0 + 0.6 * (temp_diff ** 0.25)
        elif geometry_type in ['surface', 'سطح']:
            # Vertical surface
            h_conv = 1.42 * (temp_diff ** 0.25)
        elif geometry_type in ['cube', 'مکعب']:
            # Cube (approximate as vertical surface)
            h_conv = 1.3 * (temp_diff ** 0.25)
        else:
            # Default correlation
            h_conv = 1.5 * (temp_diff ** 0.25)
        
        # Ensure minimum reasonable value
        return max(h_conv, 5.0)
    
    def get_thermal_conductivity(self, insulation_type):
        """Get thermal conductivity for insulation material"""
        return self.thermal_conductivities.get(insulation_type, 0.035)
    
    def get_typical_thickness(self, geometry_type):
        """Get typical insulation thickness for geometry"""
        return self.typical_thickness.get(geometry_type, 0.06)

class HTMLParser:
    """Class for processing HTML output files"""
    
    def __init__(self):
        self.supported_patterns = {
            'temperature': r'(\d+\.?\d*)\s*°?[CF]',
            'geometry': r'(pipe|sphere|cube|surface|لوله|کره|مکعب|سطح)',
            'insulation': r'(polyurethane|foam|glass\s*wool|پلی\s*اورتان|فوم|پشم\s*شیشه)',
            'area': r'(\d+\.?\d*)\s*(m²|square\s*meter)',
            'coefficient': r'(\d+\.?\d*)\s*(W/m²\.K|W/m2\.K)',
            'thickness': r'(\d+\.?\d*)\s*(cm|mm|m)'
        }
    
    def parse_html_file(self, file_path):
        """Process an HTML file and extract thermal information"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Remove HTML tags
            text = re.sub(r'<[^>]+>', ' ', content)
            
            # Extract information using regex
            extracted_data = {}
            for key, pattern in self.supported_patterns.items():
                matches = re.findall(pattern, text, re.IGNORECASE)
                extracted_data[key] = matches
            
            return self._process_extracted_data(extracted_data)
            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return None
    
    def _process_extracted_data(self, raw_data):
        """Process and clean extracted data"""
        processed = {}
        
        # Process temperatures
        if raw_data.get('temperature'):
            temps = [float(t) for t in raw_data['temperature']]
            processed['equipment_temp'] = max(temps) if temps else None
            processed['insulation_temp'] = min(temps) if len(temps) > 1 else None
        
        # Process geometry type
        if raw_data.get('geometry'):
            processed['geometry_type'] = raw_data['geometry'][0]
        
        # Process insulation type
        if raw_data.get('insulation'):
            processed['insulation_type'] = raw_data['insulation'][0]
        
        # Process cross-section area
        if raw_data.get('area'):
            processed['cross_section_area'] = float(raw_data['area'][0])
        
        # Process heat transfer coefficient
        if raw_data.get('coefficient'):
            processed['convection_coefficient'] = float(raw_data['coefficient'][0])
        
        # Process thickness
        if raw_data.get('thickness'):
            thickness_value = float(raw_data['thickness'][0])
            # Convert to meters if needed
            if 'cm' in str(raw_data['thickness']):
                thickness_value = thickness_value / 100
            elif 'mm' in str(raw_data['thickness']):
                thickness_value = thickness_value / 1000
            processed['thickness'] = thickness_value
        
        return processed

class ThermalDatabase:
    """Class for managing thermal database"""
    
    def __init__(self, db_path="thermal_data_enhanced.db"):
        self.db_path = db_path
        self._initialize_database()
    
    def _initialize_database(self):
        """Create database and required tables"""
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
                heat_flux REAL,
                calculated_h_conv REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def insert_data(self, thermal_data, heat_flux=None, calculated_h_conv=None):
        """Insert new data into database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO thermal_records 
            (geometry_type, equipment_surface_temp, insulation_surface_temp,
             insulation_type, cross_section_area, convection_coefficient,
             thermal_conductivity, thickness, heat_flux, calculated_h_conv)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            thermal_data.geometry_type,
            thermal_data.equipment_surface_temp,
            thermal_data.insulation_surface_temp,
            thermal_data.insulation_type,
            thermal_data.cross_section_area,
            thermal_data.convection_coefficient,
            thermal_data.thermal_conductivity,
            thermal_data.thickness,
            heat_flux,
            calculated_h_conv
        ))
        
        conn.commit()
        conn.close()
    
    def get_all_data(self):
        """Get all data from database"""
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

class DecisionTreeNode:
    """Node class for Decision Tree"""
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index  # Index of feature to split on
        self.threshold = threshold          # Threshold value for splitting
        self.left = left                   # Left child node
        self.right = right                 # Right child node
        self.value = value                 # Predicted value (for leaf nodes)

class DecisionTreePredictor:
    """Decision Tree predictor for thermal properties"""
    
    def __init__(self, max_depth=10, min_samples_split=3, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.is_trained = False
        self.temp_tree = None
        self.hconv_tree = None
        
        self.geometry_mapping = {
            'pipe': 1, 'لوله': 1,
            'sphere': 2, 'کره': 2,
            'cube': 3, 'مکعب': 3,
            'surface': 4, 'سطح': 4
        }
        self.insulation_mapping = {
            'polyurethane': 1, 'پلی اورتان': 1,
            'foam': 2, 'فوم': 2,
            'glass wool': 3, 'پشم شیشه': 3
        }
    
    def _calculate_mse(self, y):
        """Calculate Mean Squared Error for a set of target values"""
        if len(y) == 0:
            return 0
        mean_y = sum(y) / len(y)
        return sum((yi - mean_y) ** 2 for yi in y) / len(y)
    
    def _best_split(self, X, y):
        """Find the best feature and threshold to split on"""
        best_mse = float('inf')
        best_feature = None
        best_threshold = None
        
        n_features = len(X[0])
        
        for feature_index in range(n_features):
            # Get unique values for this feature
            feature_values = [x[feature_index] for x in X]
            unique_values = sorted(set(feature_values))
            
            # Try splitting at each unique value
            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2
                
                # Split data
                left_indices = [j for j, x in enumerate(X) if x[feature_index] <= threshold]
                right_indices = [j for j, x in enumerate(X) if x[feature_index] > threshold]
                
                if len(left_indices) < self.min_samples_leaf or len(right_indices) < self.min_samples_leaf:
                    continue
                
                # Calculate weighted MSE
                left_y = [y[j] for j in left_indices]
                right_y = [y[j] for j in right_indices]
                
                left_mse = self._calculate_mse(left_y)
                right_mse = self._calculate_mse(right_y)
                
                weighted_mse = (len(left_y) * left_mse + len(right_y) * right_mse) / len(y)
                
                if weighted_mse < best_mse:
                    best_mse = weighted_mse
                    best_feature = feature_index
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _build_tree(self, X, y, depth=0):
        """Build decision tree recursively"""
        # Base cases
        if depth >= self.max_depth or len(X) < self.min_samples_split or len(set(y)) == 1:
            return DecisionTreeNode(value=sum(y) / len(y))
        
        # Find best split
        best_feature, best_threshold = self._best_split(X, y)
        
        if best_feature is None:
            return DecisionTreeNode(value=sum(y) / len(y))
        
        # Split data
        left_indices = [i for i, x in enumerate(X) if x[best_feature] <= best_threshold]
        right_indices = [i for i, x in enumerate(X) if x[best_feature] > best_threshold]
        
        left_X = [X[i] for i in left_indices]
        left_y = [y[i] for i in left_indices]
        right_X = [X[i] for i in right_indices]
        right_y = [y[i] for i in right_indices]
        
        # Build child nodes
        left_child = self._build_tree(left_X, left_y, depth + 1)
        right_child = self._build_tree(right_X, right_y, depth + 1)
        
        return DecisionTreeNode(
            feature_index=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child
        )
    
    def _predict_single(self, tree, x):
        """Predict single sample using decision tree"""
        if tree.value is not None:
            return tree.value
        
        if x[tree.feature_index] <= tree.threshold:
            return self._predict_single(tree.left, x)
        else:
            return self._predict_single(tree.right, x)
    
    def train_model(self, data_list):
        """Train decision tree models"""
        if len(data_list) < 3:
            print("Not enough data for training. At least 3 samples needed.")
            return False
        
        # Prepare training data
        X_temp = []
        y_temp = []
        X_hconv = []
        y_hconv = []
        
        for record in data_list:
            geometry_code = self.geometry_mapping.get(record['geometry_type'], 0)
            insulation_code = self.insulation_mapping.get(record['insulation_type'], 0)
            
            # Features for temperature prediction
            features_temp = [
                record['equipment_surface_temp'],
                record['cross_section_area'],
                record['convection_coefficient'],
                geometry_code,
                insulation_code
            ]
            
            # Features for convection coefficient prediction
            features_hconv = [
                record['equipment_surface_temp'],
                record['insulation_surface_temp'],
                record['cross_section_area'],
                geometry_code,
                insulation_code
            ]
            
            X_temp.append(features_temp)
            y_temp.append(record['insulation_surface_temp'])
            X_hconv.append(features_hconv)
            y_hconv.append(record.get('calculated_h_conv', record['convection_coefficient']))
        
        # Build decision trees
        print("Building decision tree for temperature prediction...")
        self.temp_tree = self._build_tree(X_temp, y_temp)
        
        print("Building decision tree for convection coefficient prediction...")
        self.hconv_tree = self._build_tree(X_hconv, y_hconv)
        
        self.is_trained = True
        print(f"Decision trees trained with {len(data_list)} samples.")
        return True
    
    def predict_temperature(self, equipment_temp, cross_section_area, 
                          convection_coefficient, geometry_type, insulation_type):
        """Predict insulation surface temperature using decision tree"""
        
        if not self.is_trained:
            raise ValueError("Model has not been trained yet.")
        
        geometry_code = self.geometry_mapping.get(geometry_type, 0)
        insulation_code = self.insulation_mapping.get(insulation_type, 0)
        
        input_features = [equipment_temp, cross_section_area, convection_coefficient, 
                         geometry_code, insulation_code]
        
        prediction = self._predict_single(self.temp_tree, input_features)
        return prediction
    
    def predict_convection_coefficient(self, equipment_temp, insulation_temp, 
                                     cross_section_area, geometry_type, insulation_type):
        """Predict convection coefficient using decision tree"""
        
        if not self.is_trained:
            raise ValueError("Model has not been trained yet.")
        
        geometry_code = self.geometry_mapping.get(geometry_type, 0)
        insulation_code = self.insulation_mapping.get(insulation_type, 0)
        
        input_features = [equipment_temp, insulation_temp, cross_section_area, 
                         geometry_code, insulation_code]
        
        prediction = self._predict_single(self.hconv_tree, input_features)
        return prediction
    
    def print_tree_info(self):
        """Print information about the trained trees"""
        if not self.is_trained:
            print("Models have not been trained yet.")
            return
        
        def count_nodes(node):
            if node is None:
                return 0
            if node.value is not None:  # Leaf node
                return 1
            return 1 + count_nodes(node.left) + count_nodes(node.right)
        
        temp_nodes = count_nodes(self.temp_tree)
        hconv_nodes = count_nodes(self.hconv_tree)
        
        print(f"\nDecision Tree Model Information:")
        print(f"================================")
        print(f"Temperature prediction tree: {temp_nodes} nodes")
        print(f"Convection coefficient prediction tree: {hconv_nodes} nodes")
        print(f"Maximum depth: {self.max_depth}")
        print(f"Minimum samples for split: {self.min_samples_split}")
        print(f"Minimum samples per leaf: {self.min_samples_leaf}")

class EnhancedThermalAnalyzer:
    """Enhanced thermal analyzer with heat transfer calculations and decision tree prediction"""
    
    def __init__(self, db_path="thermal_data_enhanced.db"):
        self.database = ThermalDatabase(db_path)
        self.parser = HTMLParser()
        self.predictor = DecisionTreePredictor()
        self.heat_calculator = HeatTransferCalculator()
    
    def import_html_files(self, directory_path):
        """Import HTML files and calculate heat transfer properties"""
        if not os.path.exists(directory_path):
            print(f"Directory {directory_path} does not exist.")
            return
        
        imported_count = 0
        
        for filename in os.listdir(directory_path):
            if filename.lower().endswith('.html'):
                file_path = os.path.join(directory_path, filename)
                data = self.parser.parse_html_file(file_path)
                
                if data and self._validate_data(data):
                    # Get thermal properties
                    k_insulation = self.heat_calculator.get_thermal_conductivity(
                        data.get('insulation_type', 'foam'))
                    thickness = data.get('thickness') or self.heat_calculator.get_typical_thickness(
                        data.get('geometry_type', 'pipe'))
                    
                    # Calculate heat flux
                    heat_flux = self.heat_calculator.calculate_heat_flux(
                        data.get('equipment_temp', 0),
                        data.get('insulation_temp', 0),
                        k_insulation,
                        thickness
                    )
                    
                    # Calculate convection coefficient
                    calculated_h_conv = self.heat_calculator.calculate_convection_coefficient(
                        data.get('insulation_temp', 0),
                        25,  # ambient temperature
                        heat_flux
                    )
                    
                    thermal_data = ThermalData(
                        geometry_type=data.get('geometry_type', 'unknown'),
                        equipment_surface_temp=data.get('equipment_temp', 0),
                        insulation_surface_temp=data.get('insulation_temp', 0),
                        insulation_type=data.get('insulation_type', 'unknown'),
                        cross_section_area=data.get('cross_section_area', 0),
                        convection_coefficient=data.get('convection_coefficient', calculated_h_conv),
                        thermal_conductivity=k_insulation,
                        thickness=thickness
                    )
                    
                    self.database.insert_data(thermal_data, heat_flux, calculated_h_conv)
                    imported_count += 1
                    print(f"File {filename} successfully imported.")
                    print(f"  Calculated heat flux: {heat_flux:.2f} W/m²")
                    print(f"  Calculated h_conv: {calculated_h_conv:.2f} W/m²·K")
                else:
                    print(f"File {filename} does not have sufficient data.")
        
        print(f"Total {imported_count} files successfully imported.")
    
    def _validate_data(self, data):
        """Validate extracted data"""
        required_fields = ['equipment_temp', 'insulation_temp']
        return all(field in data and data[field] is not None for field in required_fields)
    
    def add_manual_data_enhanced(self, geometry_type, equipment_temp, insulation_temp, 
                               insulation_type, cross_section_area, ambient_temp=25, 
                               thickness=None):
        """Add thermal data with automatic calculations"""
        
        # Get thermal properties
        k_insulation = self.heat_calculator.get_thermal_conductivity(insulation_type)
        if not thickness:
            thickness = self.heat_calculator.get_typical_thickness(geometry_type)
        
        # Calculate heat flux
        heat_flux = self.heat_calculator.calculate_heat_flux(
            equipment_temp, insulation_temp, k_insulation, thickness)
        
        # Calculate convection coefficient
        calculated_h_conv = self.heat_calculator.calculate_convection_coefficient(
            insulation_temp, ambient_temp, heat_flux)
        
        thermal_data = ThermalData(
            geometry_type=geometry_type,
            equipment_surface_temp=equipment_temp,
            insulation_surface_temp=insulation_temp,
            insulation_type=insulation_type,
            cross_section_area=cross_section_area,
            convection_coefficient=calculated_h_conv,
            thermal_conductivity=k_insulation,
            thickness=thickness
        )
        
        self.database.insert_data(thermal_data, heat_flux, calculated_h_conv)
        
        print("Enhanced data successfully added:")
        print(f"  Thermal conductivity: {k_insulation:.3f} W/m·K")
        print(f"  Insulation thickness: {thickness:.3f} m")
        print(f"  Heat flux: {heat_flux:.2f} W/m²")
        print(f"  Calculated convection coefficient: {calculated_h_conv:.2f} W/m²·K")
    
    def calculate_convection_coefficient(self, equipment_temp, insulation_temp, 
                                       insulation_type, geometry_type, thickness=None, 
                                       ambient_temp=25):
        """Calculate convection coefficient from given parameters"""
        
        # Get thermal properties
        k_insulation = self.heat_calculator.get_thermal_conductivity(insulation_type)
        if not thickness:
            thickness = self.heat_calculator.get_typical_thickness(geometry_type)
        
        # Calculate heat flux through insulation
        heat_flux = self.heat_calculator.calculate_heat_flux(
            equipment_temp, insulation_temp, k_insulation, thickness)
        
        # Calculate convection coefficient
        h_conv = self.heat_calculator.calculate_convection_coefficient(
            insulation_temp, ambient_temp, heat_flux)
        
        print(f"\nHeat Transfer Calculation Results:")
        print(f"================================")
        print(f"Geometry: {geometry_type}")
        print(f"Insulation: {insulation_type}")
        print(f"Thermal conductivity: {k_insulation:.3f} W/m·K")
        print(f"Insulation thickness: {thickness:.3f} m")
        print(f"Equipment temperature: {equipment_temp:.1f} °C")
        print(f"Insulation surface temperature: {insulation_temp:.1f} °C")
        print(f"Ambient temperature: {ambient_temp:.1f} °C")
        print(f"Heat flux through insulation: {heat_flux:.2f} W/m²")
        print(f"Convection coefficient: {h_conv:.2f} W/m²·K")
        print(f"Temperature difference (insulation-ambient): {insulation_temp-ambient_temp:.1f} °C")
        
        return h_conv
    
    def train_prediction_model(self):
        """Train decision tree prediction model"""
        data_list = self.database.get_all_data()
        if len(data_list) == 0:
            print("No data available for training the model.")
            return False
        
        success = self.predictor.train_model(data_list)
        if success:
            self.predictor.print_tree_info()
        return success
    
    def predict_properties(self, equipment_temp, geometry_type, insulation_type, 
                         cross_section_area, ambient_temp=25, thickness=None):
        """Predict thermal properties for new configuration using decision tree"""
        
        try:
            # Get thermal properties
            k_insulation = self.heat_calculator.get_thermal_conductivity(insulation_type)
            if not thickness:
                thickness = self.heat_calculator.get_typical_thickness(geometry_type)
            
            # Estimate convection coefficient
            estimated_h_conv = self.heat_calculator.estimate_convection_coefficient(
                geometry_type, equipment_temp * 0.7, ambient_temp)  # Rough estimate
            
            # Predict insulation temperature using decision tree
            predicted_insulation_temp = self.predictor.predict_temperature(
                equipment_temp, cross_section_area, estimated_h_conv, 
                geometry_type, insulation_type)
            
            # Calculate accurate heat flux and convection coefficient
            heat_flux = self.heat_calculator.calculate_heat_flux(
                equipment_temp, predicted_insulation_temp, k_insulation, thickness)
            
            accurate_h_conv = self.heat_calculator.calculate_convection_coefficient(
                predicted_insulation_temp, ambient_temp, heat_flux)
            
            # Calculate efficiency
            temp_reduction = equipment_temp - predicted_insulation_temp
            efficiency = (temp_reduction / equipment_temp) * 100
            
            print(f"\nComplete Thermal Analysis Results (Decision Tree):")
            print(f"================================================")
            print(f"Input Parameters:")
            print(f"  Geometry: {geometry_type}")
            print(f"  Insulation: {insulation_type}")
            print(f"  Equipment temperature: {equipment_temp:.1f} °C")
            print(f"  Cross-section area: {cross_section_area:.2f} m²")
            print(f"  Ambient temperature: {ambient_temp:.1f} °C")
            print(f"\nMaterial Properties:")
            print(f"  Thermal conductivity: {k_insulation:.3f} W/m·K")
            print(f"  Insulation thickness: {thickness:.3f} m")
            print(f"\nDecision Tree Predictions:")
            print(f"  Insulation surface temperature: {predicted_insulation_temp:.1f} °C")
            print(f"  Heat flux: {heat_flux:.2f} W/m²")
            print(f"  Convection coefficient: {accurate_h_conv:.2f} W/m²·K")
            print(f"  Temperature reduction: {temp_reduction:.1f} °C")
            print(f"  Insulation efficiency: {efficiency:.1f}%")
            
            return {
                'insulation_temp': predicted_insulation_temp,
                'heat_flux': heat_flux,
                'convection_coefficient': accurate_h_conv,
                'efficiency': efficiency,
                'thermal_conductivity': k_insulation,
                'thickness': thickness
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None

def main():
    """Main enhanced program function with decision tree"""
    print("=== Enhanced Thermal Insulation Analysis System ===")
    print("With Heat Transfer Coefficient Calculation and Decision Tree Prediction")
    
    analyzer = EnhancedThermalAnalyzer()
    
    while True:
        print("\n" + "="*65)
        print("Available options:")
        print("1. Import HTML files (with heat transfer calculations)")
        print("2. Add manual data (with automatic calculations)")
        print("3. Calculate convection coefficient from parameters")
        print("4. Train decision tree prediction model")
        print("5. Complete thermal analysis prediction (Decision Tree)")
        print("6. View data statistics")
        print("7. View decision tree information")
        print("8. Exit")
        print("="*65)
        
        choice = input("\nYour choice (1-8): ").strip()
        
        if choice == '1':
            directory = input("Path to folder containing HTML files (default: ./html_files_english): ").strip()
            if not directory:
                directory = "./html_files_english"
            analyzer.import_html_files(directory)
        
        elif choice == '2':
            try:
                print("\nAdding enhanced thermal data:")
                geometry = input("Geometry type (pipe/sphere/cube/surface): ").strip()
                eq_temp = float(input("Equipment surface temperature (°C): "))
                ins_temp = float(input("Insulation surface temperature (°C): "))
                ins_type = input("Insulation type (polyurethane/foam/glass wool): ").strip()
                area = float(input("Cross-section area (m²): "))
                ambient = float(input("Ambient temperature (°C, default 25): ") or "25")
                thickness_input = input("Insulation thickness (m, press Enter for typical): ").strip()
                thickness = float(thickness_input) if thickness_input else None
                
                analyzer.add_manual_data_enhanced(geometry, eq_temp, ins_temp, 
                                                ins_type, area, ambient, thickness)
            except ValueError:
                print("Error: Please enter valid numerical values.")
        
        elif choice == '3':
            try:
                print("\nCalculating convection coefficient:")
                geometry = input("Geometry type (pipe/sphere/cube/surface): ").strip()
                eq_temp = float(input("Equipment surface temperature (°C): "))
                ins_temp = float(input("Insulation surface temperature (°C): "))
                ins_type = input("Insulation type (polyurethane/foam/glass wool): ").strip()
                ambient = float(input("Ambient temperature (°C, default 25): ") or "25")
                thickness_input = input("Insulation thickness (m, press Enter for typical): ").strip()
                thickness = float(thickness_input) if thickness_input else None
                
                h_conv = analyzer.calculate_convection_coefficient(
                    eq_temp, ins_temp, ins_type, geometry, thickness, ambient)
                
            except ValueError:
                print("Error: Please enter valid numerical values.")
        
        elif choice == '4':
            print("\nStarting decision tree model training...")
            analyzer.train_prediction_model()
        
        elif choice == '5':
            try:
                print("\nComplete thermal analysis prediction with decision tree:")
                geometry = input("Geometry type (pipe/sphere/cube/surface): ").strip()
                eq_temp = float(input("Equipment surface temperature (°C): "))
                ins_type = input("Insulation type (polyurethane/foam/glass wool): ").strip()
                area = float(input("Cross-section area (m²): "))
                ambient = float(input("Ambient temperature (°C, default 25): ") or "25")
                thickness_input = input("Insulation thickness (m, press Enter for typical): ").strip()
                thickness = float(thickness_input) if thickness_input else None
                
                results = analyzer.predict_properties(
                    eq_temp, geometry, ins_type, area, ambient, thickness)
                
            except ValueError:
                print("Error: Please enter valid numerical values.")
        
        elif choice == '6':
            data_list = analyzer.database.get_all_data()
            if len(data_list) > 0:
                print(f"\nEnhanced data statistics:")
                print(f"Total records: {len(data_list)}")
                
                geometries = list(set([d['geometry_type'] for d in data_list if d['geometry_type']]))
                insulations = list(set([d['insulation_type'] for d in data_list if d['insulation_type']]))
                eq_temps = [d['equipment_surface_temp'] for d in data_list if d['equipment_surface_temp']]
                ins_temps = [d['insulation_surface_temp'] for d in data_list if d['insulation_surface_temp']]
                h_convs = [d['calculated_h_conv'] for d in data_list if d.get('calculated_h_conv')]
                heat_fluxes = [d['heat_flux'] for d in data_list if d.get('heat_flux')]
                
                print(f"Available geometries: {geometries}")
                print(f"Available insulation types: {insulations}")
                print(f"Equipment temperature range: {min(eq_temps):.1f} to {max(eq_temps):.1f} °C")
                print(f"Insulation temperature range: {min(ins_temps):.1f} to {max(ins_temps):.1f} °C")
                
                if h_convs:
                    print(f"Convection coefficient range: {min(h_convs):.1f} to {max(h_convs):.1f} W/m²·K")
                if heat_fluxes:
                    print(f"Heat flux range: {min(heat_fluxes):.1f} to {max(heat_fluxes):.1f} W/m²")
            else:
                print("No data available.")
        
        elif choice == '7':
            analyzer.predictor.print_tree_info()
        
        elif choice == '8':
            print("Enhanced program with decision tree closed. Good luck!")
            break
        
        else:
            print("Invalid option. Please enter a number between 1 and 8.")

if __name__ == "__main__":
    main()