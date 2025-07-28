#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Thermal Insulation Analysis System
With Multi-Layer Insulation and Geometric Parameter-Based Equipment Identification
"""

import sqlite3
import json
import os
from datetime import datetime
import pickle
import re
import math

class InsulationLayer:
    """Class for storing individual insulation layer information"""
    def __init__(self, name, thickness, density, thermal_conductivity=None):
        self.name = name
        self.thickness = thickness  # in meters
        self.density = density      # in kg/m³
        self.thermal_conductivity = thermal_conductivity  # W/m·K

class EquipmentGeometry:
    """Class for storing equipment geometric parameters"""
    def __init__(self, geometry_type, **parameters):
        self.geometry_type = geometry_type
        self.parameters = parameters
    
    def get_characteristic_length(self):
        """Calculate characteristic length based on geometry type"""
        if self.geometry_type == 'pipe':
            return self.parameters.get('diameter', 0)
        elif self.geometry_type == 'sphere':
            return self.parameters.get('diameter', 0)
        elif self.geometry_type == 'cube':
            return self.parameters.get('side_length', 0)
        elif self.geometry_type == 'surface':
            length = self.parameters.get('length', 0)
            width = self.parameters.get('width', 0)
            return math.sqrt(length * width) if length and width else 0
        return 0
    
    def get_surface_area(self):
        """Calculate surface area based on geometry type"""
        if self.geometry_type == 'pipe':
            diameter = self.parameters.get('diameter', 0)
            length = self.parameters.get('length', 0)
            return math.pi * diameter * length
        elif self.geometry_type == 'sphere':
            diameter = self.parameters.get('diameter', 0)
            radius = diameter / 2
            return 4 * math.pi * radius**2
        elif self.geometry_type == 'cube':
            side = self.parameters.get('side_length', 0)
            return 6 * side**2
        elif self.geometry_type == 'surface':
            length = self.parameters.get('length', 0)
            width = self.parameters.get('width', 0)
            return length * width
        return 0

class ThermalData:
    """Class for storing thermal information with multi-layer support"""
    def __init__(self, equipment_geometry, equipment_surface_temp, 
                 insulation_surface_temp, insulation_layers,
                 convection_coefficient):
        self.equipment_geometry = equipment_geometry
        self.equipment_surface_temp = equipment_surface_temp
        self.insulation_surface_temp = insulation_surface_temp
        self.insulation_layers = insulation_layers  # List of InsulationLayer objects
        self.convection_coefficient = convection_coefficient
        self.timestamp = datetime.now()

class HeatTransferCalculator:
    """Class for heat transfer calculations with multi-layer support"""
    
    def __init__(self):
        # New insulation materials with their properties
        self.insulation_materials = {
            'cerablanket': {
                'name': 'Cerablanket',
                'thermal_conductivity': 0.040,  # W/m·K (typical for ceramic blanket)
                'available_thickness': [0.013, 0.025, 0.050],  # 13, 25, 50 mm
                'available_density': [96, 128]  # kg/m³
            },
            'silika_needeled_mat': {
                'name': 'Silika Needeled Mat',
                'thermal_conductivity': 0.035,  # W/m·K
                'available_thickness': [0.003, 0.012],  # 3, 12 mm
                'available_density': [150]  # kg/m³
            },
            'mineral_wool': {
                'name': 'Mineral Wool',
                'thermal_conductivity': 0.045,  # W/m·K
                'available_thickness': [0.025, 0.030, 0.040, 0.050, 0.070, 0.080, 0.100],  # mm
                'available_density': [130]  # kg/m³
            },
            'needeled_mat': {
                'name': 'Needeled Mat',
                'thermal_conductivity': 0.038,  # W/m·K
                'available_thickness': [0.006, 0.010, 0.012, 0.025],  # 6, 10, 12, 25 mm
                'available_density': [160]  # kg/m³
            }
        }
    
    def get_thermal_conductivity(self, material_name, density=None):
        """Get thermal conductivity for insulation material (may vary with density)"""
        material = self.insulation_materials.get(material_name.lower())
        if not material:
            return 0.040  # Default value
        
        # Thermal conductivity can be adjusted based on density
        base_k = material['thermal_conductivity']
        if density and len(material['available_density']) > 1:
            # Simple linear adjustment for density effect
            min_density = min(material['available_density'])
            max_density = max(material['available_density'])
            density_factor = 0.8 + 0.4 * (density - min_density) / (max_density - min_density)
            return base_k * density_factor
        
        return base_k
    
    def get_available_options(self, material_name):
        """Get available thickness and density options for a material"""
        material = self.insulation_materials.get(material_name.lower())
        if material:
            return {
                'thickness_options': material['available_thickness'],
                'density_options': material['available_density']
            }
        return None
    
    def calculate_multilayer_heat_flux(self, equipment_temp, insulation_temp, insulation_layers):
        """Calculate heat flux through multiple insulation layers"""
        if not insulation_layers:
            return 0
        
        total_thermal_resistance = 0
        
        for layer in insulation_layers:
            if layer.thickness <= 0:
                continue
            
            k_layer = layer.thermal_conductivity or self.get_thermal_conductivity(
                layer.name, layer.density)
            thermal_resistance = layer.thickness / k_layer
            total_thermal_resistance += thermal_resistance
        
        if total_thermal_resistance <= 0:
            return 0
        
        heat_flux = (equipment_temp - insulation_temp) / total_thermal_resistance
        return heat_flux
    
    def calculate_convection_coefficient(self, insulation_temp, ambient_temp, heat_flux):
        """Calculate convection heat transfer coefficient (W/m²·K)"""
        temp_diff = insulation_temp - ambient_temp
        if temp_diff <= 0:
            return 0
        
        h_conv = heat_flux / temp_diff
        return h_conv
    
    def estimate_convection_coefficient(self, equipment_geometry, insulation_temp, ambient_temp=25):
        """Estimate convection coefficient based on geometry and temperature"""
        temp_diff = insulation_temp - ambient_temp
        
        if temp_diff <= 0:
            return 10  # Default minimum value
        
        # Get characteristic length
        char_length = equipment_geometry.get_characteristic_length()
        
        # Natural convection correlations based on geometry
        if equipment_geometry.geometry_type == 'pipe':
            # Horizontal cylinder - Churchill and Chu correlation
            h_conv = 1.32 * (temp_diff / char_length) ** 0.25
        elif equipment_geometry.geometry_type == 'sphere':
            # Sphere - Ranz-Marshall correlation
            h_conv = 2.0 + 0.6 * (temp_diff / char_length) ** 0.25
        elif equipment_geometry.geometry_type == 'surface':
            # Vertical/horizontal surface
            h_conv = 1.42 * (temp_diff / char_length) ** 0.25
        elif equipment_geometry.geometry_type == 'cube':
            # Cube (approximate as vertical surface)
            h_conv = 1.3 * (temp_diff / char_length) ** 0.25
        else:
            # Default correlation
            h_conv = 1.5 * (temp_diff) ** 0.25
        
        # Ensure minimum reasonable value
        return max(h_conv, 5.0)

class HTMLParser:
    """Class for processing HTML output files"""
    
    def __init__(self):
        self.supported_patterns = {
            'temperature': r'(\d+\.?\d*)\s*°?[CF]',
            'geometry': r'(pipe|sphere|cube|surface|لوله|کره|مکعب|سطح)',
            'insulation': r'(cerablanket|silika.*needeled.*mat|mineral.*wool|needeled.*mat)',
            'area': r'(\d+\.?\d*)\s*(m²|square\s*meter)',
            'coefficient': r'(\d+\.?\d*)\s*(W/m²\.K|W/m2\.K)',
            'thickness': r'(\d+\.?\d*)\s*(cm|mm|m)',
            'density': r'(\d+\.?\d*)\s*(kg/m³|kg/m3)'
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
        
        # Process density
        if raw_data.get('density'):
            processed['density'] = float(raw_data['density'][0])
        
        return processed

class ThermalDatabase:
    """Class for managing thermal database with multi-layer support"""
    
    def __init__(self, db_path="thermal_data_enhanced.db"):
        self.db_path = db_path
        self._initialize_database()
    
    def _initialize_database(self):
        """Create database and required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Main thermal records table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS thermal_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                geometry_type TEXT NOT NULL,
                geometry_parameters TEXT,
                equipment_surface_temp REAL NOT NULL,
                insulation_surface_temp REAL NOT NULL,
                surface_area REAL,
                convection_coefficient REAL NOT NULL,
                heat_flux REAL,
                calculated_h_conv REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Insulation layers table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS insulation_layers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thermal_record_id INTEGER,
                layer_order INTEGER,
                material_name TEXT NOT NULL,
                thickness REAL NOT NULL,
                density REAL NOT NULL,
                thermal_conductivity REAL,
                FOREIGN KEY (thermal_record_id) REFERENCES thermal_records (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def insert_data(self, thermal_data, heat_flux=None, calculated_h_conv=None):
        """Insert new data into database with multi-layer support"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insert main thermal record
        geometry_params = json.dumps(thermal_data.equipment_geometry.parameters)
        surface_area = thermal_data.equipment_geometry.get_surface_area()
        
        cursor.execute('''
            INSERT INTO thermal_records 
            (geometry_type, geometry_parameters, equipment_surface_temp, insulation_surface_temp,
             surface_area, convection_coefficient, heat_flux, calculated_h_conv)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            thermal_data.equipment_geometry.geometry_type,
            geometry_params,
            thermal_data.equipment_surface_temp,
            thermal_data.insulation_surface_temp,
            surface_area,
            thermal_data.convection_coefficient,
            heat_flux,
            calculated_h_conv
        ))
        
        record_id = cursor.lastrowid
        
        # Insert insulation layers
        for i, layer in enumerate(thermal_data.insulation_layers):
            cursor.execute('''
                INSERT INTO insulation_layers
                (thermal_record_id, layer_order, material_name, thickness, density, thermal_conductivity)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                record_id,
                i + 1,
                layer.name,
                layer.thickness,
                layer.density,
                layer.thermal_conductivity
            ))
        
        conn.commit()
        conn.close()
    
    def get_all_data(self):
        """Get all data from database with layers"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get main records
        cursor.execute("SELECT * FROM thermal_records")
        records = cursor.fetchall()
        record_columns = [description[0] for description in cursor.description]
        
        data = []
        for record in records:
            record_dict = dict(zip(record_columns, record))
            
            # Get insulation layers for this record
            cursor.execute('''
                SELECT material_name, thickness, density, thermal_conductivity, layer_order
                FROM insulation_layers 
                WHERE thermal_record_id = ? 
                ORDER BY layer_order
            ''', (record_dict['id'],))
            
            layers = cursor.fetchall()
            record_dict['insulation_layers'] = [
                {
                    'material_name': layer[0],
                    'thickness': layer[1],
                    'density': layer[2],
                    'thermal_conductivity': layer[3],
                    'layer_order': layer[4]
                }
                for layer in layers
            ]
            
            # Parse geometry parameters
            if record_dict['geometry_parameters']:
                record_dict['geometry_parameters'] = json.loads(record_dict['geometry_parameters'])
            
            data.append(record_dict)
        
        conn.close()
        return data

class DecisionTreeNode:
    """Node class for Decision Tree"""
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTreePredictor:
    """Decision Tree predictor for thermal properties with multi-layer support"""
    
    def __init__(self, max_depth=10, min_samples_split=3, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.is_trained = False
        self.temp_tree = None
        self.hconv_tree = None
        
        self.geometry_mapping = {
            'pipe': 1,
            'sphere': 2,
            'cube': 3,
            'surface': 4
        }
        
        self.material_mapping = {
            'cerablanket': 1,
            'silika_needeled_mat': 2,
            'mineral_wool': 3,
            'needeled_mat': 4
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
            feature_values = [x[feature_index] for x in X]
            unique_values = sorted(set(feature_values))
            
            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2
                
                left_indices = [j for j, x in enumerate(X) if x[feature_index] <= threshold]
                right_indices = [j for j, x in enumerate(X) if x[feature_index] > threshold]
                
                if len(left_indices) < self.min_samples_leaf or len(right_indices) < self.min_samples_leaf:
                    continue
                
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
        if depth >= self.max_depth or len(X) < self.min_samples_split or len(set(y)) == 1:
            return DecisionTreeNode(value=sum(y) / len(y))
        
        best_feature, best_threshold = self._best_split(X, y)
        
        if best_feature is None:
            return DecisionTreeNode(value=sum(y) / len(y))
        
        left_indices = [i for i, x in enumerate(X) if x[best_feature] <= best_threshold]
        right_indices = [i for i, x in enumerate(X) if x[best_feature] > best_threshold]
        
        left_X = [X[i] for i in left_indices]
        left_y = [y[i] for i in left_indices]
        right_X = [X[i] for i in right_indices]
        right_y = [y[i] for i in right_indices]
        
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
    
    def _extract_features(self, record):
        """Extract features from a database record with multi-layer support"""
        geometry_code = self.geometry_mapping.get(record['geometry_type'], 0)
        
        # Multi-layer features
        total_thickness = sum(layer['thickness'] for layer in record['insulation_layers'])
        avg_density = sum(layer['density'] for layer in record['insulation_layers']) / len(record['insulation_layers'])
        
        # Material composition (simplified to primary material)
        primary_material = record['insulation_layers'][0]['material_name'] if record['insulation_layers'] else 'unknown'
        material_code = self.material_mapping.get(primary_material.lower(), 0)
        
        # Geometric features
        surface_area = record.get('surface_area', 0)
        
        return [
            record['equipment_surface_temp'],
            surface_area,
            total_thickness,
            avg_density,
            geometry_code,
            material_code,
            len(record['insulation_layers'])  # Number of layers
        ]
    
    def train_model(self, data_list):
        """Train decision tree models"""
        if len(data_list) < 3:
            print("Not enough data for training. At least 3 samples needed.")
            return False
        
        X_temp = []
        y_temp = []
        X_hconv = []
        y_hconv = []
        
        for record in data_list:
            if not record['insulation_layers']:
                continue
                
            features = self._extract_features(record)
            
            # Features for temperature prediction
            features_temp = features + [record.get('convection_coefficient', 0)]
            
            # Features for convection coefficient prediction  
            features_hconv = features + [record['insulation_surface_temp']]
            
            X_temp.append(features_temp)
            y_temp.append(record['insulation_surface_temp'])
            X_hconv.append(features_hconv)
            y_hconv.append(record.get('calculated_h_conv', record['convection_coefficient']))
        
        if len(X_temp) < 3:
            print("Not enough valid records for training.")
            return False
        
        print("Building decision tree for temperature prediction...")
        self.temp_tree = self._build_tree(X_temp, y_temp)
        
        print("Building decision tree for convection coefficient prediction...")
        self.hconv_tree = self._build_tree(X_hconv, y_hconv)
        
        self.is_trained = True
        print(f"Decision trees trained with {len(X_temp)} samples.")
        return True
    
    def predict_temperature(self, equipment_temp, surface_area, insulation_layers, 
                          convection_coefficient, geometry_type):
        """Predict insulation surface temperature using decision tree"""
        if not self.is_trained:
            raise ValueError("Model has not been trained yet.")
        
        geometry_code = self.geometry_mapping.get(geometry_type, 0)
        total_thickness = sum(layer.thickness for layer in insulation_layers)
        avg_density = sum(layer.density for layer in insulation_layers) / len(insulation_layers)
        
        primary_material = insulation_layers[0].name.lower() if insulation_layers else 'unknown'
        material_code = self.material_mapping.get(primary_material, 0)
        
        input_features = [
            equipment_temp, surface_area, total_thickness, avg_density,
            geometry_code, material_code, len(insulation_layers), convection_coefficient
        ]
        
        prediction = self._predict_single(self.temp_tree, input_features)
        return prediction
    
    def print_tree_info(self):
        """Print information about the trained trees"""
        if not self.is_trained:
            print("Models have not been trained yet.")
            return
        
        def count_nodes(node):
            if node is None:
                return 0
            if node.value is not None:
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
    """Enhanced thermal analyzer with multi-layer insulation support"""
    
    def __init__(self, db_path="thermal_data_enhanced.db"):
        self.database = ThermalDatabase(db_path)
        self.parser = HTMLParser()
        self.predictor = DecisionTreePredictor()
        self.heat_calculator = HeatTransferCalculator()
    
    def show_insulation_materials(self):
        """Display available insulation materials and their options"""
        print("\nAvailable Insulation Materials:")
        print("=" * 50)
        
        for material_key, material_info in self.heat_calculator.insulation_materials.items():
            print(f"\nMaterial: {material_info['name']}")
            print(f"  Thermal Conductivity: {material_info['thermal_conductivity']:.3f} W/m·K")
            
            print("  Available Thicknesses (mm):")
            thickness_mm = [t * 1000 for t in material_info['available_thickness']]
            print(f"    {thickness_mm}")
            
            print(f"  Available Densities (kg/m³): {material_info['available_density']}")
    
    def create_equipment_geometry(self):
        """Interactive creation of equipment geometry"""
        print("\nEquipment Geometry Configuration:")
        print("1. Pipe (cylinder)")
        print("2. Sphere")
        print("3. Cube") 
        print("4. Flat Surface")
        
        choice = input("Select geometry type (1-4): ").strip()
        
        if choice == '1':
            geometry_type = 'pipe'
            diameter = float(input("Enter pipe diameter (m): "))
            length = float(input("Enter pipe length (m): "))
            parameters = {'diameter': diameter, 'length': length}
        elif choice == '2':
            geometry_type = 'sphere'
            diameter = float(input("Enter sphere diameter (m): "))
            parameters = {'diameter': diameter}
        elif choice == '3':
            geometry_type = 'cube'
            side_length = float(input("Enter cube side length (m): "))
            parameters = {'side_length': side_length}
        elif choice == '4':
            geometry_type = 'surface'
            length = float(input("Enter surface length (m): "))
            width = float(input("Enter surface width (m): "))
            parameters = {'length': length, 'width': width}
        else:
            print("Invalid choice. Using default pipe geometry.")
            geometry_type = 'pipe'
            parameters = {'diameter': 0.1, 'length': 1.0}
        
        equipment_geometry = EquipmentGeometry(geometry_type, **parameters)
        
        print(f"\nGeometry created:")
        print(f"  Type: {geometry_type}")
        print(f"  Parameters: {parameters}")
        print(f"  Surface Area: {equipment_geometry.get_surface_area():.3f} m²")
        print(f"  Characteristic Length: {equipment_geometry.get_characteristic_length():.3f} m")
        
        return equipment_geometry
    
    def create_insulation_layers(self):
        """Interactive creation of insulation layers"""
        layers = []
        
        print("\nInsulation Layer Configuration:")
        self.show_insulation_materials()
        
        num_layers = int(input("\nNumber of insulation layers: "))
        
        for i in range(num_layers):
            print(f"\n--- Layer {i+1} ---")
            
            # Material selection
            print("Available materials:")
            materials = list(self.heat_calculator.insulation_materials.keys())
            for j, material in enumerate(materials, 1):
                print(f"  {j}. {self.heat_calculator.insulation_materials[material]['name']}")
            
            material_choice = int(input("Select material (number): ") or "1") - 1
            if material_choice < 0 or material_choice >= len(materials):
                material_choice = 0
            
            material_key = materials[material_choice]
            material_info = self.heat_calculator.insulation_materials[material_key]
            
            # Thickness selection
            print(f"Available thicknesses for {material_info['name']} (mm):")
            thickness_options = [t * 1000 for t in material_info['available_thickness']]
            for j, thickness in enumerate(thickness_options, 1):
                print(f"  {j}. {thickness} mm")
            
            thickness_choice = int(input("Select thickness (number): ") or "1") - 1
            if thickness_choice < 0 or thickness_choice >= len(thickness_options):
                thickness_choice = 0
            
            thickness_m = material_info['available_thickness'][thickness_choice]
            
            # Density selection
            print(f"Available densities for {material_info['name']} (kg/m³):")
            for j, density in enumerate(material_info['available_density'], 1):
                print(f"  {j}. {density} kg/m³")
            
            density_choice = int(input("Select density (number): ") or "1") - 1
            if density_choice < 0 or density_choice >= len(material_info['available_density']):
                density_choice = 0
            
            density = material_info['available_density'][density_choice]
            
            # Get thermal conductivity
            thermal_conductivity = self.heat_calculator.get_thermal_conductivity(material_key, density)
            
            # Create layer
            layer = InsulationLayer(
                name=material_key,
                thickness=thickness_m,
                density=density,
                thermal_conductivity=thermal_conductivity
            )
            
            layers.append(layer)
            
            print(f"Layer {i+1} created:")
            print(f"  Material: {material_info['name']}")
            print(f"  Thickness: {thickness_m*1000:.1f} mm")
            print(f"  Density: {density} kg/m³")
            print(f"  Thermal Conductivity: {thermal_conductivity:.3f} W/m·K")
        
        return layers
    
    def add_manual_data_enhanced(self):
        """Add thermal data with multi-layer insulation and geometric parameters"""
        try:
            print("\n=== Adding Enhanced Multi-Layer Thermal Data ===")
            
            # Create equipment geometry
            equipment_geometry = self.create_equipment_geometry()
            
            # Get temperatures
            equipment_temp = float(input("\nEquipment surface temperature (°C): "))
            insulation_temp = float(input("Insulation surface temperature (°C): "))
            ambient_temp = float(input("Ambient temperature (°C, default 25): ") or "25")
            
            # Create insulation layers
            insulation_layers = self.create_insulation_layers()
            
            # Calculate heat transfer properties
            heat_flux = self.heat_calculator.calculate_multilayer_heat_flux(
                equipment_temp, insulation_temp, insulation_layers)
            
            calculated_h_conv = self.heat_calculator.calculate_convection_coefficient(
                insulation_temp, ambient_temp, heat_flux)
            
            # Create thermal data object
            thermal_data = ThermalData(
                equipment_geometry=equipment_geometry,
                equipment_surface_temp=equipment_temp,
                insulation_surface_temp=insulation_temp,
                insulation_layers=insulation_layers,
                convection_coefficient=calculated_h_conv
            )
            
            # Save to database
            self.database.insert_data(thermal_data, heat_flux, calculated_h_conv)
            
            # Display results
            print("\n=== Enhanced Multi-Layer Data Successfully Added ===")
            print(f"Equipment geometry: {equipment_geometry.geometry_type}")
            print(f"Surface area: {equipment_geometry.get_surface_area():.3f} m²")
            print(f"Number of insulation layers: {len(insulation_layers)}")
            
            total_thickness = sum(layer.thickness for layer in insulation_layers) * 1000
            print(f"Total insulation thickness: {total_thickness:.1f} mm")
            
            print(f"Multi-layer heat flux: {heat_flux:.2f} W/m²")
            print(f"Calculated convection coefficient: {calculated_h_conv:.2f} W/m²·K")
            
            print("\nLayer details:")
            for i, layer in enumerate(insulation_layers, 1):
                material_name = self.heat_calculator.insulation_materials[layer.name]['name']
                print(f"  Layer {i}: {material_name} - {layer.thickness*1000:.1f}mm - {layer.density}kg/m³")
                
        except ValueError as e:
            print(f"Error: Invalid input - {e}")
        except Exception as e:
            print(f"Error adding data: {e}")
    
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
    
    def predict_properties(self):
        """Predict thermal properties for new multi-layer configuration"""
        try:
            print("\n=== Multi-Layer Thermal Analysis Prediction ===")
            
            # Create equipment geometry
            equipment_geometry = self.create_equipment_geometry()
            
            # Get equipment temperature
            equipment_temp = float(input("\nEquipment surface temperature (°C): "))
            ambient_temp = float(input("Ambient temperature (°C, default 25): ") or "25")
            
            # Create insulation layers
            insulation_layers = self.create_insulation_layers()
            
            # Estimate convection coefficient
            estimated_h_conv = self.heat_calculator.estimate_convection_coefficient(
                equipment_geometry, equipment_temp * 0.7, ambient_temp)
            
            # Predict insulation temperature using decision tree
            surface_area = equipment_geometry.get_surface_area()
            predicted_insulation_temp = self.predictor.predict_temperature(
                equipment_temp, surface_area, insulation_layers,
                estimated_h_conv, equipment_geometry.geometry_type)
            
            # Calculate accurate heat flux and convection coefficient
            heat_flux = self.heat_calculator.calculate_multilayer_heat_flux(
                equipment_temp, predicted_insulation_temp, insulation_layers)
            
            accurate_h_conv = self.heat_calculator.calculate_convection_coefficient(
                predicted_insulation_temp, ambient_temp, heat_flux)
            
            # Calculate efficiency
            temp_reduction = equipment_temp - predicted_insulation_temp
            efficiency = (temp_reduction / equipment_temp) * 100
            
            # Display results
            print(f"\n=== Multi-Layer Thermal Analysis Results ===")
            print(f"Equipment Geometry:")
            print(f"  Type: {equipment_geometry.geometry_type}")
            print(f"  Surface area: {surface_area:.3f} m²")
            print(f"  Characteristic length: {equipment_geometry.get_characteristic_length():.3f} m")
            
            print(f"\nInsulation System:")
            print(f"  Number of layers: {len(insulation_layers)}")
            total_thickness = sum(layer.thickness for layer in insulation_layers) * 1000
            print(f"  Total thickness: {total_thickness:.1f} mm")
            
            print(f"\nLayer composition:")
            for i, layer in enumerate(insulation_layers, 1):
                material_name = self.heat_calculator.insulation_materials[layer.name]['name']
                print(f"  Layer {i}: {material_name}")
                print(f"    Thickness: {layer.thickness*1000:.1f} mm")
                print(f"    Density: {layer.density} kg/m³")
                print(f"    k: {layer.thermal_conductivity:.3f} W/m·K")
            
            print(f"\nThermal Performance:")
            print(f"  Equipment temperature: {equipment_temp:.1f} °C")
            print(f"  Predicted insulation surface temp: {predicted_insulation_temp:.1f} °C")
            print(f"  Ambient temperature: {ambient_temp:.1f} °C")
            print(f"  Heat flux: {heat_flux:.2f} W/m²")
            print(f"  Convection coefficient: {accurate_h_conv:.2f} W/m²·K")
            print(f"  Temperature reduction: {temp_reduction:.1f} °C")
            print(f"  Insulation efficiency: {efficiency:.1f}%")
            
            return {
                'insulation_temp': predicted_insulation_temp,
                'heat_flux': heat_flux,
                'convection_coefficient': accurate_h_conv,
                'efficiency': efficiency,
                'equipment_geometry': equipment_geometry,
                'insulation_layers': insulation_layers
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None
    
    def view_data_statistics(self):
        """View enhanced data statistics"""
        data_list = self.database.get_all_data()
        if len(data_list) == 0:
            print("No data available.")
            return
        
        print(f"\n=== Enhanced Multi-Layer Data Statistics ===")
        print(f"Total records: {len(data_list)}")
        
        # Geometry statistics
        geometries = [d['geometry_type'] for d in data_list]
        geometry_counts = {geo: geometries.count(geo) for geo in set(geometries)}
        print(f"\nGeometry distribution:")
        for geo, count in geometry_counts.items():
            print(f"  {geo}: {count} records")
        
        # Temperature statistics
        eq_temps = [d['equipment_surface_temp'] for d in data_list]
        ins_temps = [d['insulation_surface_temp'] for d in data_list]
        print(f"\nTemperature ranges:")
        print(f"  Equipment: {min(eq_temps):.1f} to {max(eq_temps):.1f} °C")
        print(f"  Insulation surface: {min(ins_temps):.1f} to {max(ins_temps):.1f} °C")
        
        # Insulation layer statistics
        layer_counts = [len(d['insulation_layers']) for d in data_list]
        print(f"\nInsulation layers:")
        print(f"  Average layers per equipment: {sum(layer_counts)/len(layer_counts):.1f}")
        print(f"  Range: {min(layer_counts)} to {max(layer_counts)} layers")
        
        # Material usage
        all_materials = []
        for record in data_list:
            for layer in record['insulation_layers']:
                all_materials.append(layer['material_name'])
        
        material_counts = {mat: all_materials.count(mat) for mat in set(all_materials)}
        print(f"\nMaterial usage:")
        for material, count in material_counts.items():
            material_name = self.heat_calculator.insulation_materials.get(material, {}).get('name', material)
            print(f"  {material_name}: {count} layers")
        
        # Thickness statistics
        all_thicknesses = []
        for record in data_list:
            total_thickness = sum(layer['thickness'] for layer in record['insulation_layers'])
            all_thicknesses.append(total_thickness * 1000)  # Convert to mm
        
        if all_thicknesses:
            print(f"\nTotal thickness statistics (mm):")
            print(f"  Average: {sum(all_thicknesses)/len(all_thicknesses):.1f} mm")
            print(f"  Range: {min(all_thicknesses):.1f} to {max(all_thicknesses):.1f} mm")

def main():
    """Main enhanced program function with multi-layer insulation support"""
    print("=== Enhanced Multi-Layer Thermal Insulation Analysis System ===")
    print("With Geometric Parameter-Based Equipment Identification")
    
    analyzer = EnhancedThermalAnalyzer()
    
    while True:
        print("\n" + "="*70)
        print("Available options:")
        print("1. View insulation materials and options")
        print("2. Add manual multi-layer thermal data")
        print("3. Train decision tree prediction model")
        print("4. Multi-layer thermal analysis prediction")
        print("5. View enhanced data statistics")
        print("6. View decision tree information")
        print("7. Exit")
        print("="*70)
        
        choice = input("\nYour choice (1-7): ").strip()
        
        if choice == '1':
            analyzer.show_insulation_materials()
        
        elif choice == '2':
            analyzer.add_manual_data_enhanced()
        
        elif choice == '3':
            print("\nStarting decision tree model training...")
            analyzer.train_prediction_model()
        
        elif choice == '4':
            analyzer.predict_properties()
        
        elif choice == '5':
            analyzer.view_data_statistics()
        
        elif choice == '6':
            analyzer.predictor.print_tree_info()
        
        elif choice == '7':
            print("Enhanced multi-layer thermal analysis system closed. Good luck!")
            break
        
        else:
            print("Invalid option. Please enter a number between 1 and 7.")

if __name__ == "__main__":
    main()