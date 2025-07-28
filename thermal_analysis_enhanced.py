#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Thermal Insulation Analysis System
With Multi-Layer Insulation and Neural Network Prediction
"""

import sqlite3
import json
import os
from datetime import datetime
import pickle
import re
import math
import numpy as np
from typing import List, Tuple, Optional

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
    """Enhanced class for processing HTML output files and extracting thermal data"""
    
    def __init__(self):
        # Enhanced regex patterns for better data extraction
        self.supported_patterns = {
            'temperature': r'(\d+\.?\d*)\s*°?[CF]?\s*(?:درجه|degree|temp)',
            'geometry': r'(pipe|cylinder|sphere|cube|surface|rectangular|لوله|کره|مکعب|سطح|استوانه)',
            'insulation': r'(cerablanket|ceramic.*blanket|silika.*needeled.*mat|silika.*mat|mineral.*wool|needeled.*mat|عایق|ceramic|سرامیک)',
            'area': r'(\d+\.?\d*)\s*(m²|m2|square.*meter|متر.*مربع)',
            'coefficient': r'(\d+\.?\d*)\s*(W/m²\.K|W/m2\.K|W/m²K|W/m2K)',
            'thickness': r'(\d+\.?\d*)\s*(cm|mm|m|سانتی.*متر|میلی.*متر|متر)',
            'density': r'(\d+\.?\d*)\s*(kg/m³|kg/m3|کیلوگرم.*متر.*مکعب)',
            'diameter': r'(?:diameter|قطر).*?(\d+\.?\d*)\s*(cm|mm|m)',
            'length': r'(?:length|طول).*?(\d+\.?\d*)\s*(cm|mm|m)',
            'width': r'(?:width|عرض).*?(\d+\.?\d*)\s*(cm|mm|m)',
            'height': r'(?:height|ارتفاع).*?(\d+\.?\d*)\s*(cm|mm|m)',
            'side': r'(?:side|ضلع).*?(\d+\.?\d*)\s*(cm|mm|m)'
        }
        
        # Material mapping for different naming conventions
        self.material_mapping = {
            'ceramic': 'cerablanket',
            'ceramic blanket': 'cerablanket',
            'cerablanket': 'cerablanket',
            'silika': 'silika_needeled_mat',
            'silika mat': 'silika_needeled_mat',
            'silika needeled mat': 'silika_needeled_mat',
            'mineral wool': 'mineral_wool',
            'wool': 'mineral_wool',
            'needeled mat': 'needeled_mat',
            'mat': 'needeled_mat',
            'سرامیک': 'cerablanket',
            'عایق سرامیک': 'cerablanket'
        }
        
        # Geometry mapping
        self.geometry_mapping = {
            'pipe': 'pipe',
            'cylinder': 'pipe',
            'لوله': 'pipe',
            'استوانه': 'pipe',
            'sphere': 'sphere',
            'کره': 'sphere',
            'cube': 'cube',
            'مکعب': 'cube',
            'surface': 'surface',
            'rectangular': 'surface',
            'سطح': 'surface'
        }
    
    def parse_html_file(self, file_path):
        """Enhanced HTML file processing with better data extraction"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Remove HTML tags but preserve structure
            text = re.sub(r'<script.*?</script>', '', content, flags=re.DOTALL)
            text = re.sub(r'<style.*?</style>', '', text, flags=re.DOTALL)
            text = re.sub(r'<[^>]+>', ' ', text)
            
            # Clean up text
            text = re.sub(r'\s+', ' ', text)
            text = text.lower()
            
            # Extract information using regex
            extracted_data = {}
            for key, pattern in self.supported_patterns.items():
                matches = re.findall(pattern, text, re.IGNORECASE)
                extracted_data[key] = matches
            
            return self._process_extracted_data(extracted_data, file_path)
            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return None
    
    def _process_extracted_data(self, raw_data, file_path):
        """Enhanced data processing with intelligent defaults"""
        processed = {}
        
        # Process temperatures with better logic
        if raw_data.get('temperature'):
            temps = []
            for temp_str in raw_data['temperature']:
                try:
                    temp = float(temp_str)
                    if 0 < temp < 2000:  # Reasonable temperature range
                        temps.append(temp)
                except ValueError:
                    continue
            
            if len(temps) >= 2:
                temps.sort(reverse=True)
                processed['equipment_temp'] = temps[0]
                processed['insulation_temp'] = temps[1]
            elif len(temps) == 1:
                # Estimate insulation temperature as 70% of equipment temperature
                processed['equipment_temp'] = temps[0]
                processed['insulation_temp'] = temps[0] * 0.7
        
        # Process geometry type with mapping
        if raw_data.get('geometry'):
            geo_raw = raw_data['geometry'][0].lower()
            processed['geometry_type'] = self.geometry_mapping.get(geo_raw, 'pipe')
        else:
            processed['geometry_type'] = 'pipe'  # Default
        
        # Process insulation type with intelligent mapping
        if raw_data.get('insulation'):
            ins_raw = raw_data['insulation'][0].lower()
            for key, mapped_value in self.material_mapping.items():
                if key in ins_raw:
                    processed['insulation_type'] = mapped_value
                    break
            else:
                processed['insulation_type'] = 'cerablanket'  # Default
        else:
            processed['insulation_type'] = 'cerablanket'  # Default
        
        # Process geometric dimensions
        dimensions = self._extract_dimensions(raw_data)
        processed.update(dimensions)
        
        # Process cross-section area
        if raw_data.get('area'):
            try:
                processed['cross_section_area'] = float(raw_data['area'][0])
            except ValueError:
                pass
        
        # Calculate area if not provided but dimensions are available
        if 'cross_section_area' not in processed:
            processed['cross_section_area'] = self._calculate_area_from_dimensions(
                processed['geometry_type'], dimensions)
        
        # Process heat transfer coefficient
        if raw_data.get('coefficient'):
            try:
                processed['convection_coefficient'] = float(raw_data['coefficient'][0])
            except ValueError:
                pass
        
        # Process thickness
        thickness_data = self._process_thickness(raw_data)
        if thickness_data:
            processed.update(thickness_data)
        
        # Process density
        if raw_data.get('density'):
            try:
                processed['density'] = float(raw_data['density'][0])
            except ValueError:
                pass
        
        # Add file source for tracking
        processed['source_file'] = os.path.basename(file_path)
        
        return processed
    
    def _extract_dimensions(self, raw_data):
        """Extract and convert geometric dimensions"""
        dimensions = {}
        
        for dim_type in ['diameter', 'length', 'width', 'height', 'side']:
            if raw_data.get(dim_type):
                try:
                    value = float(raw_data[dim_type][0])
                    unit = raw_data[dim_type][1] if len(raw_data[dim_type]) > 1 else 'm'
                    
                    # Convert to meters
                    if unit in ['cm', 'سانتی متر']:
                        value = value / 100
                    elif unit in ['mm', 'میلی متر']:
                        value = value / 1000
                    
                    dimensions[dim_type] = value
                except (ValueError, IndexError):
                    continue
        
        return dimensions
    
    def _calculate_area_from_dimensions(self, geometry_type, dimensions):
        """Calculate surface area from dimensions"""
        if geometry_type == 'pipe':
            diameter = dimensions.get('diameter', 0.1)
            length = dimensions.get('length', 1.0)
            return math.pi * diameter * length
        elif geometry_type == 'sphere':
            diameter = dimensions.get('diameter', 0.1)
            radius = diameter / 2
            return 4 * math.pi * radius**2
        elif geometry_type == 'cube':
            side = dimensions.get('side', 0.1)
            return 6 * side**2
        elif geometry_type == 'surface':
            length = dimensions.get('length', 1.0)
            width = dimensions.get('width', 1.0)
            return length * width
        else:
            return 1.0  # Default 1 m²
    
    def _process_thickness(self, raw_data):
        """Process thickness data with unit conversion"""
        if not raw_data.get('thickness'):
            return None
        
        try:
            thickness_value = float(raw_data['thickness'][0])
            unit = raw_data['thickness'][1] if len(raw_data['thickness']) > 1 else 'm'
            
            # Convert to meters
            if unit in ['cm', 'سانتی متر']:
                thickness_value = thickness_value / 100
            elif unit in ['mm', 'میلی متر']:
                thickness_value = thickness_value / 1000
            
            return {'thickness': thickness_value}
        except (ValueError, IndexError):
            return None

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
                source_file TEXT,
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
    
    def insert_data(self, thermal_data, heat_flux=None, calculated_h_conv=None, source_file=None):
        """Insert new data into database with multi-layer support"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insert main thermal record
        geometry_params = json.dumps(thermal_data.equipment_geometry.parameters)
        surface_area = thermal_data.equipment_geometry.get_surface_area()
        
        cursor.execute('''
            INSERT INTO thermal_records 
            (geometry_type, geometry_parameters, equipment_surface_temp, insulation_surface_temp,
             surface_area, convection_coefficient, heat_flux, calculated_h_conv, source_file)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            thermal_data.equipment_geometry.geometry_type,
            geometry_params,
            thermal_data.equipment_surface_temp,
            thermal_data.insulation_surface_temp,
            surface_area,
            thermal_data.convection_coefficient,
            heat_flux,
            calculated_h_conv,
            source_file
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

class ActivationFunction:
    """Activation functions for neural network"""
    
    @staticmethod
    def sigmoid(x):
        """Sigmoid activation function"""
        # Clip x to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoid_derivative(x):
        """Derivative of sigmoid function"""
        s = ActivationFunction.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def relu(x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        """Derivative of ReLU function"""
        return (x > 0).astype(float)
    
    @staticmethod
    def tanh(x):
        """Tanh activation function"""
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x):
        """Derivative of tanh function"""
        return 1 - np.tanh(x) ** 2
    
    @staticmethod
    def linear(x):
        """Linear activation function"""
        return x
    
    @staticmethod
    def linear_derivative(x):
        """Derivative of linear function"""
        return np.ones_like(x)

class NeuralLayer:
    """Individual layer in neural network"""
    
    def __init__(self, input_size: int, output_size: int, activation: str = 'sigmoid'):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        
        # Initialize weights with Xavier/Glorot initialization
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / (input_size + output_size))
        self.biases = np.zeros((1, output_size))
        
        # For momentum and adaptive learning
        self.weights_momentum = np.zeros_like(self.weights)
        self.biases_momentum = np.zeros_like(self.biases)
        
        # Store last input and output for backpropagation
        self.last_input = None
        self.last_z = None
        self.last_output = None
    
    def forward(self, inputs):
        """Forward pass through the layer"""
        self.last_input = inputs
        self.last_z = np.dot(inputs, self.weights) + self.biases
        
        # Apply activation function
        if self.activation == 'sigmoid':
            self.last_output = ActivationFunction.sigmoid(self.last_z)
        elif self.activation == 'relu':
            self.last_output = ActivationFunction.relu(self.last_z)
        elif self.activation == 'tanh':
            self.last_output = ActivationFunction.tanh(self.last_z)
        elif self.activation == 'linear':
            self.last_output = ActivationFunction.linear(self.last_z)
        
        return self.last_output
    
    def backward(self, error_gradient, learning_rate=0.01, momentum=0.9):
        """Backward pass through the layer"""
        # Calculate activation derivative
        if self.activation == 'sigmoid':
            activation_derivative = ActivationFunction.sigmoid_derivative(self.last_z)
        elif self.activation == 'relu':
            activation_derivative = ActivationFunction.relu_derivative(self.last_z)
        elif self.activation == 'tanh':
            activation_derivative = ActivationFunction.tanh_derivative(self.last_z)
        elif self.activation == 'linear':
            activation_derivative = ActivationFunction.linear_derivative(self.last_z)
        
        # Calculate deltas
        delta = error_gradient * activation_derivative
        
        # Calculate gradients
        weights_gradient = np.dot(self.last_input.T, delta)
        biases_gradient = np.sum(delta, axis=0, keepdims=True)
        
        # Calculate error to propagate to previous layer
        error_to_propagate = np.dot(delta, self.weights.T)
        
        # Update weights and biases with momentum
        self.weights_momentum = momentum * self.weights_momentum + learning_rate * weights_gradient
        self.biases_momentum = momentum * self.biases_momentum + learning_rate * biases_gradient
        
        self.weights -= self.weights_momentum
        self.biases -= self.biases_momentum
        
        return error_to_propagate

class NeuralNetworkPredictor:
    """Neural Network predictor for thermal properties with multi-layer support"""
    
    def __init__(self, hidden_layers=[20, 15, 10], learning_rate=0.01, epochs=1000, 
                 batch_size=32, validation_split=0.2):
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        
        self.is_trained = False
        self.temp_network = None
        self.hconv_network = None
        
        # Feature scaling parameters
        self.feature_scaler_params = None
        self.temp_target_scaler_params = None
        self.hconv_target_scaler_params = None
        
        # Training history
        self.training_history = {
            'temp_loss': [],
            'temp_val_loss': [],
            'hconv_loss': [],
            'hconv_val_loss': []
        }
        
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
    
    def _normalize_features(self, features, scaler_params=None):
        """Normalize features using min-max scaling"""
        features = np.array(features)
        
        if scaler_params is None:
            # Calculate scaling parameters
            min_vals = np.min(features, axis=0)
            max_vals = np.max(features, axis=0)
            
            # Avoid division by zero
            ranges = max_vals - min_vals
            ranges[ranges == 0] = 1
            
            scaler_params = {'min': min_vals, 'max': max_vals, 'range': ranges}
        
        # Apply scaling
        normalized = (features - scaler_params['min']) / scaler_params['range']
        return normalized, scaler_params
    
    def _denormalize_target(self, normalized_target, scaler_params):
        """Denormalize target values"""
        return normalized_target * scaler_params['range'] + scaler_params['min']
    
    def _create_network(self, input_size, output_size=1):
        """Create a neural network with specified architecture"""
        layers = []
        
        # Input layer to first hidden layer
        if self.hidden_layers:
            layers.append(NeuralLayer(input_size, self.hidden_layers[0], 'relu'))
            
            # Hidden layers
            for i in range(1, len(self.hidden_layers)):
                layers.append(NeuralLayer(self.hidden_layers[i-1], self.hidden_layers[i], 'relu'))
            
            # Last hidden layer to output
            layers.append(NeuralLayer(self.hidden_layers[-1], output_size, 'linear'))
        else:
            # Direct input to output
            layers.append(NeuralLayer(input_size, output_size, 'linear'))
        
        return layers
    
    def _forward_pass(self, network, inputs):
        """Forward pass through the entire network"""
        current_input = inputs
        for layer in network:
            current_input = layer.forward(current_input)
        return current_input
    
    def _backward_pass(self, network, target_output):
        """Backward pass through the entire network"""
        # Calculate output error
        output_error = network[-1].last_output - target_output
        
        # Backpropagate through layers (reverse order)
        current_error = output_error
        for layer in reversed(network):
            current_error = layer.backward(current_error, self.learning_rate)
    
    def _calculate_loss(self, predictions, targets):
        """Calculate mean squared error loss"""
        return np.mean((predictions - targets) ** 2)
    
    def _extract_features(self, record):
        """Extract features from a database record with multi-layer support"""
        geometry_code = self.geometry_mapping.get(record['geometry_type'], 0)
        
        # Multi-layer features
        total_thickness = sum(layer['thickness'] for layer in record['insulation_layers'])
        avg_density = sum(layer['density'] for layer in record['insulation_layers']) / len(record['insulation_layers'])
        
        # Weighted average thermal conductivity
        total_resistance = sum(layer['thickness'] / (layer['thermal_conductivity'] or 0.04) 
                             for layer in record['insulation_layers'])
        effective_k = total_thickness / total_resistance if total_resistance > 0 else 0.04
        
        # Material composition (simplified to primary material)
        primary_material = record['insulation_layers'][0]['material_name'] if record['insulation_layers'] else 'unknown'
        material_code = self.material_mapping.get(primary_material.lower(), 0)
        
        # Geometric features
        surface_area = record.get('surface_area', 0)
        
        # Temperature-related features
        equipment_temp = record['equipment_surface_temp']
        temp_ratio = record['insulation_surface_temp'] / equipment_temp if equipment_temp > 0 else 0
        
        return [
            equipment_temp,
            surface_area,
            total_thickness,
            avg_density,
            effective_k,
            geometry_code,
            material_code,
            len(record['insulation_layers']),  # Number of layers
            temp_ratio  # Temperature efficiency indicator
        ]
    
    def _split_data(self, X, y):
        """Split data into training and validation sets"""
        n_samples = len(X)
        n_val = int(n_samples * self.validation_split)
        
        # Shuffle indices
        indices = np.random.permutation(n_samples)
        
        train_indices = indices[n_val:]
        val_indices = indices[:n_val]
        
        X_train = np.array([X[i] for i in train_indices])
        y_train = np.array([y[i] for i in train_indices])
        X_val = np.array([X[i] for i in val_indices])
        y_val = np.array([y[i] for i in val_indices])
        
        return X_train, y_train, X_val, y_val
    
    def train_model(self, data_list):
        """Train neural network models"""
        if len(data_list) < 5:
            print("Not enough data for training. At least 5 samples needed.")
            return False
        
        print(f"Preparing training data from {len(data_list)} records...")
        
        # Extract features and targets
        features_list = []
        temp_targets = []
        hconv_targets = []
        
        for record in data_list:
            if not record['insulation_layers']:
                continue
                
            features = self._extract_features(record)
            features_list.append(features)
            temp_targets.append(record['insulation_surface_temp'])
            hconv_targets.append(record.get('calculated_h_conv', record['convection_coefficient']))
        
        if len(features_list) < 5:
            print("Not enough valid records for training.")
            return False
        
        # Convert to numpy arrays
        X = np.array(features_list)
        y_temp = np.array(temp_targets).reshape(-1, 1)
        y_hconv = np.array(hconv_targets).reshape(-1, 1)
        
        print(f"Training with {len(X)} valid samples...")
        print(f"Feature dimensions: {X.shape[1]}")
        
        # Normalize features
        X_norm, self.feature_scaler_params = self._normalize_features(X)
        y_temp_norm, self.temp_target_scaler_params = self._normalize_features(y_temp)
        y_hconv_norm, self.hconv_target_scaler_params = self._normalize_features(y_hconv)
        
        # Split data
        X_train, y_temp_train, X_val, y_temp_val = self._split_data(X_norm, y_temp_norm)
        _, y_hconv_train, _, y_hconv_val = self._split_data(X_norm, y_hconv_norm)
        
        # Create networks
        input_size = X.shape[1]
        self.temp_network = self._create_network(input_size)
        self.hconv_network = self._create_network(input_size)
        
        print(f"Neural Network Architecture:")
        print(f"  Input size: {input_size}")
        print(f"  Hidden layers: {self.hidden_layers}")
        print(f"  Output size: 1")
        print(f"  Total parameters (approx): {self._count_parameters()}")
        
        # Train temperature prediction network
        print("\nTraining temperature prediction network...")
        self._train_network(self.temp_network, X_train, y_temp_train, X_val, y_temp_val, 'temp')
        
        # Train convection coefficient network
        print("\nTraining convection coefficient prediction network...")
        self._train_network(self.hconv_network, X_train, y_hconv_train, X_val, y_hconv_val, 'hconv')
        
        self.is_trained = True
        print(f"\nNeural networks trained successfully!")
        self._print_training_summary()
        
        return True
    
    def _train_network(self, network, X_train, y_train, X_val, y_val, network_name):
        """Train a single network"""
        n_batches = max(1, len(X_train) // self.batch_size)
        
        for epoch in range(self.epochs):
            # Shuffle training data
            indices = np.random.permutation(len(X_train))
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_loss = 0
            
            # Mini-batch training
            for i in range(n_batches):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, len(X_train))
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward pass
                predictions = self._forward_pass(network, X_batch)
                
                # Backward pass
                self._backward_pass(network, y_batch)
                
                # Calculate batch loss
                batch_loss = self._calculate_loss(predictions, y_batch)
                epoch_loss += batch_loss
            
            # Calculate average epoch loss
            epoch_loss /= n_batches
            
            # Validation loss
            val_predictions = self._forward_pass(network, X_val)
            val_loss = self._calculate_loss(val_predictions, y_val)
            
            # Store training history
            self.training_history[f'{network_name}_loss'].append(epoch_loss)
            self.training_history[f'{network_name}_val_loss'].append(val_loss)
            
            # Print progress
            if (epoch + 1) % (self.epochs // 10) == 0 or epoch == 0:
                print(f"  Epoch {epoch + 1}/{self.epochs} - Loss: {epoch_loss:.6f} - Val Loss: {val_loss:.6f}")
    
    def _count_parameters(self):
        """Count approximate number of parameters in the network"""
        if not self.hidden_layers:
            return 0
        
        total_params = 0
        
        # First layer
        total_params += (9 * self.hidden_layers[0]) + self.hidden_layers[0]  # weights + biases
        
        # Hidden layers
        for i in range(1, len(self.hidden_layers)):
            total_params += (self.hidden_layers[i-1] * self.hidden_layers[i]) + self.hidden_layers[i]
        
        # Output layer
        total_params += self.hidden_layers[-1] + 1
        
        return total_params * 2  # Two networks
    
    def predict_temperature(self, equipment_temp, surface_area, insulation_layers, 
                          convection_coefficient, geometry_type):
        """Predict insulation surface temperature using neural network"""
        
        if not self.is_trained:
            raise ValueError("Model has not been trained yet.")
        
        # Extract features
        geometry_code = self.geometry_mapping.get(geometry_type, 0)
        total_thickness = sum(layer.thickness for layer in insulation_layers)
        avg_density = sum(layer.density for layer in insulation_layers) / len(insulation_layers)
        
        # Calculate effective thermal conductivity
        total_resistance = sum(layer.thickness / (layer.thermal_conductivity or 0.04) 
                             for layer in insulation_layers)
        effective_k = total_thickness / total_resistance if total_resistance > 0 else 0.04
        
        primary_material = insulation_layers[0].name.lower() if insulation_layers else 'unknown'
        material_code = self.material_mapping.get(primary_material, 0)
        
        # For prediction, we need to estimate temperature ratio
        temp_ratio = 0.7  # Initial estimate
        
        features = np.array([[
            equipment_temp, surface_area, total_thickness, avg_density, effective_k,
            geometry_code, material_code, len(insulation_layers), temp_ratio
        ]])
        
        # Normalize features
        features_norm, _ = self._normalize_features(features, self.feature_scaler_params)
        
        # Predict normalized temperature
        temp_norm = self._forward_pass(self.temp_network, features_norm)
        
        # Denormalize prediction
        temp_prediction = self._denormalize_target(temp_norm, self.temp_target_scaler_params)
        
        return float(temp_prediction[0, 0])
    
    def predict_convection_coefficient(self, equipment_temp, insulation_temp, 
                                     surface_area, insulation_layers, geometry_type):
        """Predict convection coefficient using neural network"""
        
        if not self.is_trained:
            raise ValueError("Model has not been trained yet.")
        
        # Extract features
        geometry_code = self.geometry_mapping.get(geometry_type, 0)
        total_thickness = sum(layer.thickness for layer in insulation_layers)
        avg_density = sum(layer.density for layer in insulation_layers) / len(insulation_layers)
        
        # Calculate effective thermal conductivity
        total_resistance = sum(layer.thickness / (layer.thermal_conductivity or 0.04) 
                             for layer in insulation_layers)
        effective_k = total_thickness / total_resistance if total_resistance > 0 else 0.04
        
        primary_material = insulation_layers[0].name.lower() if insulation_layers else 'unknown'
        material_code = self.material_mapping.get(primary_material, 0)
        
        temp_ratio = insulation_temp / equipment_temp if equipment_temp > 0 else 0
        
        features = np.array([[
            equipment_temp, surface_area, total_thickness, avg_density, effective_k,
            geometry_code, material_code, len(insulation_layers), temp_ratio
        ]])
        
        # Normalize features
        features_norm, _ = self._normalize_features(features, self.feature_scaler_params)
        
        # Predict normalized convection coefficient
        hconv_norm = self._forward_pass(self.hconv_network, features_norm)
        
        # Denormalize prediction
        hconv_prediction = self._denormalize_target(hconv_norm, self.hconv_target_scaler_params)
        
        return float(hconv_prediction[0, 0])
    
    def _print_training_summary(self):
        """Print training summary and final metrics"""
        print(f"\n=== Neural Network Training Summary ===")
        print(f"Architecture: {9} -> {' -> '.join(map(str, self.hidden_layers))} -> 1")
        print(f"Training epochs: {self.epochs}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Batch size: {self.batch_size}")
        print(f"Validation split: {self.validation_split}")
        
        if self.training_history['temp_loss']:
            final_temp_loss = self.training_history['temp_loss'][-1]
            final_temp_val_loss = self.training_history['temp_val_loss'][-1]
            print(f"\nTemperature Network:")
            print(f"  Final training loss: {final_temp_loss:.6f}")
            print(f"  Final validation loss: {final_temp_val_loss:.6f}")
        
        if self.training_history['hconv_loss']:
            final_hconv_loss = self.training_history['hconv_loss'][-1]
            final_hconv_val_loss = self.training_history['hconv_val_loss'][-1]
            print(f"\nConvection Coefficient Network:")
            print(f"  Final training loss: {final_hconv_loss:.6f}")
            print(f"  Final validation loss: {final_hconv_val_loss:.6f}")
    
    def print_network_info(self):
        """Print information about the trained networks"""
        if not self.is_trained:
            print("Neural networks have not been trained yet.")
            return
        
        print(f"\n=== Neural Network Model Information ===")
        print(f"Status: Trained")
        print(f"Architecture: Input({9}) -> Hidden{self.hidden_layers} -> Output(1)")
        print(f"Total parameters: ~{self._count_parameters()}")
        print(f"Activation functions: ReLU (hidden), Linear (output)")
        print(f"Training configuration:")
        print(f"  Epochs: {self.epochs}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Validation split: {self.validation_split}")
        
        # Network details
        if self.temp_network:
            print(f"\nTemperature Prediction Network:")
            for i, layer in enumerate(self.temp_network):
                print(f"  Layer {i+1}: {layer.input_size} -> {layer.output_size} ({layer.activation})")
        
        if self.hconv_network:
            print(f"\nConvection Coefficient Network:")
            for i, layer in enumerate(self.hconv_network):
                print(f"  Layer {i+1}: {layer.input_size} -> {layer.output_size} ({layer.activation})")
    
    def save_model(self, filepath):
        """Save trained model to file"""
        if not self.is_trained:
            print("No trained model to save.")
            return False
        
        model_data = {
            'temp_network': self.temp_network,
            'hconv_network': self.hconv_network,
            'feature_scaler_params': self.feature_scaler_params,
            'temp_target_scaler_params': self.temp_target_scaler_params,
            'hconv_target_scaler_params': self.hconv_target_scaler_params,
            'training_history': self.training_history,
            'hidden_layers': self.hidden_layers,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'batch_size': self.batch_size
        }
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"Model saved to {filepath}")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath):
        """Load trained model from file"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.temp_network = model_data['temp_network']
            self.hconv_network = model_data['hconv_network']
            self.feature_scaler_params = model_data['feature_scaler_params']
            self.temp_target_scaler_params = model_data['temp_target_scaler_params']
            self.hconv_target_scaler_params = model_data['hconv_target_scaler_params']
            self.training_history = model_data['training_history']
            self.hidden_layers = model_data['hidden_layers']
            self.learning_rate = model_data['learning_rate']
            self.epochs = model_data['epochs']
            self.batch_size = model_data['batch_size']
            
            self.is_trained = True
            print(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

class EnhancedThermalAnalyzer:
    """Enhanced thermal analyzer with multi-layer insulation support and neural network prediction"""
    
    def __init__(self, db_path="thermal_data_enhanced.db"):
        self.database = ThermalDatabase(db_path)
        self.parser = HTMLParser()
        self.predictor = NeuralNetworkPredictor()
        self.heat_calculator = HeatTransferCalculator()
    
    def import_html_files(self, directory_path):
        """Import HTML files and extract training data with neural network training"""
        if not os.path.exists(directory_path):
            print(f"Directory {directory_path} does not exist.")
            return
        
        print(f"\n=== Importing HTML Files for Neural Network Training ===")
        print(f"Scanning directory: {directory_path}")
        
        imported_count = 0
        skipped_count = 0
        
        # Get list of HTML files
        html_files = [f for f in os.listdir(directory_path) if f.lower().endswith(('.html', '.htm'))]
        print(f"Found {len(html_files)} HTML files")
        
        if len(html_files) == 0:
            print("No HTML files found in the directory.")
            return
        
        for filename in html_files:
            file_path = os.path.join(directory_path, filename)
            print(f"\nProcessing: {filename}")
            
            # Parse HTML file
            data = self.parser.parse_html_file(file_path)
            
            if data and self._validate_extracted_data(data):
                try:
                    # Create equipment geometry from extracted data
                    equipment_geometry = self._create_geometry_from_data(data)
                    
                    # Create insulation layers from extracted data
                    insulation_layers = self._create_layers_from_data(data)
                    
                    # Calculate heat transfer properties
                    heat_flux = self.heat_calculator.calculate_multilayer_heat_flux(
                        data['equipment_temp'], data['insulation_temp'], insulation_layers)
                    
                    calculated_h_conv = self.heat_calculator.calculate_convection_coefficient(
                        data['insulation_temp'], 25, heat_flux)  # Assume 25°C ambient
                    
                    # Use provided convection coefficient if available, otherwise use calculated
                    convection_coeff = data.get('convection_coefficient', calculated_h_conv)
                    
                    # Create thermal data object
                    thermal_data = ThermalData(
                        equipment_geometry=equipment_geometry,
                        equipment_surface_temp=data['equipment_temp'],
                        insulation_surface_temp=data['insulation_temp'],
                        insulation_layers=insulation_layers,
                        convection_coefficient=convection_coeff
                    )
                    
                    # Save to database
                    self.database.insert_data(thermal_data, heat_flux, calculated_h_conv, data['source_file'])
                    imported_count += 1
                    
                    print(f"  ✓ Successfully imported")
                    print(f"    Equipment temp: {data['equipment_temp']:.1f}°C")
                    print(f"    Insulation temp: {data['insulation_temp']:.1f}°C")
                    print(f"    Geometry: {equipment_geometry.geometry_type}")
                    print(f"    Layers: {len(insulation_layers)}")
                    print(f"    Heat flux: {heat_flux:.2f} W/m²")
                    
                except Exception as e:
                    print(f"  ✗ Error processing data: {e}")
                    skipped_count += 1
            else:
                print(f"  ✗ Insufficient or invalid data")
                skipped_count += 1
        
        print(f"\n=== Import Summary ===")
        print(f"Successfully imported: {imported_count} files")
        print(f"Skipped: {skipped_count} files")
        print(f"Total processed: {len(html_files)} files")
        
        if imported_count > 0:
            # Ask if user wants to train the neural network immediately
            train_now = input(f"\nTrain neural network with {imported_count} imported samples? (y/n): ").lower().strip()
            if train_now == 'y':
                print("\nStarting neural network training with imported data...")
                self.train_prediction_model()
        else:
            print("No data was imported. Cannot train neural network.")
    
    def _validate_extracted_data(self, data):
        """Validate extracted data for completeness"""
        required_fields = ['equipment_temp', 'insulation_temp']
        
        # Check required fields
        for field in required_fields:
            if field not in data or data[field] is None:
                return False
            if not isinstance(data[field], (int, float)) or data[field] <= 0:
                return False
        
        # Check temperature logic
        if data['insulation_temp'] >= data['equipment_temp']:
            return False
        
        return True
    
    def _create_geometry_from_data(self, data):
        """Create equipment geometry from extracted data"""
        geometry_type = data.get('geometry_type', 'pipe')
        
        # Default parameters based on geometry type
        if geometry_type == 'pipe':
            parameters = {
                'diameter': data.get('diameter', 0.1),  # 10 cm default
                'length': data.get('length', 1.0)       # 1 m default
            }
        elif geometry_type == 'sphere':
            parameters = {
                'diameter': data.get('diameter', 0.1)   # 10 cm default
            }
        elif geometry_type == 'cube':
            parameters = {
                'side_length': data.get('side', 0.1)    # 10 cm default
            }
        elif geometry_type == 'surface':
            parameters = {
                'length': data.get('length', 1.0),      # 1 m default
                'width': data.get('width', 1.0)         # 1 m default
            }
        else:
            # Default to pipe
            parameters = {'diameter': 0.1, 'length': 1.0}
        
        return EquipmentGeometry(geometry_type, **parameters)
    
    def _create_layers_from_data(self, data):
        """Create insulation layers from extracted data"""
        insulation_type = data.get('insulation_type', 'cerablanket')
        thickness = data.get('thickness', 0.025)  # 25 mm default
        density = data.get('density')
        
        # Get material properties
        material_info = self.heat_calculator.insulation_materials.get(insulation_type)
        if not material_info:
            material_info = self.heat_calculator.insulation_materials['cerablanket']
        
        # Use provided density or default
        if not density:
            density = material_info['available_density'][0]
        
        # Get thermal conductivity
        thermal_conductivity = self.heat_calculator.get_thermal_conductivity(insulation_type, density)
        
        # Create single layer (can be extended for multi-layer parsing)
        layer = InsulationLayer(
            name=insulation_type,
            thickness=thickness,
            density=density,
            thermal_conductivity=thermal_conductivity
        )
        
        return [layer]
    
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
    
    def configure_neural_network(self):
        """Configure neural network parameters"""
        print("\n=== Neural Network Configuration ===")
        
        try:
            # Hidden layers configuration
            print("Current hidden layers: [20, 15, 10]")
            change_layers = input("Change hidden layer configuration? (y/n): ").lower().strip()
            
            if change_layers == 'y':
                layers_input = input("Enter hidden layer sizes (comma-separated, e.g., 25,20,15): ")
                if layers_input.strip():
                    hidden_layers = [int(x.strip()) for x in layers_input.split(',')]
                    self.predictor.hidden_layers = hidden_layers
                    print(f"Hidden layers set to: {hidden_layers}")
            
            # Learning rate
            current_lr = self.predictor.learning_rate
            lr_input = input(f"Learning rate (current: {current_lr}): ")
            if lr_input.strip():
                self.predictor.learning_rate = float(lr_input)
            
            # Epochs
            current_epochs = self.predictor.epochs
            epochs_input = input(f"Training epochs (current: {current_epochs}): ")
            if epochs_input.strip():
                self.predictor.epochs = int(epochs_input)
            
            # Batch size
            current_batch = self.predictor.batch_size
            batch_input = input(f"Batch size (current: {current_batch}): ")
            if batch_input.strip():
                self.predictor.batch_size = int(batch_input)
            
            print(f"\nFinal Neural Network Configuration:")
            print(f"  Hidden layers: {self.predictor.hidden_layers}")
            print(f"  Learning rate: {self.predictor.learning_rate}")
            print(f"  Epochs: {self.predictor.epochs}")
            print(f"  Batch size: {self.predictor.batch_size}")
            
        except ValueError as e:
            print(f"Invalid input: {e}")
            print("Using default configuration.")
    
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
            self.database.insert_data(thermal_data, heat_flux, calculated_h_conv, "manual_input")
            
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
        """Train neural network prediction model with configuration"""
        # Configure neural network parameters
        self.configure_neural_network()
        
        data_list = self.database.get_all_data()
        if len(data_list) == 0:
            print("No data available for training the model.")
            return False
        
        print(f"\nStarting neural network training...")
        success = self.predictor.train_model(data_list)
        if success:
            self.predictor.print_network_info()
            
            # Offer to save the model
            save_model = input("\nSave trained model? (y/n): ").lower().strip()
            if save_model == 'y':
                filename = input("Model filename (default: neural_model.pkl): ").strip()
                if not filename:
                    filename = "neural_model.pkl"
                self.predictor.save_model(filename)
        
        return success
    
    def predict_properties(self):
        """Predict thermal properties for new multi-layer configuration using neural network"""
        try:
            print("\n=== Multi-Layer Thermal Analysis Prediction (Neural Network) ===")
            
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
            
            # Predict insulation temperature using neural network
            surface_area = equipment_geometry.get_surface_area()
            predicted_insulation_temp = self.predictor.predict_temperature(
                equipment_temp, surface_area, insulation_layers,
                estimated_h_conv, equipment_geometry.geometry_type)
            
            # Calculate accurate heat flux and convection coefficient
            heat_flux = self.heat_calculator.calculate_multilayer_heat_flux(
                equipment_temp, predicted_insulation_temp, insulation_layers)
            
            accurate_h_conv = self.heat_calculator.calculate_convection_coefficient(
                predicted_insulation_temp, ambient_temp, heat_flux)
            
            # Predict convection coefficient using neural network
            predicted_h_conv = self.predictor.predict_convection_coefficient(
                equipment_temp, predicted_insulation_temp, surface_area,
                insulation_layers, equipment_geometry.geometry_type)
            
            # Calculate efficiency
            temp_reduction = equipment_temp - predicted_insulation_temp
            efficiency = (temp_reduction / equipment_temp) * 100
            
            # Display results
            print(f"\n=== Multi-Layer Thermal Analysis Results (Neural Network) ===")
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
            
            print(f"\nNeural Network Predictions:")
            print(f"  Equipment temperature: {equipment_temp:.1f} °C")
            print(f"  Predicted insulation surface temp: {predicted_insulation_temp:.1f} °C")
            print(f"  Ambient temperature: {ambient_temp:.1f} °C")
            print(f"  Heat flux: {heat_flux:.2f} W/m²")
            print(f"  Calculated h_conv: {accurate_h_conv:.2f} W/m²·K")
            print(f"  Predicted h_conv (NN): {predicted_h_conv:.2f} W/m²·K")
            print(f"  Temperature reduction: {temp_reduction:.1f} °C")
            print(f"  Insulation efficiency: {efficiency:.1f}%")
            
            return {
                'insulation_temp': predicted_insulation_temp,
                'heat_flux': heat_flux,
                'convection_coefficient': accurate_h_conv,
                'predicted_h_conv': predicted_h_conv,
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
        
        # Source file statistics
        source_files = [d.get('source_file', 'unknown') for d in data_list]
        html_files = [f for f in source_files if f.endswith('.html') or f.endswith('.htm')]
        manual_entries = [f for f in source_files if f == 'manual_input']
        
        print(f"\nData sources:")
        print(f"  HTML files: {len(html_files)} records")
        print(f"  Manual entries: {len(manual_entries)} records")
        if len(html_files) > 0:
            print(f"  Unique HTML files: {len(set(html_files))}")
        
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
    """Main enhanced program function with neural network support"""
    print("=== Enhanced Multi-Layer Thermal Insulation Analysis System ===")
    print("With Neural Network Prediction and HTML Training Data Import")
    
    analyzer = EnhancedThermalAnalyzer()
    
    while True:
        print("\n" + "="*80)
        print("Available options:")
        print("1. Import HTML files and train neural network")
        print("2. View insulation materials and options")
        print("3. Add manual multi-layer thermal data")
        print("4. Train neural network prediction model")
        print("5. Multi-layer thermal analysis prediction (Neural Network)")
        print("6. View enhanced data statistics")
        print("7. View neural network information")
        print("8. Load/Save neural network model")
        print("9. Exit")
        print("="*80)
        
        choice = input("\nYour choice (1-9): ").strip()
        
        if choice == '1':
            directory = input("Path to folder containing HTML files (default: ./html_files): ").strip()
            if not directory:
                directory = "./html_files"
            analyzer.import_html_files(directory)
        
        elif choice == '2':
            analyzer.show_insulation_materials()
        
        elif choice == '3':
            analyzer.add_manual_data_enhanced()
        
        elif choice == '4':
            print("\nStarting neural network model training...")
            analyzer.train_prediction_model()
        
        elif choice == '5':
            analyzer.predict_properties()
        
        elif choice == '6':
            analyzer.view_data_statistics()
        
        elif choice == '7':
            analyzer.predictor.print_network_info()
        
        elif choice == '8':
            print("\n1. Save current model")
            print("2. Load saved model")
            sub_choice = input("Choose option (1-2): ").strip()
            
            if sub_choice == '1':
                if analyzer.predictor.is_trained:
                    filename = input("Model filename (default: neural_model.pkl): ").strip()
                    if not filename:
                        filename = "neural_model.pkl"
                    analyzer.predictor.save_model(filename)
                else:
                    print("No trained model to save.")
            
            elif sub_choice == '2':
                filename = input("Model filename (default: neural_model.pkl): ").strip()
                if not filename:
                    filename = "neural_model.pkl"
                analyzer.predictor.load_model(filename)
        
        elif choice == '9':
            print("Enhanced multi-layer thermal analysis system with neural network closed. Good luck!")
            break
        
        else:
            print("Invalid option. Please enter a number between 1 and 9.")

if __name__ == "__main__":
    main()