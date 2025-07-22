#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thermal Insulation Analysis System - English Version
Simple Thermal Insulation Analysis System
"""

import sqlite3
import json
import os
from datetime import datetime
import pickle
import re

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

class HTMLParser:
    """Class for processing HTML output files from software"""
    
    def __init__(self):
        self.supported_patterns = {
            'temperature': r'(\d+\.?\d*)\s*°?[CF]',
            'geometry': r'(pipe|sphere|cube|surface|لوله|کره|مکعب|سطح)',
            'insulation': r'(polyurethane|foam|glass\s*wool|پلی\s*اورتان|فوم|پشم\s*شیشه)',
            'area': r'(\d+\.?\d*)\s*(m²|square\s*meter)',
            'coefficient': r'(\d+\.?\d*)\s*(W/m²\.K|W/m2\.K)'
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
        
        return processed

class ThermalDatabase:
    """Class for managing thermal database"""
    
    def __init__(self, db_path="thermal_data.db"):
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
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def insert_data(self, thermal_data):
        """Insert new data into database"""
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

class SimplePredictor:
    """Simple class for prediction without using sklearn"""
    
    def __init__(self):
        self.is_trained = False
        self.training_data = []
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
    
    def train_model(self, data_list):
        """Train simple prediction model"""
        if len(data_list) < 3:
            print("Not enough data for training. At least 3 samples needed.")
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
        print(f"Model trained with {len(self.training_data)} samples.")
        return True
    
    def predict(self, equipment_temp, cross_section_area, convection_coefficient, geometry_type, insulation_type):
        """Predict insulation surface temperature using weighted average"""
        
        if not self.is_trained:
            raise ValueError("Model has not been trained yet.")
        
        geometry_code = self.geometry_mapping.get(geometry_type, 0)
        insulation_code = self.insulation_mapping.get(insulation_type, 0)
        
        input_features = [equipment_temp, cross_section_area, convection_coefficient, geometry_code, insulation_code]
        
        # Calculate distance to each training sample
        weights = []
        targets = []
        
        for sample in self.training_data:
            distance = 0
            for i, feature in enumerate(input_features):
                distance += (feature - sample['features'][i]) ** 2
            
            distance = distance ** 0.5
            
            # Inverse distance weight
            weight = 1 / (distance + 0.001)
            
            weights.append(weight)
            targets.append(sample['target'])
        
        # Calculate weighted average
        total_weight = sum(weights)
        weighted_prediction = sum(w * t for w, t in zip(weights, targets)) / total_weight
        
        return weighted_prediction

class ThermalAnalyzer:
    """Main class for thermal analysis"""
    
    def __init__(self, db_path="thermal_data.db"):
        self.database = ThermalDatabase(db_path)
        self.parser = HTMLParser()
        self.predictor = SimplePredictor()
    
    def import_html_files(self, directory_path):
        """Import HTML files from a folder"""
        if not os.path.exists(directory_path):
            print(f"Directory {directory_path} does not exist.")
            return
        
        imported_count = 0
        
        for filename in os.listdir(directory_path):
            if filename.lower().endswith('.html'):
                file_path = os.path.join(directory_path, filename)
                data = self.parser.parse_html_file(file_path)
                
                if data and self._validate_data(data):
                    thermal_data = ThermalData(
                        geometry_type=data.get('geometry_type', 'unknown'),
                        equipment_surface_temp=data.get('equipment_temp', 0),
                        insulation_surface_temp=data.get('insulation_temp', 0),
                        insulation_type=data.get('insulation_type', 'unknown'),
                        cross_section_area=data.get('cross_section_area', 0),
                        convection_coefficient=data.get('convection_coefficient', 0)
                    )
                    
                    self.database.insert_data(thermal_data)
                    imported_count += 1
                    print(f"File {filename} successfully imported.")
                else:
                    print(f"File {filename} does not have sufficient data.")
        
        print(f"Total {imported_count} files successfully imported.")
    
    def _validate_data(self, data):
        """Validate extracted data"""
        required_fields = ['equipment_temp', 'insulation_temp', 'cross_section_area']
        return all(field in data and data[field] is not None for field in required_fields)
    
    def train_prediction_model(self):
        """Train prediction model with existing data"""
        data_list = self.database.get_all_data()
        if len(data_list) == 0:
            print("No data available for training the model.")
            return False
        
        return self.predictor.train_model(data_list)
    
    def predict_insulation_temperature(self, equipment_temp, cross_section_area, convection_coefficient, geometry_type, insulation_type):
        """Predict insulation surface temperature for new geometry"""
        try:
            prediction = self.predictor.predict(
                equipment_temp, cross_section_area, convection_coefficient,
                geometry_type, insulation_type
            )
            return prediction
        except Exception as e:
            print(f"Prediction error: {e}")
            return None
    
    def add_manual_data(self, geometry_type, equipment_temp, insulation_temp, insulation_type, cross_section_area, convection_coefficient):
        """Add thermal data manually"""
        thermal_data = ThermalData(
            geometry_type=geometry_type,
            equipment_surface_temp=equipment_temp,
            insulation_surface_temp=insulation_temp,
            insulation_type=insulation_type,
            cross_section_area=cross_section_area,
            convection_coefficient=convection_coefficient
        )
        
        self.database.insert_data(thermal_data)
        print("New data successfully added.")

def main():
    """Main program function"""
    print("=== Thermal Insulation Analysis System ===")
    print("Simple version without external dependencies")
    
    analyzer = ThermalAnalyzer()
    
    while True:
        print("\n" + "="*50)
        print("Available options:")
        print("1. Import HTML files")
        print("2. Add manual data")
        print("3. Train prediction model")
        print("4. Predict insulation temperature for new geometry")
        print("5. View data statistics")
        print("6. Exit")
        print("="*50)
        
        choice = input("\nYour choice (1-6): ").strip()
        
        if choice == '1':
            directory = input("Path to folder containing HTML files (default: ./html_files): ").strip()
            if not directory:
                directory = "./html_files"
            analyzer.import_html_files(directory)
        
        elif choice == '2':
            try:
                print("\nAdding new data:")
                geometry = input("Geometry type (pipe/sphere/cube/surface): ").strip()
                eq_temp = float(input("Equipment surface temperature (°C): "))
                ins_temp = float(input("Insulation surface temperature (°C): "))
                ins_type = input("Insulation type (polyurethane/foam/glass wool): ").strip()
                area = float(input("Cross-section area (m²): "))
                coeff = float(input("Heat transfer coefficient (W/m².K): "))
                
                analyzer.add_manual_data(geometry, eq_temp, ins_temp, ins_type, area, coeff)
            except ValueError:
                print("Error: Please enter valid numerical values.")
        
        elif choice == '3':
            print("\nStarting model training...")
            analyzer.train_prediction_model()
        
        elif choice == '4':
            try:
                print("\nPredicting insulation temperature:")
                geometry = input("Geometry type (pipe/sphere/cube/surface): ").strip()
                eq_temp = float(input("Equipment surface temperature (°C): "))
                area = float(input("Cross-section area (m²): "))
                coeff = float(input("Heat transfer coefficient (W/m².K): "))
                ins_type = input("Insulation type (polyurethane/foam/glass wool): ").strip()
                
                prediction = analyzer.predict_insulation_temperature(
                    eq_temp, area, coeff, geometry, ins_type
                )
                
                if prediction is not None:
                    print(f"\n*** Predicted insulation surface temperature: {prediction:.1f} °C ***")
                    
                    # Calculate temperature reduction
                    temp_reduction = eq_temp - prediction
                    print(f"Temperature reduction after insulation: {temp_reduction:.1f} °C")
                    print(f"Temperature reduction percentage: {(temp_reduction/eq_temp)*100:.1f}%")
                
            except ValueError:
                print("Error: Please enter valid numerical values.")
        
        elif choice == '5':
            data_list = analyzer.database.get_all_data()
            if len(data_list) > 0:
                print(f"\nData statistics:")
                print(f"Total records: {len(data_list)}")
                
                geometries = list(set([d['geometry_type'] for d in data_list]))
                insulations = list(set([d['insulation_type'] for d in data_list]))
                eq_temps = [d['equipment_surface_temp'] for d in data_list]
                ins_temps = [d['insulation_surface_temp'] for d in data_list]
                
                print(f"Available geometries: {geometries}")
                print(f"Available insulation types: {insulations}")
                print(f"Equipment temperature range: {min(eq_temps):.1f} to {max(eq_temps):.1f} °C")
                print(f"Insulation temperature range: {min(ins_temps):.1f} to {max(ins_temps):.1f} °C")
            else:
                print("No data available.")
        
        elif choice == '6':
            print("Program closed. Good luck!")
            break
        
        else:
            print("Invalid option. Please enter a number between 1 and 6.")

if __name__ == "__main__":
    main()
