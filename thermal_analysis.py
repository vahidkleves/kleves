#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thermal Insulation Analysis System - English Version
Simple Thermal Insulation Analysis System with BeautifulSoup HTML Parser
"""

import sqlite3
import json
import os
from datetime import datetime
import pickle
import re
from bs4 import BeautifulSoup

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
    """Class for processing HTML output files from software using BeautifulSoup"""
    
    def __init__(self):
        # Field mappings from English to Persian and variations
        self.field_mappings = {
            'Inner Surface Temperature': ['inner_surface_temp', 'equipment_surface_temp'],
            'دمای سطح داخلی': ['inner_surface_temp', 'equipment_surface_temp'],
            'Ambient Temperature': ['ambient_temp', 'insulation_surface_temp'],
            'دمای محیط': ['ambient_temp', 'insulation_surface_temp'],
            'Maximum Surface Temperature': ['max_surface_temp', 'insulation_surface_temp'],
            'حداکثر دمای سطح': ['max_surface_temp', 'insulation_surface_temp'],
            'Air Speed': ['air_speed', 'convection_coefficient'],
            'سرعت هوا': ['air_speed', 'convection_coefficient'],
            'Maximum Insulation Thickness': ['max_insulation_thickness', 'thickness'],
            'حداکثر ضخامت عایق': ['max_insulation_thickness', 'thickness'],
            'Maximum Number of Layers': ['max_layers'],
            'حداکثر تعداد لایه': ['max_layers'],
        }
        
        # Geometry type detection patterns
        self.geometry_patterns = {
            'pipe': ['pipe', 'cylinder', 'لوله', 'استوانه'],
            'sphere': ['sphere', 'ball', 'کره', 'گوی'],
            'cube': ['cube', 'box', 'مکعب', 'جعبه'],
            'surface': ['surface', 'flat', 'plate', 'سطح', 'صفحه']
        }
        
        # Insulation type patterns
        self.insulation_patterns = {
            'polyurethane': ['polyurethane', 'pu', 'پلی اورتان', 'پلی‌اورتان'],
            'foam': ['foam', 'فوم'],
            'glass wool': ['glass wool', 'glasswool', 'پشم شیشه', 'پشم‌شیشه'],
            'mineral wool': ['mineral wool', 'پشم معدنی'],
            'ceramic': ['ceramic', 'سرامیک']
        }
    
    def parse_html_file(self, file_path):
        """Process an HTML file and extract thermal information from Model Summary section"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            soup = BeautifulSoup(content, 'html.parser')
            
            # Find Model Summary section
            extracted_data = self._extract_model_summary(soup)
            
            # Also try to extract additional information from the entire document
            additional_data = self._extract_additional_info(soup)
            
            # Merge the data
            extracted_data.update(additional_data)
            
            return self._process_extracted_data(extracted_data)
            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return None
    
    def _extract_model_summary(self, soup):
        """Extract data specifically from Model Summary section"""
        extracted_data = {}
        
        # Try to find Model Summary section by different methods
        model_summary_section = None
        
        # Method 1: Look for heading containing "Model Summary"
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        for heading in headings:
            if heading.text and ('model summary' in heading.text.lower() or 'خلاصه مدل' in heading.text):
                model_summary_section = heading.find_next_sibling()
                break
        
        # Method 2: Look for div with class containing "summary" or similar
        if not model_summary_section:
            model_summary_section = soup.find('div', class_=re.compile(r'summary|model', re.I))
        
        # Method 3: Look for table with Model Summary data pattern
        if not model_summary_section:
            tables = soup.find_all('table')
            for table in tables:
                if self._contains_thermal_data(table):
                    model_summary_section = table
                    break
        
        # Method 4: If still not found, search the entire document
        if not model_summary_section:
            model_summary_section = soup
        
        # Extract data from the identified section
        if model_summary_section:
            extracted_data = self._extract_from_section(model_summary_section)
        
        return extracted_data
    
    def _contains_thermal_data(self, element):
        """Check if an element contains thermal data patterns"""
        text = element.get_text().lower()
        thermal_keywords = ['temperature', 'دما', 'insulation', 'عایق', 'thickness', 'ضخامت', 'surface', 'سطح']
        return any(keyword in text for keyword in thermal_keywords)
    
    def _extract_from_section(self, section):
        """Extract thermal data from a specific section"""
        extracted_data = {}
        
        # Try to find table data first
        tables = section.find_all('table') if hasattr(section, 'find_all') else []
        if tables:
            for table in tables:
                table_data = self._extract_from_table(table)
                extracted_data.update(table_data)
        
        # Extract from general text content
        text_data = self._extract_from_text(section.get_text() if hasattr(section, 'get_text') else str(section))
        extracted_data.update(text_data)
        
        return extracted_data
    
    def _extract_from_table(self, table):
        """Extract data from HTML table"""
        extracted_data = {}
        
        rows = table.find_all('tr')
        for row in rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) >= 2:
                # Assume first cell is label, second is value, third might be unit
                label = cells[0].get_text().strip()
                value_text = cells[1].get_text().strip()
                unit = cells[2].get_text().strip() if len(cells) > 2 else ""
                
                # Try to parse the value
                value = self._parse_numeric_value(value_text)
                if value is not None:
                    field_name = self._map_field_name(label)
                    if field_name:
                        extracted_data[field_name] = {
                            'value': value,
                            'unit': unit,
                            'original_label': label
                        }
        
        return extracted_data
    
    def _extract_from_text(self, text):
        """Extract data from plain text using regex patterns"""
        extracted_data = {}
        
        # Temperature patterns - more comprehensive
        temp_patterns = [
            r'(?:inner\s+surface\s+temperature|دمای\s+سطح\s+داخلی)[\s:]*(\d+\.?\d*)\s*(?:°?[CF]|درجه)',
            r'(?:ambient\s+temperature|دمای\s+محیط)[\s:]*(\d+\.?\d*)\s*(?:°?[CF]|درجه)',
            r'(?:maximum\s+surface\s+temperature|حداکثر\s+دمای\s+سطح)[\s:]*(\d+\.?\d*)\s*(?:°?[CF]|درجه)',
            r'(?:دمای\s+سطح\s+تجهیز)[\s:]*(\d+\.?\d*)\s*(?:°?[CF]|درجه)',
            r'(?:دمای\s+سطح\s+عایق)[\s:]*(\d+\.?\d*)\s*(?:°?[CF]|درجه)',
            r'(?:equipment\s+surface\s+temperature)[\s:]*(\d+\.?\d*)\s*(?:°?[CF]|درجه)',
            r'(?:insulation\s+surface\s+temperature)[\s:]*(\d+\.?\d*)\s*(?:°?[CF]|درجه)'
        ]
        
        for pattern in temp_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                temp_value = float(matches[0])
                if 'inner' in pattern or 'داخلی' in pattern or 'تجهیز' in pattern or 'equipment' in pattern:
                    extracted_data['equipment_surface_temp'] = {'value': temp_value, 'unit': '°C'}
                elif 'ambient' in pattern or 'محیط' in pattern:
                    extracted_data['ambient_temp'] = {'value': temp_value, 'unit': '°C'}
                elif 'maximum' in pattern or 'حداکثر' in pattern or 'عایق' in pattern or 'insulation' in pattern:
                    extracted_data['insulation_surface_temp'] = {'value': temp_value, 'unit': '°C'}
        
        # Air speed / convection coefficient patterns
        air_patterns = [
            r'(?:air\s+speed|سرعت\s+هوا)[\s:]*(\d+\.?\d*)\s*(?:m/s|متر\s+بر\s+ثانیه)',
            r'(?:convection\s+coefficient|ضریب\s+همرفت)[\s:]*(\d+\.?\d*)\s*(?:W/m²\.K|W/m2\.K)'
        ]
        
        for pattern in air_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                value = float(matches[0])
                if 'speed' in pattern or 'سرعت' in pattern:
                    # Convert air speed to approximate convection coefficient
                    # h ≈ 10.45 - v + 10√v (for air)
                    v = value
                    h = 10.45 - v + 10 * (v ** 0.5)
                    extracted_data['convection_coefficient'] = {'value': h, 'unit': 'W/m².K'}
                else:
                    extracted_data['convection_coefficient'] = {'value': value, 'unit': 'W/m².K'}
        
        # Thickness patterns
        thickness_patterns = [
            r'(?:thickness|ضخامت)[\s:]*(\d+\.?\d*)\s*(?:mm|cm|m|میلی\s*متر|سانتی\s*متر|متر)',
            r'(?:insulation\s+thickness|ضخامت\s+عایق)[\s:]*(\d+\.?\d*)\s*(?:mm|cm|m)'
        ]
        
        for pattern in thickness_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                extracted_data['thickness'] = {'value': float(matches[0]), 'unit': 'mm'}
        
        # Geometry type detection
        for geom_type, patterns in self.geometry_patterns.items():
            for pattern in patterns:
                if pattern in text.lower():
                    extracted_data['geometry_type'] = geom_type
                    break
        
        # Insulation type detection
        for ins_type, patterns in self.insulation_patterns.items():
            for pattern in patterns:
                if pattern in text.lower():
                    extracted_data['insulation_type'] = ins_type
                    break
        
        return extracted_data
    
    def _extract_additional_info(self, soup):
        """Extract additional information from the entire document"""
        additional_data = {}
        
        # Look for any additional tables or data sections
        all_text = soup.get_text()
        
        # Try to find cross-sectional area
        area_patterns = [
            r'(?:cross[\s-]*section(?:al)?\s+area|سطح\s+مقطع)[\s:]*(\d+\.?\d*)\s*(?:m²|m2|square\s*meter)',
            r'(?:area|مساحت)[\s:]*(\d+\.?\d*)\s*(?:m²|m2)'
        ]
        
        for pattern in area_patterns:
            matches = re.findall(pattern, all_text, re.IGNORECASE)
            if matches:
                additional_data['cross_section_area'] = {'value': float(matches[0]), 'unit': 'm²'}
                break
        
        return additional_data
    
    def _map_field_name(self, label):
        """Map field label to internal field name"""
        label_lower = label.lower().strip()
        for key, field_names in self.field_mappings.items():
            if key.lower() in label_lower or any(alt in label_lower for alt in key.split()):
                return field_names[0]
        return None
    
    def _parse_numeric_value(self, value_text):
        """Parse numeric value from text"""
        # Remove common non-numeric characters
        cleaned = re.sub(r'[^\d.-]', '', value_text)
        try:
            return float(cleaned)
        except ValueError:
            return None
    
    def _process_extracted_data(self, raw_data):
        """Process and clean extracted data"""
        processed = {}
        
        # Process temperature data
        if 'equipment_surface_temp' in raw_data:
            processed['equipment_temp'] = raw_data['equipment_surface_temp']['value']
        
        if 'insulation_surface_temp' in raw_data:
            processed['insulation_temp'] = raw_data['insulation_surface_temp']['value']
        elif 'ambient_temp' in raw_data:
            processed['insulation_temp'] = raw_data['ambient_temp']['value']
        
        # Process geometry type
        if 'geometry_type' in raw_data:
            processed['geometry_type'] = raw_data['geometry_type']
        else:
            processed['geometry_type'] = 'surface'  # default
        
        # Process insulation type
        if 'insulation_type' in raw_data:
            processed['insulation_type'] = raw_data['insulation_type']
        else:
            processed['insulation_type'] = 'polyurethane'  # default
        
        # Process cross-section area
        if 'cross_section_area' in raw_data:
            processed['cross_section_area'] = raw_data['cross_section_area']['value']
        else:
            processed['cross_section_area'] = 1.0  # default 1 m²
        
        # Process convection coefficient
        if 'convection_coefficient' in raw_data:
            processed['convection_coefficient'] = raw_data['convection_coefficient']['value']
        else:
            processed['convection_coefficient'] = 25.0  # default for air
        
        # Process thickness
        if 'thickness' in raw_data:
            processed['thickness'] = raw_data['thickness']['value']
        
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
            'glass wool': 3, 'پشم شیشه': 3,
            'mineral wool': 4, 'پشم معدنی': 4,
            'ceramic': 5, 'سرامیک': 5
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
            if filename.lower().endswith(('.html', '.htm')):
                file_path = os.path.join(directory_path, filename)
                print(f"Processing file: {filename}")
                
                data = self.parser.parse_html_file(file_path)
                
                if data and self._validate_data(data):
                    thermal_data = ThermalData(
                        geometry_type=data.get('geometry_type', 'surface'),
                        equipment_surface_temp=data.get('equipment_temp', 0),
                        insulation_surface_temp=data.get('insulation_temp', 0),
                        insulation_type=data.get('insulation_type', 'polyurethane'),
                        cross_section_area=data.get('cross_section_area', 1.0),
                        convection_coefficient=data.get('convection_coefficient', 25.0),
                        thickness=data.get('thickness')
                    )
                    
                    self.database.insert_data(thermal_data)
                    imported_count += 1
                    print(f"✓ File {filename} successfully imported.")
                    print(f"  - Equipment temp: {data.get('equipment_temp', 'N/A')}°C")
                    print(f"  - Insulation temp: {data.get('insulation_temp', 'N/A')}°C")
                    print(f"  - Geometry: {data.get('geometry_type', 'N/A')}")
                    print(f"  - Insulation: {data.get('insulation_type', 'N/A')}")
                else:
                    print(f"✗ File {filename} does not have sufficient data or failed validation.")
                    if data:
                        print(f"  - Available data: {list(data.keys())}")
        
        print(f"\nTotal {imported_count} files successfully imported.")
    
    def _validate_data(self, data):
        """Validate extracted data"""
        required_fields = ['equipment_temp', 'insulation_temp']
        missing_fields = [field for field in required_fields if field not in data or data[field] is None]
        
        if missing_fields:
            print(f"  - Missing required fields: {missing_fields}")
            return False
        
        return True
    
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
    print("Enhanced version with BeautifulSoup HTML parser")
    print("Supports Model Summary extraction from HTML files")
    
    # Check if BeautifulSoup is available
    try:
        from bs4 import BeautifulSoup
        print("✓ BeautifulSoup library is available")
    except ImportError:
        print("✗ BeautifulSoup library is not installed!")
        print("Please install it using: pip install beautifulsoup4")
        return
    
    analyzer = ThermalAnalyzer()
    
    while True:
        print("\n" + "="*50)
        print("Available options:")
        print("1. Import HTML files (with Model Summary extraction)")
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
                ins_type = input("Insulation type (polyurethane/foam/glass wool/mineral wool/ceramic): ").strip()
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
                ins_type = input("Insulation type (polyurethane/foam/glass wool/mineral wool/ceramic): ").strip()
                
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
                
                # Show recent records
                print(f"\nLast 5 records:")
                for i, record in enumerate(data_list[-5:], 1):
                    print(f"{i}. {record['geometry_type']} - {record['insulation_type']}")
                    print(f"   Equipment: {record['equipment_surface_temp']}°C -> Insulation: {record['insulation_surface_temp']}°C")
            else:
                print("No data available.")
        
        elif choice == '6':
            print("Program closed. Good luck!")
            break
        
        else:
            print("Invalid option. Please enter a number between 1 and 6.")

if __name__ == "__main__":
    main()