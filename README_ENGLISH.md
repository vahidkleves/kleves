# Thermal Insulation Analysis System
# Advanced Heat Transfer and Thermal Insulation Analysis Tool

## 🔥 Project Overview

This project is a comprehensive and advanced system for thermal heat transfer analysis and thermal insulation optimization that provides the following unique capabilities:

### 🎯 Main Objective
Create an intelligent tool that can:
- Process existing data from thermal analysis software
- Build machine learning models for predicting thermal properties
- Provide accurate predictions for complex and new geometries

## 🚀 Key Features

### 1. 📄 HTML File Processing
- **Automatic extraction** of thermal information from HTML reports
- **Intelligent recognition** of geometry types (pipe, sphere, cube, surface)
- **Automatic detection** of insulation types and thermal parameters
- **Support** for various Persian and English formats

### 2. 🗄️ Data Management
- **SQLite database** for secure storage
- **Simple user interface** for adding manual data
- **Retrieval and analysis** capabilities
- **Comprehensive statistics** display

### 3. 🤖 Intelligent Prediction System
- **Machine learning algorithm** for training from existing data
- **Accurate prediction** of insulation surface temperature for new geometries
- **Efficiency calculation** for insulation and temperature reduction
- **No dependency** on complex external libraries

### 4. 📊 Advanced Reporting
- **Beautiful HTML reports** generation
- **Statistics display** and performance analysis
- **User-friendly interface** and comprehensible output

## 🛠️ Installation and Setup

### Prerequisites
This program only requires Python 3.6+ and uses standard Python libraries.

### Download and Run
```bash
# Download project
git clone [repository-url]
cd thermal-analyzer

# Run main program
python3 thermal_analyzer_english.py

# Run demo
python3 demo_english_clean.py
```

## 📋 Usage Guide

### 1. Running Demo
For quick familiarization with features:
```bash
python3 demo_english_clean.py
```

### 2. Using Main Program
```bash
python3 thermal_analyzer_english.py
```

### Main Program Menu:
```
1. Import HTML files
2. Add manual data
3. Train prediction model
4. Predict insulation temperature for new geometry
5. View data statistics
6. Exit
```

## 📁 File Structure

```
thermal-analyzer/
├── 📄 thermal_analyzer_english.py    # Main English program
├── 📄 simple_thermal_analyzer.py     # Persian version
├── �� demo_english_clean.py         # English demo
├── 📋 requirements.txt              # Dependencies
├── 📖 README_ENGLISH.md            # English usage guide
├── 📁 html_files_english/          # English HTML samples
│   ├── 📄 report_pipe.html
│   ├── 📄 report_sphere.html
│   └── 📄 report_surface.html
├── 🗄️ thermal_data_english.db     # English database
└── 🤖 thermal_model.pkl           # Trained model
```

## 🔧 Supported Geometry Types

| Geometry | English Name | Persian Name | Application |
|----------|-------------|-------------|-------------|
| 🟢 | pipe | لوله | Horizontal/vertical pipes |
| 🔵 | sphere | کره | Spherical equipment |
| 🟠 | cube | مکعب | Cubic equipment |
| 🟡 | surface | سطح | Flat surfaces |

## 🧪 Supported Insulation Types

| Insulation | English Name | Persian Name | Feature |
|------------|-------------|-------------|---------|
| 🟣 | polyurethane | پلی اورتان | High efficiency |
| 🟤 | foam | فوم | Economical |
| ⚪ | glass wool | پشم شیشه | Heat resistant |

## 📊 Sample Output

```
Geometry: sphere | Insulation: polyurethane
Equipment temperature: 220°C
Predicted insulation temperature: 37.4°C
Temperature reduction: 182.6°C (83.0%)
```

## 🎯 Key Advantages

### 1. **Ease of Use**
- Simple text user interface
- No programming knowledge required
- Step-by-step guidance

### 2. **Reliability**
- Local database usage
- Automatic data backup
- Advanced error handling

### 3. **Flexibility**
- Support for various geometry types
- Ability to add new insulation types
- Algorithm development capability

### 4. **High Performance**
- Fast HTML file processing
- Optimized prediction algorithm
- Low system resource consumption

## 🔮 Advanced Features

### Prediction Algorithm
The program uses **distance-based weighted average** method:
- Calculate similarity with training samples
- Assign weights based on feature proximity
- Accurate prediction for new data

### HTML Processing
- Automatic text pattern recognition
- Number and unit extraction
- Support for various formats

## 🎨 Usage Scenarios

### 1. **Mechanical Engineers**
- Analyze existing insulation performance
- Design new insulation systems
- Optimize material selection

### 2. **Consulting Companies**
- Provide specialized reports
- Compare different options
- Calculate economic efficiency

### 3. **Researchers**
- Analyze laboratory data
- Develop new models
- Parametric studies

## 🔄 Future Development Phases

### Phase 1: Base Improvements
- [ ] Add new insulation types
- [ ] Improve prediction accuracy
- [ ] Simple graphical interface

### Phase 2: Advanced Features
- [ ] Excel file support
- [ ] Economic analysis
- [ ] Visual reports

### Phase 3: Artificial Intelligence
- [ ] Deep neural networks
- [ ] Automatic optimization
- [ ] Service life prediction

## 🤝 Contribution and Development

This project is ready to receive suggestions and improvements:
- Bug reports
- New feature suggestions
- Documentation improvements
- Translation to other languages

## 📞 Support

For:
- Technical questions
- New feature requests
- Problem reports
- Usage guidance

Please contact the development team.

## 🎯 Example Usage Session

```bash
$ python3 thermal_analyzer_english.py

=== Thermal Insulation Analysis System ===
Simple version without external dependencies

Available options:
1. Import HTML files
2. Add manual data
3. Train prediction model
4. Predict insulation temperature for new geometry
5. View data statistics
6. Exit

Your choice (1-6): 4

Predicting insulation temperature:
Geometry type (pipe/sphere/cube/surface): pipe
Equipment surface temperature (°C): 200
Cross-section area (m²): 2.5
Heat transfer coefficient (W/m².K): 15
Insulation type (polyurethane/foam/glass wool): polyurethane

*** Predicted insulation surface temperature: 36.6 °C ***
Temperature reduction after insulation: 163.4 °C
Temperature reduction percentage: 81.7%
```

## 📈 Performance Results

### Sample Predictions:
- **Sphere with polyurethane:** 83.0% efficiency
- **Surface with foam:** 80.9% efficiency  
- **Pipe with glass wool:** 84.3% efficiency
- **Cube with foam:** 80.0% efficiency

### Overall Statistics:
- **Equipment temperature range:** 150-310°C
- **Insulation temperature range:** 28-52°C
- **Average efficiency:** 82.5%

---

**Important Note:** This system is designed as a design assistance tool and its results should be reviewed and validated by expert engineers.

🌟 **Good luck using this tool!** 🌟
