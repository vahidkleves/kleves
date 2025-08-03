# Thermal Insulation Analysis System

## نظام تحلیل عایق‌کاری حرارتی

Enhanced thermal insulation analysis system with **BeautifulSoup HTML parser** support for extracting data from Model Summary sections in HTML reports.

## Features / ویژگی‌ها

### ✨ HTML Parsing Capabilities
- **BeautifulSoup Integration**: Advanced HTML parsing using BeautifulSoup4
- **Model Summary Extraction**: Specifically designed to extract data from Model Summary tables
- **Multilingual Support**: Supports both English and Persian (Farsi) content
- **Multiple Detection Methods**: Various strategies to locate and extract thermal data
- **Flexible Format Support**: Handles different HTML structures and layouts

### 🔧 Core Features
- **Database Management**: SQLite database for storing thermal data
- **Machine Learning Prediction**: Simple weighted-average prediction model
- **Multiple Geometry Types**: Support for pipe, sphere, cube, and surface geometries
- **Various Insulation Types**: Polyurethane, foam, glass wool, mineral wool, ceramic
- **Comprehensive Data Extraction**: Temperature, geometry, insulation type, area, coefficients

## Installation / نصب

### Prerequisites / پیش‌نیازها

```bash
# Install BeautifulSoup and dependencies
pip install beautifulsoup4 lxml html5lib

# Or if using system package manager
apt install python3-bs4 python3-lxml

# Or using the break-system-packages flag
pip install --break-system-packages beautifulsoup4 lxml html5lib
```

### Files Required / فایل‌های مورد نیاز

- `thermal_analysis.py` - Main system file
- `requirements.txt` - Dependencies list
- `html_files/` - Directory for HTML input files

## Usage / نحوه استفاده

### Running the System / اجرای سیستم

```bash
python3 thermal_analysis.py
```

### Menu Options / گزینه‌های منو

1. **Import HTML files** - Import and parse HTML files from a directory
2. **Add manual data** - Manually input thermal data
3. **Train prediction model** - Train the ML model with existing data
4. **Predict insulation temperature** - Predict for new scenarios
5. **View data statistics** - Display database statistics
6. **Exit** - Close the program

### HTML File Format / فرمت فایل HTML

The system supports HTML files with Model Summary tables like:

```html
<div class="summary-section">
    <h2>Model Summary</h2>
    <table>
        <thead>
            <tr>
                <th>Info.</th>
                <th>Value</th>
                <th>Unit</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Inner Surface Temperature</td>
                <td>500</td>
                <td>C</td>
            </tr>
            <tr>
                <td>Ambient Temperature</td>
                <td>25</td>
                <td>C</td>
            </tr>
            <tr>
                <td>Air Speed</td>
                <td>2</td>
                <td>m/s</td>
            </tr>
            <!-- More rows... -->
        </tbody>
    </table>
</div>
```

### Persian Support / پشتیبانی فارسی

The system also supports Persian HTML files with fields like:
- دمای سطح داخلی (Inner Surface Temperature)
- دمای محیط (Ambient Temperature)
- سرعت هوا (Air Speed)
- حداکثر ضخامت عایق (Maximum Insulation Thickness)

## Data Fields Extracted / فیلدهای استخراج شده

### Temperature Data / داده‌های دما
- **Equipment Surface Temperature** / دمای سطح تجهیز
- **Insulation Surface Temperature** / دمای سطح عایق
- **Ambient Temperature** / دمای محیط

### Geometric Data / داده‌های هندسی
- **Geometry Type**: pipe, sphere, cube, surface
- **Cross-sectional Area** / سطح مقطع
- **Insulation Thickness** / ضخامت عایق

### Material Properties / خواص مواد
- **Insulation Type** / نوع عایق
- **Thermal Conductivity** / ضریب هدایت حرارتی
- **Convection Coefficient** / ضریب همرفت

## Testing / تست

### Run Tests / اجرای تست‌ها

```bash
# Basic parser test
python3 test_parser.py

# Comprehensive system test
python3 comprehensive_test.py
```

### Sample Files / فایل‌های نمونه

The system includes sample HTML files:
- `sample_model_summary.html` - English Model Summary
- `persian_model_summary.html` - Persian Model Summary
- `third_sample.html` - Additional English sample
- Various report files with different formats

## Technical Details / جزئیات فنی

### HTML Parser Features / ویژگی‌های تحلیل‌گر HTML

1. **Multi-method Detection**:
   - Searches for "Model Summary" headings
   - Looks for tables with thermal data patterns
   - Extracts from div elements with relevant classes
   - Falls back to full document parsing

2. **Regex Patterns**:
   - Temperature extraction with unit recognition
   - Geometry type detection
   - Insulation material identification
   - Area and coefficient parsing

3. **Data Validation**:
   - Required field checking
   - Numeric value validation
   - Unit conversion support
   - Default value assignment

### Machine Learning / یادگیری ماشین

- **Algorithm**: Weighted K-Nearest Neighbors
- **Features**: Temperature, area, convection coefficient, geometry code, insulation code
- **Minimum Training Data**: 3 samples required
- **Prediction Output**: Insulation surface temperature

## Output Example / نمونه خروجی

```
Processing file: sample_model_summary.html
✓ File sample_model_summary.html successfully imported.
  - Equipment temp: 500.0°C
  - Insulation temp: 100.0°C
  - Geometry: surface
  - Insulation: polyurethane

Model trained with 7 samples.
✓ Prediction successful: 64.2°C
  - Temperature reduction: 335.8°C (83.9%)
```

## Error Handling / مدیریت خطا

- **File Not Found**: Graceful handling of missing files
- **Parse Errors**: Detailed error reporting for HTML issues
- **Validation Failures**: Clear feedback on missing required data
- **Training Failures**: Informative messages for insufficient data

## Development / توسعه

### Architecture / معماری

```
ThermalAnalyzer
├── HTMLParser (BeautifulSoup-based)
├── ThermalDatabase (SQLite)
├── SimplePredictor (ML model)
└── ThermalData (data structure)
```

### Extending the Parser / گسترش تحلیل‌گر

To add support for new HTML formats:
1. Update `field_mappings` dictionary
2. Add new regex patterns
3. Extend geometry/insulation pattern dictionaries
4. Update validation logic

## License / مجوز

This project is open source and available under the MIT License.

## Contributors / مشارکت‌کنندگان

- Enhanced with BeautifulSoup HTML parsing capabilities
- Multilingual support for Persian and English
- Comprehensive testing framework
