# نرم‌افزار تحلیل انتقال حرارت و پیش‌بینی دمای سطح

این پروژه یک سیستم هوشمند برای تحلیل انتقال حرارت و پیش‌بینی دمای سطح تجهیزات عایق‌کاری شده است که با استفاده از یادگیری ماشین کار می‌کند.

## ویژگی‌ها

- خواندن داده‌های خروجی نرم‌افزار Colors (فایل‌های HTML)
- خواندن مشخصات عایق‌ها از فایل Excel
- محاسبه انتقال حرارت با استفاده از معادلات فیزیکی
- پیش‌بینی دمای سطح با استفاده از مدل‌های یادگیری ماشین
- پشتیبانی از انواع مختلف عایق‌ها: Cerablanket, Silika Needeled Mat, Rock Wool, Needeled Mat

## نصب

```bash
pip install -r requirements.txt
```

## استفاده

```python
from heat_transfer_analyzer import HeatTransferAnalyzer

analyzer = HeatTransferAnalyzer()
analyzer.load_colors_data('colors_output.html')
analyzer.load_insulation_data('inputdata.xlsx')
surface_temp = analyzer.predict_surface_temperature()
```

## ساختار فایل‌ها

- `colors_output.html`: خروجی نرم‌افزار Colors
- `inputdata.xlsx`: مشخصات عایق‌ها
- `heat_transfer_analyzer.py`: کلاس اصلی تحلیل
- `ml_model.py`: مدل‌های یادگیری ماشین
- `heat_calculations.py`: محاسبات انتقال حرارت
