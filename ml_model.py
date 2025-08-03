"""
مدل‌های یادگیری ماشین برای پیش‌بینی دمای سطح تجهیزات عایق‌کاری شده
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional

class SurfaceTemperaturePredictor:
    """
    کلاس پیش‌بینی دمای سطح با استفاده از یادگیری ماشین
    """
    
    def __init__(self):
        self.models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1)
        }
        
        self.best_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.is_trained = False
        
        # نام‌های ویژگی‌ها
        self.feature_names = [
            'inner_temperature',
            'ambient_temperature', 
            'wind_speed',
            'surface_area',
            'total_insulation_thickness',
            'average_thermal_conductivity',
            'equipment_type_encoded',
            'layer_count',
            'thermal_resistance'
        ]
    
    def prepare_features(self, colors_data: List[Dict], insulation_layers: List[List[Dict]]) -> pd.DataFrame:
        """
        آماده‌سازی ویژگی‌ها برای مدل یادگیری ماشین
        """
        features_list = []
        
        for i, data in enumerate(colors_data):
            layers = insulation_layers[i] if i < len(insulation_layers) else []
            
            # ویژگی‌های پایه
            features = {
                'inner_temperature': data.get('inner_temperature', 100.0),
                'ambient_temperature': data.get('ambient_temperature', 25.0),
                'wind_speed': data.get('wind_speed', 2.0),
                'surface_area': data.get('surface_area', 1.0),
                'equipment_type': data.get('equipment_type', 'unknown')
            }
            
            # ویژگی‌های محاسبه شده از لایه‌های عایق
            if layers:
                features['total_insulation_thickness'] = sum(layer.get('thickness', 0) for layer in layers)
                features['average_thermal_conductivity'] = np.mean([layer.get('thermal_conductivity', 0.04) for layer in layers])
                features['layer_count'] = len(layers)
                
                # محاسبه مقاومت حرارتی تقریبی
                thermal_resistance = sum(layer.get('thickness', 0) / layer.get('thermal_conductivity', 0.04) for layer in layers)
                features['thermal_resistance'] = thermal_resistance
            else:
                features['total_insulation_thickness'] = 0.05
                features['average_thermal_conductivity'] = 0.04
                features['layer_count'] = 1
                features['thermal_resistance'] = 0.05 / 0.04
            
            # ویژگی‌های ترکیبی
            features['temp_difference'] = features['inner_temperature'] - features['ambient_temperature']
            features['wind_effect'] = features['wind_speed'] * features['surface_area']
            features['insulation_efficiency'] = features['total_insulation_thickness'] / features['average_thermal_conductivity']
            
            # هدف (دمای سطح)
            if 'surface_temperature' in data:
                features['target'] = data['surface_temperature']
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        کدگذاری ویژگی‌های کتگوری
        """
        df_encoded = df.copy()
        
        categorical_columns = ['equipment_type']
        
        for col in categorical_columns:
            if col in df_encoded.columns:
                if fit:
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                    df_encoded[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
                else:
                    if col in self.label_encoders:
                        try:
                            df_encoded[f'{col}_encoded'] = self.label_encoders[col].transform(df_encoded[col].astype(str))
                        except ValueError:
                            # اگر مقدار جدیدی وجود دارد، از حالت عمومی استفاده کن
                            df_encoded[f'{col}_encoded'] = 0
                    else:
                        df_encoded[f'{col}_encoded'] = 0
                
                # حذف ستون اصلی
                df_encoded = df_encoded.drop(columns=[col])
        
        return df_encoded
    
    def train_models(self, df: pd.DataFrame) -> Dict:
        """
        آموزش مدل‌های مختلف و انتخاب بهترین مدل
        """
        if 'target' not in df.columns:
            raise ValueError("ستون target (دمای سطح) در داده‌ها موجود نیست")
        
        # کدگذاری ویژگی‌های کتگوری
        df_encoded = self.encode_categorical_features(df, fit=True)
        
        # جداسازی ویژگی‌ها و هدف
        X = df_encoded.drop(columns=['target'])
        y = df_encoded['target']
        
        self.feature_columns = X.columns.tolist()
        
        # تقسیم داده‌ها
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # نرمال‌سازی ویژگی‌ها
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # آموزش و ارزیابی مدل‌ها
        results = {}
        
        for name, model in self.models.items():
            try:
                # آموزش مدل
                if name in ['LinearRegression', 'Ridge']:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                elif name == 'SVR':
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # ارزیابی
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Cross-validation
                if name in ['LinearRegression', 'Ridge', 'SVR']:
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
                else:
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                
                results[name] = {
                    'model': model,
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'predictions': y_pred
                }
                
            except Exception as e:
                print(f"خطا در آموزش مدل {name}: {str(e)}")
                continue
        
        # انتخاب بهترین مدل بر اساس R²
        if results:
            best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
            self.best_model = results[best_model_name]['model']
            self.best_model_name = best_model_name
            self.is_trained = True
            
            print(f"بهترین مدل: {best_model_name} با R² = {results[best_model_name]['r2']:.4f}")
        
        return results
    
    def predict(self, colors_data: Dict, insulation_layers: List[Dict]) -> Tuple[float, Dict]:
        """
        پیش‌بینی دمای سطح برای داده‌های جدید
        """
        if not self.is_trained:
            raise ValueError("مدل هنوز آموزش نداده شده است")
        
        # آماده‌سازی ویژگی‌ها
        df = self.prepare_features([colors_data], [insulation_layers])
        df_encoded = self.encode_categorical_features(df, fit=False)
        
        # اطمینان از وجود همه ویژگی‌ها
        for col in self.feature_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        
        # مرتب کردن ستون‌ها
        X = df_encoded[self.feature_columns]
        
        # پیش‌بینی
        if self.best_model_name in ['LinearRegression', 'Ridge', 'SVR']:
            X_scaled = self.scaler.transform(X)
            prediction = self.best_model.predict(X_scaled)[0]
        else:
            prediction = self.best_model.predict(X)[0]
        
        # محاسبه اطلاعات اضافی
        info = {
            'model_used': self.best_model_name,
            'input_features': X.iloc[0].to_dict(),
            'temperature_reduction': colors_data.get('inner_temperature', 100) - prediction,
            'efficiency_percentage': ((colors_data.get('inner_temperature', 100) - prediction) / 
                                    (colors_data.get('inner_temperature', 100) - colors_data.get('ambient_temperature', 25))) * 100
        }
        
        return prediction, info
    
    def optimize_hyperparameters(self, df: pd.DataFrame, model_name: str = 'RandomForest') -> Dict:
        """
        بهینه‌سازی هایپرپارامترها
        """
        df_encoded = self.encode_categorical_features(df, fit=True)
        X = df_encoded.drop(columns=['target'])
        y = df_encoded['target']
        
        param_grids = {
            'RandomForest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            'GradientBoosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'SVR': {
                'C': [0.1, 1, 10],
                'epsilon': [0.01, 0.1, 1],
                'gamma': ['scale', 'auto']
            }
        }
        
        if model_name not in param_grids:
            raise ValueError(f"مدل {model_name} پشتیبانی نمی‌شود")
        
        model = self.models[model_name]
        param_grid = param_grids[model_name]
        
        # بهینه‌سازی
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
        
        if model_name in ['SVR']:
            X_scaled = self.scaler.fit_transform(X)
            grid_search.fit(X_scaled, y)
        else:
            grid_search.fit(X, y)
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_model': grid_search.best_estimator_
        }
    
    def get_feature_importance(self) -> Dict:
        """
        اهمیت ویژگی‌ها (فقط برای مدل‌های tree-based)
        """
        if not self.is_trained:
            raise ValueError("مدل هنوز آموزش نداده شده است")
        
        if hasattr(self.best_model, 'feature_importances_'):
            importance_dict = dict(zip(self.feature_columns, self.best_model.feature_importances_))
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        else:
            return {}
    
    def save_model(self, file_path: str):
        """
        ذخیره مدل آموزش دیده
        """
        if not self.is_trained:
            raise ValueError("مدل هنوز آموزش نداده شده است")
        
        model_data = {
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, file_path)
        print(f"مدل در {file_path} ذخیره شد")
    
    def load_model(self, file_path: str):
        """
        بارگذاری مدل ذخیره شده
        """
        try:
            model_data = joblib.load(file_path)
            
            self.best_model = model_data['best_model']
            self.best_model_name = model_data['best_model_name']
            self.scaler = model_data['scaler']
            self.label_encoders = model_data['label_encoders']
            self.feature_columns = model_data['feature_columns']
            self.is_trained = model_data['is_trained']
            
            print(f"مدل از {file_path} بارگذاری شد")
            
        except FileNotFoundError:
            raise FileNotFoundError(f"فایل مدل {file_path} پیدا نشد")
        except Exception as e:
            raise Exception(f"خطا در بارگذاری مدل: {str(e)}")
    
    def plot_results(self, results: Dict, save_path: str = None):
        """
        رسم نتایج مقایسه مدل‌ها
        """
        if not results:
            print("نتایجی برای رسم وجود ندارد")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('مقایسه مدل‌های یادگیری ماشین', fontsize=16)
        
        models = list(results.keys())
        
        # R² Score
        r2_scores = [results[model]['r2'] for model in models]
        axes[0, 0].bar(models, r2_scores)
        axes[0, 0].set_title('R² Score')
        axes[0, 0].set_ylim(0, 1)
        
        # MSE
        mse_scores = [results[model]['mse'] for model in models]
        axes[0, 1].bar(models, mse_scores)
        axes[0, 1].set_title('Mean Squared Error')
        
        # MAE
        mae_scores = [results[model]['mae'] for model in models]
        axes[1, 0].bar(models, mae_scores)
        axes[1, 0].set_title('Mean Absolute Error')
        
        # Cross-validation scores
        cv_means = [results[model]['cv_mean'] for model in models]
        cv_stds = [results[model]['cv_std'] for model in models]
        axes[1, 1].bar(models, cv_means, yerr=cv_stds, capsize=5)
        axes[1, 1].set_title('Cross-Validation R² (Mean ± Std)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()