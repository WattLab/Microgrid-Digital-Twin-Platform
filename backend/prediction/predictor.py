"""
预测模块 - 负荷预测和可再生能源发电预测
"""
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional


class LoadPredictor:
    """负荷预测器"""
    
    def __init__(self):
        self.history_data = []
        # 典型日负荷曲线（归一化） - 工商业负荷特性
        self.load_pattern = {
            'weekday': [
                0.45, 0.40, 0.38, 0.35, 0.35, 0.40,  # 0-5
                0.55, 0.75, 0.90, 0.95, 1.00, 0.95,  # 6-11
                0.85, 0.90, 0.98, 1.00, 0.95, 0.90,  # 12-17
                0.85, 0.80, 0.75, 0.65, 0.55, 0.50   # 18-23
            ],
            'weekend': [
                0.35, 0.32, 0.30, 0.28, 0.28, 0.30,  # 0-5
                0.40, 0.50, 0.60, 0.65, 0.70, 0.68,  # 6-11
                0.65, 0.68, 0.72, 0.75, 0.72, 0.70,  # 12-17
                0.68, 0.65, 0.60, 0.55, 0.45, 0.38   # 18-23
            ]
        }
        self.base_load = 100  # 基准负荷 kW
        self.peak_load = 200  # 峰值负荷 kW
        
    def add_history(self, data: List[Dict]):
        """添加历史数据"""
        for d in data:
            if 'load_power' in d:
                self.history_data.append(d['load_power'])
        
        # 根据历史数据更新基准负荷
        if len(self.history_data) > 24:
            self.base_load = np.mean(self.history_data[-24:])
            self.peak_load = max(self.history_data[-24:]) * 1.1
    
    def predict(self, hours: int = 24, start_time: Optional[datetime] = None) -> List[Dict]:
        """
        负荷预测
        
        Parameters:
        -----------
        hours : int
            预测时长
        start_time : datetime
            起始时间
        """
        if start_time is None:
            start_time = datetime.now()
        
        predictions = []
        
        for i in range(hours):
            forecast_time = start_time + timedelta(hours=i)
            hour = forecast_time.hour
            is_weekday = forecast_time.weekday() < 5
            
            # 获取负荷模式
            pattern = self.load_pattern['weekday'] if is_weekday else self.load_pattern['weekend']
            base_factor = pattern[hour]
            
            # 季节调整（夏季和冬季负荷更高）
            month = forecast_time.month
            if month in [6, 7, 8]:  # 夏季
                season_factor = 1.2
            elif month in [12, 1, 2]:  # 冬季
                season_factor = 1.15
            else:
                season_factor = 1.0
            
            # 添加随机波动
            noise = np.random.normal(0, 0.05)
            
            # 预测值
            predicted_load = self.base_load * base_factor * season_factor * (1 + noise)
            predicted_load = max(self.base_load * 0.3, min(self.peak_load, predicted_load))
            
            # 置信区间
            uncertainty = 0.1 * (1 + i * 0.02)  # 随时间增加不确定性
            lower = predicted_load * (1 - uncertainty)
            upper = predicted_load * (1 + uncertainty)
            
            predictions.append({
                'hour': i,
                'forecast_time': forecast_time.isoformat(),
                'predicted_value': round(predicted_load, 2),
                'lower_bound': round(lower, 2),
                'upper_bound': round(upper, 2),
                'confidence': round(1 - uncertainty, 2),
                'is_peak': hour in [10, 11, 14, 15, 16, 17],
            })
        
        return predictions


class RenewablePredictor:
    """可再生能源发电预测器"""
    
    def __init__(self, source_type: str = 'solar'):
        """
        Parameters:
        -----------
        source_type : str
            'solar' 或 'wind'
        """
        self.source_type = source_type
        self.history_data = []
        
        # 光伏发电曲线
        self.solar_pattern = [
            0.00, 0.00, 0.00, 0.00, 0.00, 0.00,  # 0-5 (夜间)
            0.05, 0.20, 0.45, 0.70, 0.85, 0.95,  # 6-11 (上午)
            1.00, 0.98, 0.90, 0.75, 0.50, 0.25,  # 12-17 (下午)
            0.05, 0.00, 0.00, 0.00, 0.00, 0.00   # 18-23 (夜间)
        ]
        
        # 风力发电曲线（相对平稳，夜间稍强）
        self.wind_pattern = [
            0.75, 0.78, 0.80, 0.82, 0.85, 0.82,  # 0-5
            0.70, 0.60, 0.55, 0.50, 0.48, 0.50,  # 6-11
            0.52, 0.55, 0.58, 0.60, 0.65, 0.70,  # 12-17
            0.75, 0.78, 0.80, 0.82, 0.78, 0.76   # 18-23
        ]
        
        self.rated_power = 100 if source_type == 'solar' else 50  # 装机容量 kW
        
    def add_history(self, data: List[Dict]):
        """添加历史数据"""
        key = f'{self.source_type}_power'
        for d in data:
            if key in d:
                self.history_data.append(d[key])
    
    def predict(self, hours: int = 24, start_time: Optional[datetime] = None,
                weather_factor: float = 1.0) -> List[Dict]:
        """
        发电预测
        
        Parameters:
        -----------
        hours : int
            预测时长
        weather_factor : float
            天气影响因子 (0-1.2)
        """
        if start_time is None:
            start_time = datetime.now()
        
        predictions = []
        pattern = self.solar_pattern if self.source_type == 'solar' else self.wind_pattern
        
        for i in range(hours):
            forecast_time = start_time + timedelta(hours=i)
            hour = forecast_time.hour
            
            # 基础发电曲线
            base_factor = pattern[hour]
            
            # 天气影响
            if self.source_type == 'solar':
                # 光伏受云量影响大
                weather_effect = weather_factor * np.random.uniform(0.85, 1.15)
            else:
                # 风力受风速影响
                weather_effect = weather_factor * np.random.uniform(0.7, 1.3)
            
            # 季节影响（光伏夏季发电量更高）
            month = forecast_time.month
            if self.source_type == 'solar':
                if month in [5, 6, 7, 8, 9]:  # 夏季
                    season_factor = 1.2
                elif month in [11, 12, 1, 2]:  # 冬季
                    season_factor = 0.7
                else:
                    season_factor = 1.0
            else:
                # 风力春冬季更强
                if month in [3, 4, 10, 11]:
                    season_factor = 1.2
                else:
                    season_factor = 1.0
            
            # 预测值
            predicted_power = self.rated_power * base_factor * weather_effect * season_factor
            predicted_power = max(0, min(self.rated_power * 1.1, predicted_power))
            
            # 置信区间
            uncertainty = 0.15 * (1 + i * 0.02)
            lower = predicted_power * (1 - uncertainty)
            upper = predicted_power * (1 + uncertainty)
            
            predictions.append({
                'hour': i,
                'forecast_time': forecast_time.isoformat(),
                'predicted_value': round(predicted_power, 2),
                'lower_bound': round(max(0, lower), 2),
                'upper_bound': round(upper, 2),
                'confidence': round(1 - uncertainty, 2),
                'capacity_factor': round(predicted_power / self.rated_power, 2),
            })
        
        return predictions


class IntegratedPredictor:
    """综合预测器 - 结合多种预测"""
    
    def __init__(self):
        self.load_predictor = LoadPredictor()
        self.solar_predictor = RenewablePredictor('solar')
        self.wind_predictor = RenewablePredictor('wind')
    
    def predict_all(self, hours: int = 24) -> Dict:
        """预测所有数据"""
        return {
            'load': self.load_predictor.predict(hours),
            'solar': self.solar_predictor.predict(hours),
            'wind': self.wind_predictor.predict(hours),
        }
    
    def predict_net_load(self, hours: int = 24) -> List[Dict]:
        """预测净负荷（负荷 - 可再生能源）"""
        all_pred = self.predict_all(hours)
        
        net_load = []
        for i in range(hours):
            load = all_pred['load'][i]['predicted_value']
            solar = all_pred['solar'][i]['predicted_value']
            wind = all_pred['wind'][i]['predicted_value']
            
            net = load - solar - wind
            
            net_load.append({
                'hour': i,
                'load': load,
                'solar': solar,
                'wind': wind,
                'net_load': round(net, 2),
                'surplus': round(-net, 2) if net < 0 else 0,
                'deficit': round(net, 2) if net > 0 else 0,
            })
        
        return net_load
