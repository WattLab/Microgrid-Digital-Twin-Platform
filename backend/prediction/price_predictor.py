"""
电价预测模块 - 基于分时电价和市场预测
参考国内典型工商业分时电价政策
"""
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional


class ElectricityPricePredictor:
    """电价预测器 - 基于中国分时电价政策"""
    
    def __init__(self, region: str = "guangdong"):
        """
        初始化电价预测器
        
        Parameters:
        -----------
        region : str
            地区，不同地区电价政策不同
            支持: guangdong(广东), jiangsu(江苏), zhejiang(浙江)
        """
        self.region = region
        self.price_policies = self._init_price_policies()
        
    def _init_price_policies(self) -> Dict:
        """初始化各地区电价政策 (单位: 元/kWh)"""
        policies = {
            # 广东省工商业分时电价 (2024年标准)
            "guangdong": {
                "peak": 1.2538,      # 尖峰时段
                "high": 1.0538,      # 高峰时段  
                "mid": 0.6838,       # 平段
                "low": 0.3338,       # 低谷时段
                "deep_low": 0.2338,  # 深谷时段
                # 时段划分 (小时)
                "peak_hours": [14, 15, 16, 17],  # 尖峰: 14:00-18:00
                "high_hours": [10, 11, 12, 13, 18, 19, 20, 21],  # 高峰
                "low_hours": [0, 1, 2, 3, 4, 5, 6, 7],  # 低谷: 0:00-8:00
                "deep_low_hours": [4, 5],  # 深谷: 4:00-6:00
                # 售电价格 (上网电价)
                "sell_peak": 0.65,
                "sell_high": 0.55,
                "sell_mid": 0.45,
                "sell_low": 0.35,
            },
            # 江苏省工商业分时电价
            "jiangsu": {
                "peak": 1.1823,
                "high": 0.9823,
                "mid": 0.6523,
                "low": 0.3223,
                "deep_low": 0.2223,
                "peak_hours": [10, 11, 14, 15, 16, 17],
                "high_hours": [8, 9, 12, 13, 18, 19, 20, 21],
                "low_hours": [0, 1, 2, 3, 4, 5, 6, 7, 22, 23],
                "deep_low_hours": [3, 4, 5],
                "sell_peak": 0.62,
                "sell_high": 0.52,
                "sell_mid": 0.42,
                "sell_low": 0.32,
            },
            # 浙江省工商业分时电价
            "zhejiang": {
                "peak": 1.2156,
                "high": 1.0156,
                "mid": 0.6656,
                "low": 0.3356,
                "deep_low": 0.2356,
                "peak_hours": [13, 14, 15, 16, 17],
                "high_hours": [9, 10, 11, 12, 18, 19, 20, 21],
                "low_hours": [0, 1, 2, 3, 4, 5, 6, 7, 22, 23],
                "deep_low_hours": [2, 3, 4, 5],
                "sell_peak": 0.63,
                "sell_high": 0.53,
                "sell_mid": 0.43,
                "sell_low": 0.33,
            }
        }
        return policies.get(self.region, policies["guangdong"])
    
    def get_period_type(self, hour: int) -> str:
        """获取时段类型"""
        policy = self.price_policies
        
        if hour in policy.get("deep_low_hours", []):
            return "deep_low"
        elif hour in policy["low_hours"]:
            return "low"
        elif hour in policy["peak_hours"]:
            return "peak"
        elif hour in policy["high_hours"]:
            return "high"
        else:
            return "mid"
    
    def get_buy_price(self, hour: int) -> float:
        """获取购电价格"""
        period = self.get_period_type(hour)
        return self.price_policies.get(period, self.price_policies["mid"])
    
    def get_sell_price(self, hour: int) -> float:
        """获取售电价格"""
        period = self.get_period_type(hour)
        sell_key = f"sell_{period}"
        if sell_key in self.price_policies:
            return self.price_policies[sell_key]
        return self.price_policies.get("sell_mid", 0.45)
    
    def predict_prices(self, hours: int = 24, start_time: Optional[datetime] = None) -> List[Dict]:
        """
        预测未来电价
        
        Returns:
        --------
        list: 包含每小时电价预测的列表
        """
        if start_time is None:
            start_time = datetime.now()
        
        predictions = []
        for i in range(hours):
            forecast_time = start_time + timedelta(hours=i)
            hour = forecast_time.hour
            
            buy_price = self.get_buy_price(hour)
            sell_price = self.get_sell_price(hour)
            period_type = self.get_period_type(hour)
            
            # 添加一些随机波动模拟市场
            price_noise = np.random.normal(0, 0.01)
            buy_price = round(buy_price * (1 + price_noise), 4)
            sell_price = round(sell_price * (1 + price_noise), 4)
            
            predictions.append({
                "hour": i,
                "forecast_time": forecast_time.isoformat(),
                "period_type": period_type,
                "buy_price": buy_price,
                "sell_price": sell_price,
                "price_spread": round(buy_price - sell_price, 4),
                "is_optimal_charge": period_type in ["low", "deep_low"],
                "is_optimal_discharge": period_type in ["peak", "high"],
            })
        
        return predictions
    
    def get_daily_price_summary(self) -> Dict:
        """获取每日电价摘要"""
        policy = self.price_policies
        return {
            "region": self.region,
            "peak_price": policy["peak"],
            "high_price": policy["high"],
            "mid_price": policy["mid"],
            "low_price": policy["low"],
            "deep_low_price": policy.get("deep_low", policy["low"]),
            "peak_hours": policy["peak_hours"],
            "high_hours": policy["high_hours"],
            "low_hours": policy["low_hours"],
            "avg_buy_price": round(np.mean([policy["peak"], policy["high"], policy["mid"], policy["low"]]), 4),
            "price_ratio": round(policy["peak"] / policy["low"], 2),  # 峰谷比
        }


class MarketPricePredictor:
    """市场化电价预测器 - 用于电力现货市场"""
    
    def __init__(self):
        self.base_price = 0.5  # 基准电价
        self.volatility = 0.15  # 价格波动率
        
    def predict_spot_prices(self, hours: int = 24) -> List[Dict]:
        """预测现货市场电价"""
        predictions = []
        now = datetime.now()
        
        for i in range(hours):
            forecast_time = now + timedelta(hours=i)
            hour = forecast_time.hour
            
            # 基于时间的价格模式
            time_factor = 1.0
            if 10 <= hour <= 12 or 18 <= hour <= 21:  # 用电高峰
                time_factor = 1.5
            elif 14 <= hour <= 17:  # 光伏高峰，价格低
                time_factor = 0.7
            elif 0 <= hour <= 6:  # 夜间低谷
                time_factor = 0.5
            
            # 添加随机波动
            random_factor = 1 + np.random.normal(0, self.volatility)
            
            spot_price = self.base_price * time_factor * random_factor
            spot_price = max(0.1, min(2.0, spot_price))  # 限制价格范围
            
            predictions.append({
                "hour": i,
                "forecast_time": forecast_time.isoformat(),
                "spot_price": round(spot_price, 4),
                "confidence": round(0.9 - i * 0.02, 2),  # 置信度随时间降低
            })
        
        return predictions

