"""
微电网仿真引擎
模拟光伏、风电、储能、负荷和电网交互
"""
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json


class MicrogridSimulator:
    """微电网综合仿真器"""
    
    def __init__(self):
        # 默认参数配置
        self.parameters = {
            'solar_capacity': 100,      # 光伏装机容量 kW
            'wind_capacity': 50,        # 风电装机容量 kW
            'diesel_capacity': 100,     # 柴油发电机容量 kW
            'battery_capacity': 500,    # 储能容量 kWh
            'battery_power': 200,       # 储能功率 kW
            'base_load': 100,           # 基础负荷 kW
            'max_load': 200,            # 最大负荷 kW
            'grid_max_import': 300,     # 电网最大输入功率 kW
            'grid_max_export': 200,     # 电网最大输出功率 kW
        }
        
        # 运行状态
        self.current_soc = 0.5
        self.current_hour = 0
        self.diesel_running = False
        
        # 历史数据
        self.simulation_history = []
        
        # 光伏发电曲线
        self.solar_profile = [
            0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
            0.05, 0.20, 0.45, 0.70, 0.85, 0.95,
            1.00, 0.98, 0.90, 0.75, 0.50, 0.25,
            0.05, 0.00, 0.00, 0.00, 0.00, 0.00
        ]
        
        # 负荷曲线
        self.load_profile = [
            0.45, 0.40, 0.38, 0.35, 0.35, 0.40,
            0.55, 0.75, 0.90, 0.95, 1.00, 0.95,
            0.85, 0.90, 0.98, 1.00, 0.95, 0.90,
            0.85, 0.80, 0.75, 0.65, 0.55, 0.50
        ]
        
    def get_parameters_json(self) -> str:
        """获取参数JSON"""
        return json.dumps(self.parameters)
    
    def update_parameters(self, params: Dict):
        """更新参数"""
        self.parameters.update(params)
    
    def generate_solar_power(self, hour: int, weather_factor: float = 1.0) -> float:
        """生成光伏发电功率"""
        base = self.solar_profile[hour % 24]
        noise = np.random.normal(0, 0.05)
        power = self.parameters['solar_capacity'] * base * weather_factor * (1 + noise)
        return max(0, round(power, 2))
    
    def generate_wind_power(self, hour: int, wind_speed: float = 8.0) -> float:
        """生成风力发电功率"""
        # 简化的风速-功率曲线
        if wind_speed < 3:
            factor = 0
        elif wind_speed < 12:
            factor = (wind_speed - 3) / 9
        elif wind_speed < 25:
            factor = 1.0
        else:
            factor = 0  # 超过切出风速
        
        # 添加波动
        noise = np.random.normal(0, 0.1)
        power = self.parameters['wind_capacity'] * factor * (1 + noise)
        return max(0, round(power, 2))
    
    def calculate_load(self, hour: int) -> float:
        """计算负荷"""
        base = self.load_profile[hour % 24]
        noise = np.random.normal(0, 0.08)
        load = self.parameters['base_load'] * (1 + base) * (1 + noise)
        return max(self.parameters['base_load'] * 0.3, round(load, 2))
    
    def simulate_step(self, hour: int, weather_factor: float = 1.0, 
                      wind_speed: float = 8.0) -> Dict:
        """单步仿真"""
        # 发电
        solar_power = self.generate_solar_power(hour, weather_factor)
        wind_power = self.generate_wind_power(hour, wind_speed)
        
        # 负荷
        load_power = self.calculate_load(hour)
        
        # 净负荷
        net_load = load_power - solar_power - wind_power
        
        # 柴油机和电网决策
        diesel_power = 0.0
        battery_power = 0.0
        grid_power = 0.0
        
        if net_load > 0:
            # 需要额外电源
            # 优先使用储能
            max_discharge = min(
                self.parameters['battery_power'],
                (self.current_soc - 0.1) * self.parameters['battery_capacity']
            )
            
            if max_discharge > 0:
                battery_power = min(max_discharge, net_load)
                net_load -= battery_power
                # 更新SOC
                self.current_soc -= battery_power / self.parameters['battery_capacity']
            
            # 剩余从电网购电
            if net_load > 0:
                grid_power = min(net_load, self.parameters['grid_max_import'])
                net_load -= grid_power
            
            # 如果还不够，启动柴油机
            if net_load > 0:
                diesel_power = min(net_load, self.parameters['diesel_capacity'])
                self.diesel_running = True
            else:
                self.diesel_running = False
        else:
            # 有多余电力
            excess = -net_load
            
            # 优先给储能充电
            max_charge = min(
                self.parameters['battery_power'],
                (0.9 - self.current_soc) * self.parameters['battery_capacity']
            )
            
            if max_charge > 0:
                charge = min(max_charge, excess)
                battery_power = -charge  # 负值表示充电
                excess -= charge
                self.current_soc += charge / self.parameters['battery_capacity']
            
            # 多余的卖给电网
            if excess > 0:
                grid_power = -min(excess, self.parameters['grid_max_export'])
        
        # 电压和频率模拟
        voltage = 380 + np.random.normal(0, 2)
        frequency = 50 + np.random.normal(0, 0.02)
        
        result = {
            'hour': hour,
            'timestamp': datetime.now().isoformat(),
            'solar_power': solar_power,
            'wind_power': wind_power,
            'diesel_power': diesel_power,
            'load_power': load_power,
            'battery_power': battery_power,
            'battery_soc': round(self.current_soc, 3),
            'current_soc': round(self.current_soc, 3),
            'grid_power': round(grid_power, 2),
            'voltage': round(voltage, 1),
            'frequency': round(frequency, 2),
            'renewable_ratio': round((solar_power + wind_power) / max(load_power, 1) * 100, 1),
            'power_flows': {
                'solar_to_load': min(solar_power, load_power),
                'wind_to_load': min(wind_power, max(0, load_power - solar_power)),
                'battery_to_load': max(0, battery_power),
                'grid_to_load': max(0, grid_power),
                'solar_to_battery': 0,
                'wind_to_battery': 0,
                'diesel_to_load': diesel_power,
                'diesel_to_battery': 0,
                'load_to_grid': 0,
                'battery_to_grid': 0,
            }
        }
        
        self.simulation_history.append(result)
        return result
    
    def simulate(self, duration_hours: int = 24, weather_factor: float = 1.0,
                 wind_speed: float = 8.0) -> List[Dict]:
        """运行完整仿真"""
        results = []
        for hour in range(duration_hours):
            # 风速随机波动
            ws = wind_speed + np.random.normal(0, 2)
            ws = max(0, min(30, ws))
            
            result = self.simulate_step(hour, weather_factor, ws)
            results.append(result)
        
        return results
    
    def get_summary(self) -> Dict:
        """获取仿真汇总"""
        if not self.simulation_history:
            return {}
        
        total_solar = sum(d['solar_power'] for d in self.simulation_history)
        total_wind = sum(d['wind_power'] for d in self.simulation_history)
        total_load = sum(d['load_power'] for d in self.simulation_history)
        total_grid_import = sum(max(0, d['grid_power']) for d in self.simulation_history)
        total_grid_export = sum(abs(min(0, d['grid_power'])) for d in self.simulation_history)
        
        return {
            'total_solar_generation': round(total_solar, 2),
            'total_wind_generation': round(total_wind, 2),
            'total_load': round(total_load, 2),
            'total_grid_import': round(total_grid_import, 2),
            'total_grid_export': round(total_grid_export, 2),
            'renewable_coverage': round((total_solar + total_wind) / max(total_load, 1) * 100, 1),
            'self_consumption_rate': round((total_load - total_grid_import) / max(total_load, 1) * 100, 1),
        }
    
    def reset(self):
        """重置仿真状态"""
        self.current_soc = 0.5
        self.current_hour = 0
        self.diesel_running = False
        self.simulation_history = []
