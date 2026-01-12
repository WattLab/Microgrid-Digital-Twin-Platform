"""
能量调度优化器 - 基础版
"""
import numpy as np
from typing import Dict, List, Optional


class EnergyScheduler:
    """能量调度优化器"""
    
    def __init__(self):
        self.parameters = {
            'battery_capacity': 500,  # kWh
            'battery_power': 200,     # kW
            'diesel_capacity': 100,   # kW
            'grid_max_power': 300,    # kW
            'diesel_cost': 0.8,       # 元/kWh
            'min_soc': 0.1,
            'max_soc': 0.9,
        }
        
    def optimize(self, load_forecast: List[float], 
                 solar_forecast: List[float],
                 wind_forecast: List[float]) -> Dict:
        """
        优化能量调度
        
        目标: 最小化运行成本
        """
        hours = len(load_forecast)
        
        schedule = []
        soc = 0.5  # 初始SOC
        total_cost = 0
        
        for i in range(hours):
            load = load_forecast[i]
            solar = solar_forecast[i]
            wind = wind_forecast[i]
            
            renewable = solar + wind
            net_load = load - renewable
            
            # 调度决策
            battery_power = 0
            diesel_power = 0
            grid_power = 0
            
            if net_load > 0:
                # 需要补充电力
                # 优先使用储能
                max_discharge = min(
                    self.parameters['battery_power'],
                    (soc - self.parameters['min_soc']) * self.parameters['battery_capacity']
                )
                battery_power = min(max_discharge, net_load)
                soc -= battery_power / self.parameters['battery_capacity']
                net_load -= battery_power
                
                # 从电网购电
                if net_load > 0:
                    grid_power = min(net_load, self.parameters['grid_max_power'])
                    net_load -= grid_power
                
                # 柴油机补充
                if net_load > 0:
                    diesel_power = min(net_load, self.parameters['diesel_capacity'])
            else:
                # 有余电
                excess = -net_load
                # 储能充电
                max_charge = min(
                    self.parameters['battery_power'],
                    (self.parameters['max_soc'] - soc) * self.parameters['battery_capacity']
                )
                charge = min(max_charge, excess)
                battery_power = -charge
                soc += charge / self.parameters['battery_capacity']
                excess -= charge
                
                # 卖给电网
                if excess > 0:
                    grid_power = -excess
            
            # 计算成本
            hour_cost = diesel_power * self.parameters['diesel_cost']
            total_cost += hour_cost
            
            schedule.append({
                'hour': i,
                'load': round(load, 2),
                'solar': round(solar, 2),
                'wind': round(wind, 2),
                'battery_power': round(battery_power, 2),
                'diesel_power': round(diesel_power, 2),
                'grid_power': round(grid_power, 2),
                'soc': round(soc, 3),
                'cost': round(hour_cost, 2)
            })
        
        return {
            'schedule': schedule,
            'total_cost': round(total_cost, 2),
            'final_soc': round(soc, 3)
        }
