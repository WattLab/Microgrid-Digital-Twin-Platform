"""
智能调度优化器 - 基于电价预测的最优购售电策略
目标: 最小化每日购电成本
"""
import numpy as np
from scipy.optimize import minimize, linprog
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta


class SmartEnergyScheduler:
    """
    智能能量调度器
    
    优化目标:
    1. 最小化每日购电成本
    2. 最大化光伏自消纳
    3. 利用峰谷价差套利
    4. 保证负荷供电可靠性
    """
    
    def __init__(self, storage_config: Dict, price_predictor):
        """
        Parameters:
        -----------
        storage_config : dict
            储能系统配置
        price_predictor : ElectricityPricePredictor
            电价预测器
        """
        self.storage = storage_config
        self.price_predictor = price_predictor
        
        # 储能参数
        self.capacity = storage_config.get("rated_capacity", 500)  # kWh
        self.max_power = storage_config.get("rated_power", 200)  # kW
        self.efficiency_charge = storage_config.get("efficiency_charge", 0.95)
        self.efficiency_discharge = storage_config.get("efficiency_discharge", 0.95)
        self.min_soc = storage_config.get("min_soc", 0.1)
        self.max_soc = storage_config.get("max_soc", 0.9)
        
    def optimize_daily_schedule(
        self,
        load_forecast: List[float],
        solar_forecast: List[float],
        wind_forecast: List[float],
        initial_soc: float = 0.5
    ) -> Dict:
        """
        优化24小时调度计划
        
        Parameters:
        -----------
        load_forecast : list
            24小时负荷预测 (kW)
        solar_forecast : list
            24小时光伏预测 (kW)
        wind_forecast : list
            24小时风电预测 (kW)
        initial_soc : float
            初始SOC
            
        Returns:
        --------
        dict: 优化后的调度方案
        """
        # 获取电价预测
        price_forecast = self.price_predictor.predict_prices(24)
        
        hours = 24
        
        # 使用启发式优化策略
        schedule = self._heuristic_optimization(
            load_forecast, solar_forecast, wind_forecast,
            price_forecast, initial_soc
        )
        
        # 计算经济指标
        economics = self._calculate_economics(schedule, price_forecast)
        
        return {
            "success": True,
            "schedule": schedule,
            "economics": economics,
            "price_forecast": price_forecast,
        }
    
    def _heuristic_optimization(
        self,
        load_forecast: List[float],
        solar_forecast: List[float],
        wind_forecast: List[float],
        price_forecast: List[Dict],
        initial_soc: float
    ) -> List[Dict]:
        """
        启发式优化策略
        
        策略规则:
        1. 谷电时段充电 (电价低)
        2. 峰电时段放电 (电价高)
        3. 优先消纳可再生能源
        4. 峰谷价差套利
        """
        schedule = []
        soc = initial_soc
        
        for hour in range(24):
            load = load_forecast[hour] if hour < len(load_forecast) else 80
            solar = solar_forecast[hour] if hour < len(solar_forecast) else 0
            wind = wind_forecast[hour] if hour < len(wind_forecast) else 0
            price_info = price_forecast[hour]
            
            buy_price = price_info["buy_price"]
            sell_price = price_info["sell_price"]
            period_type = price_info["period_type"]
            
            # 可再生能源总发电
            renewable = solar + wind
            
            # 净负荷 = 负荷 - 可再生能源
            net_load = load - renewable
            
            # 初始化调度变量
            battery_charge = 0.0
            battery_discharge = 0.0
            grid_purchase = 0.0
            grid_sell = 0.0
            solar_used = min(solar, load)
            wind_used = min(wind, max(0, load - solar_used))
            solar_curtail = 0.0
            wind_curtail = 0.0
            
            # 策略1: 低谷时段 - 尽量充电
            if period_type in ["low", "deep_low"]:
                if soc < self.max_soc:
                    # 计算可充电量
                    charge_capacity = (self.max_soc - soc) * self.capacity
                    charge_power = min(self.max_power, charge_capacity)
                    
                    # 充电
                    battery_charge = charge_power
                    grid_purchase = net_load + battery_charge if net_load > 0 else battery_charge
                    
                    # 更新SOC
                    delta_soc = (battery_charge * self.efficiency_charge) / self.capacity
                    soc = min(self.max_soc, soc + delta_soc)
                else:
                    # 电池已满，仅满足负荷
                    if net_load > 0:
                        grid_purchase = net_load
                    else:
                        grid_sell = -net_load
            
            # 策略2: 尖峰/高峰时段 - 尽量放电
            elif period_type in ["peak", "high"]:
                if net_load > 0 and soc > self.min_soc:
                    # 计算可放电量
                    discharge_capacity = (soc - self.min_soc) * self.capacity
                    discharge_power = min(self.max_power, discharge_capacity, net_load)
                    
                    battery_discharge = discharge_power
                    remaining_load = net_load - battery_discharge
                    
                    if remaining_load > 0:
                        grid_purchase = remaining_load
                    
                    # 更新SOC
                    delta_soc = battery_discharge / (self.capacity * self.efficiency_discharge)
                    soc = max(self.min_soc, soc - delta_soc)
                else:
                    if net_load > 0:
                        grid_purchase = net_load
                    else:
                        grid_sell = -net_load
            
            # 策略3: 平段 - 优先自消纳
            else:
                if net_load > 0:
                    # 负荷大于发电，考虑从电池放电
                    if soc > 0.3:  # 保留一定电量
                        discharge_power = min(self.max_power * 0.5, net_load)
                        battery_discharge = discharge_power
                        remaining_load = net_load - battery_discharge
                        
                        if remaining_load > 0:
                            grid_purchase = remaining_load
                        
                        delta_soc = battery_discharge / (self.capacity * self.efficiency_discharge)
                        soc = max(self.min_soc, soc - delta_soc)
                    else:
                        grid_purchase = net_load
                else:
                    # 发电大于负荷
                    excess = -net_load
                    if soc < 0.7:  # 电池未满时充电
                        charge_power = min(self.max_power * 0.5, excess)
                        battery_charge = charge_power
                        remaining = excess - battery_charge
                        if remaining > 0:
                            grid_sell = remaining
                        
                        delta_soc = (battery_charge * self.efficiency_charge) / self.capacity
                        soc = min(self.max_soc, soc + delta_soc)
                    else:
                        grid_sell = excess
            
            # 计算弃电
            if solar > solar_used:
                solar_curtail = solar - solar_used - battery_charge * (solar / max(renewable, 1))
            if wind > wind_used:
                wind_curtail = wind - wind_used - battery_charge * (wind / max(renewable, 1))
            
            solar_curtail = max(0, solar_curtail)
            wind_curtail = max(0, wind_curtail)
            
            # 计算成本和收益
            hour_cost = grid_purchase * buy_price - grid_sell * sell_price
            
            schedule.append({
                "hour": hour,
                "period_type": period_type,
                "load": round(load, 2),
                "solar_generation": round(solar, 2),
                "wind_generation": round(wind, 2),
                "solar_used": round(solar_used, 2),
                "wind_used": round(wind_used, 2),
                "solar_curtail": round(solar_curtail, 2),
                "wind_curtail": round(wind_curtail, 2),
                "battery_charge": round(battery_charge, 2),
                "battery_discharge": round(battery_discharge, 2),
                "battery_soc": round(soc, 3),
                "grid_purchase": round(grid_purchase, 2),
                "grid_sell": round(grid_sell, 2),
                "buy_price": buy_price,
                "sell_price": sell_price,
                "hour_cost": round(hour_cost, 2),
            })
        
        return schedule
    
    def _calculate_economics(self, schedule: List[Dict], price_forecast: List[Dict]) -> Dict:
        """计算经济指标"""
        total_load = sum(s["load"] for s in schedule)
        total_solar = sum(s["solar_generation"] for s in schedule)
        total_wind = sum(s["wind_generation"] for s in schedule)
        total_renewable = total_solar + total_wind
        
        total_purchase = sum(s["grid_purchase"] for s in schedule)
        total_sell = sum(s["grid_sell"] for s in schedule)
        total_purchase_cost = sum(s["grid_purchase"] * s["buy_price"] for s in schedule)
        total_sell_revenue = sum(s["grid_sell"] * s["sell_price"] for s in schedule)
        
        total_charge = sum(s["battery_charge"] for s in schedule)
        total_discharge = sum(s["battery_discharge"] for s in schedule)
        
        # 计算自消纳率
        renewable_used = sum(s["solar_used"] + s["wind_used"] for s in schedule)
        self_consumption_rate = renewable_used / max(total_renewable, 1)
        
        # 计算峰谷套利收益
        peak_discharge = sum(s["battery_discharge"] for s in schedule if s["period_type"] in ["peak", "high"])
        valley_charge = sum(s["battery_charge"] for s in schedule if s["period_type"] in ["low", "deep_low"])
        
        avg_peak_price = np.mean([p["buy_price"] for p in price_forecast if p["period_type"] in ["peak", "high"]])
        avg_valley_price = np.mean([p["buy_price"] for p in price_forecast if p["period_type"] in ["low", "deep_low"]])
        
        arbitrage_profit = peak_discharge * avg_peak_price - valley_charge * avg_valley_price
        
        # 无储能情况下的成本（对比基准）
        baseline_cost = sum(
            max(0, s["load"] - s["solar_generation"] - s["wind_generation"]) * s["buy_price"]
            for s in schedule
        )
        
        # 实际净成本
        net_cost = total_purchase_cost - total_sell_revenue
        
        # 节省成本
        cost_saving = baseline_cost - net_cost
        cost_saving_rate = cost_saving / max(baseline_cost, 1)
        
        return {
            "total_load": round(total_load, 2),
            "total_solar": round(total_solar, 2),
            "total_wind": round(total_wind, 2),
            "total_renewable": round(total_renewable, 2),
            "total_purchase": round(total_purchase, 2),
            "total_sell": round(total_sell, 2),
            "total_purchase_cost": round(total_purchase_cost, 2),
            "total_sell_revenue": round(total_sell_revenue, 2),
            "net_cost": round(net_cost, 2),
            "baseline_cost": round(baseline_cost, 2),
            "cost_saving": round(cost_saving, 2),
            "cost_saving_rate": round(cost_saving_rate * 100, 1),
            "total_charge": round(total_charge, 2),
            "total_discharge": round(total_discharge, 2),
            "self_consumption_rate": round(self_consumption_rate * 100, 1),
            "arbitrage_profit": round(arbitrage_profit, 2),
            "avg_buy_price": round(total_purchase_cost / max(total_purchase, 1), 4),
            "avg_sell_price": round(total_sell_revenue / max(total_sell, 1), 4),
        }
    
    def get_realtime_strategy(
        self,
        current_hour: int,
        current_load: float,
        current_solar: float,
        current_wind: float,
        current_soc: float
    ) -> Dict:
        """
        获取实时调度策略
        """
        price_info = self.price_predictor.predict_prices(1)[0]
        period_type = price_info["period_type"]
        buy_price = price_info["buy_price"]
        
        renewable = current_solar + current_wind
        net_load = current_load - renewable
        
        action = "standby"
        power = 0.0
        reason = ""
        
        if period_type in ["low", "deep_low"]:
            if current_soc < self.max_soc - 0.1:
                action = "charge"
                power = min(self.max_power, (self.max_soc - current_soc) * self.capacity)
                reason = f"低谷电价 ({buy_price}元/kWh)，执行充电策略"
        
        elif period_type in ["peak", "high"]:
            if current_soc > self.min_soc + 0.1 and net_load > 0:
                action = "discharge"
                power = min(self.max_power, net_load, (current_soc - self.min_soc) * self.capacity)
                reason = f"高峰电价 ({buy_price}元/kWh)，执行放电策略替代购电"
        
        else:
            if net_load < 0 and current_soc < 0.7:
                action = "charge"
                power = min(self.max_power * 0.5, -net_load)
                reason = "平段时段，可再生能源过剩，适度充电"
            elif net_load > 0 and current_soc > 0.4:
                action = "discharge"
                power = min(self.max_power * 0.3, net_load)
                reason = "平段时段，适度放电减少购电"
        
        if not reason:
            reason = "当前无需调整储能状态"
        
        return {
            "action": action,
            "power": round(power, 2),
            "reason": reason,
            "period_type": period_type,
            "buy_price": buy_price,
            "current_soc": current_soc,
            "net_load": round(net_load, 2),
        }

















