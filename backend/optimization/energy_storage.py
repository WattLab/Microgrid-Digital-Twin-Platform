"""
储能系统模块 - 参考中广核储能电站配置
"""
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json


class BatteryCell:
    """电池单体模型"""
    
    def __init__(self, capacity: float = 280, voltage: float = 3.2):
        """
        Parameters:
        -----------
        capacity : float
            单体容量 (Ah)
        voltage : float
            标称电压 (V)
        """
        self.capacity = capacity  # Ah
        self.voltage = voltage    # V
        self.energy = capacity * voltage / 1000  # kWh
        self.soc = 0.5
        self.soh = 1.0  # 健康度
        self.temperature = 25.0  # 温度
        self.cycle_count = 0


class BatteryModule:
    """电池模组 - 由多个电芯串并联组成"""
    
    def __init__(self, cells_series: int = 16, cells_parallel: int = 1):
        """
        Parameters:
        -----------
        cells_series : int
            串联数量
        cells_parallel : int
            并联数量
        """
        self.cells_series = cells_series
        self.cells_parallel = cells_parallel
        self.cells = [[BatteryCell() for _ in range(cells_parallel)] 
                      for _ in range(cells_series)]
        
        # 模组参数
        self.voltage = 3.2 * cells_series  # 51.2V
        self.capacity = 280 * cells_parallel  # Ah
        self.energy = self.voltage * self.capacity / 1000  # kWh
        self.soc = 0.5
        self.temperature = 25.0


class BatteryCluster:
    """电池簇 - 由多个模组串联组成"""
    
    def __init__(self, modules_count: int = 14):
        """
        Parameters:
        -----------
        modules_count : int
            模组数量 (串联)
        """
        self.modules_count = modules_count
        self.modules = [BatteryModule() for _ in range(modules_count)]
        
        # 簇参数
        self.voltage = 51.2 * modules_count  # 约716.8V
        self.capacity = 280  # Ah
        self.energy = self.voltage * self.capacity / 1000  # 约200kWh
        self.soc = 0.5
        self.soh = 1.0
        self.temperature = 25.0
        self.status = "standby"  # standby, charging, discharging, fault


class PCS:
    """储能变流器 (Power Conversion System)"""
    
    def __init__(self, rated_power: float = 500, efficiency: float = 0.96):
        """
        Parameters:
        -----------
        rated_power : float
            额定功率 (kW)
        efficiency : float
            转换效率
        """
        self.rated_power = rated_power
        self.efficiency = efficiency
        self.current_power = 0.0
        self.status = "standby"  # standby, running, fault
        self.temperature = 35.0
        
    def charge(self, power: float, battery_soc: float) -> float:
        """充电控制"""
        # 充电功率限制
        if battery_soc > 0.95:
            power = power * 0.3  # 接近满电时降功率
        
        actual_power = min(power, self.rated_power)
        self.current_power = actual_power
        self.status = "running"
        return actual_power * self.efficiency
    
    def discharge(self, power: float, battery_soc: float) -> float:
        """放电控制"""
        # 放电功率限制
        if battery_soc < 0.1:
            power = power * 0.3  # 电量低时降功率
        
        actual_power = min(power, self.rated_power)
        self.current_power = -actual_power
        self.status = "running"
        return actual_power * self.efficiency


class BMS:
    """电池管理系统 (Battery Management System)"""
    
    def __init__(self):
        self.voltage_max = 750  # 最高允许电压
        self.voltage_min = 600  # 最低允许电压
        self.current_max = 500  # 最大电流
        self.temp_max = 55      # 最高温度
        self.temp_min = 0       # 最低温度
        self.soc_max = 0.95     # 最大SOC
        self.soc_min = 0.05     # 最小SOC
        
        self.alarms = []
        self.warnings = []
        
    def check_status(self, cluster: BatteryCluster) -> Dict:
        """检查电池状态"""
        self.alarms = []
        self.warnings = []
        
        # 电压检查
        if cluster.voltage > self.voltage_max:
            self.alarms.append("过压告警")
        elif cluster.voltage > self.voltage_max * 0.95:
            self.warnings.append("电压偏高")
        
        if cluster.voltage < self.voltage_min:
            self.alarms.append("欠压告警")
        elif cluster.voltage < self.voltage_min * 1.05:
            self.warnings.append("电压偏低")
        
        # 温度检查
        if cluster.temperature > self.temp_max:
            self.alarms.append("过温告警")
        elif cluster.temperature > self.temp_max * 0.9:
            self.warnings.append("温度偏高")
        
        if cluster.temperature < self.temp_min:
            self.alarms.append("低温告警")
        
        # SOC检查
        if cluster.soc > self.soc_max:
            self.warnings.append("SOC过高")
        elif cluster.soc < self.soc_min:
            self.warnings.append("SOC过低")
        
        return {
            "status": "fault" if self.alarms else ("warning" if self.warnings else "normal"),
            "alarms": self.alarms,
            "warnings": self.warnings,
            "voltage": cluster.voltage,
            "temperature": cluster.temperature,
            "soc": cluster.soc,
        }


class EnergyStorageStation:
    """
    储能电站 - 参考中广核储能电站配置
    
    典型配置: 100MW/200MWh 储能电站
    - 电池类型: 磷酸铁锂 (LFP)
    - 电池容量: 280Ah
    - 系统结构: 电芯 -> 模组 -> 簇 -> 集装箱 -> 电站
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Parameters:
        -----------
        config : dict
            储能电站配置参数
        """
        self.config = config or self._default_config()
        
        # 初始化组件
        self.clusters_count = self.config["clusters_count"]
        self.pcs_count = self.config["pcs_count"]
        
        # 电池簇
        self.clusters = [BatteryCluster(modules_count=14) 
                        for _ in range(self.clusters_count)]
        
        # PCS
        self.pcs_units = [PCS(rated_power=self.config["pcs_power"]) 
                         for _ in range(self.pcs_count)]
        
        # BMS
        self.bms = BMS()
        
        # 系统参数
        self.rated_power = self.config["rated_power"]  # 额定功率 kW
        self.rated_capacity = self.config["rated_capacity"]  # 额定容量 kWh
        self.current_power = 0.0
        self.soc = 0.5
        self.soh = 1.0
        self.cycle_count = 0
        self.status = "standby"
        
        # 运行统计
        self.total_charge_energy = 0.0  # 累计充电量
        self.total_discharge_energy = 0.0  # 累计放电量
        self.daily_charge_energy = 0.0
        self.daily_discharge_energy = 0.0
        
    def _default_config(self) -> Dict:
        """默认配置 - 10MW/20MWh 储能电站"""
        return {
            "station_name": "微电网储能站",
            "rated_power": 10000,  # 10MW
            "rated_capacity": 20000,  # 20MWh
            "clusters_count": 100,  # 100个电池簇
            "pcs_count": 20,  # 20台PCS
            "pcs_power": 500,  # 每台PCS 500kW
            "efficiency_charge": 0.95,
            "efficiency_discharge": 0.95,
            "min_soc": 0.05,
            "max_soc": 0.95,
            "charge_rate": 0.5,  # 0.5C
            "discharge_rate": 0.5,  # 0.5C
        }
    
    def get_available_charge_power(self) -> float:
        """获取可用充电功率"""
        if self.soc >= self.config["max_soc"]:
            return 0.0
        
        # 根据SOC调整充电功率
        soc_factor = 1.0
        if self.soc > 0.8:
            soc_factor = 0.5
        elif self.soc > 0.9:
            soc_factor = 0.2
        
        return self.rated_power * soc_factor
    
    def get_available_discharge_power(self) -> float:
        """获取可用放电功率"""
        if self.soc <= self.config["min_soc"]:
            return 0.0
        
        # 根据SOC调整放电功率
        soc_factor = 1.0
        if self.soc < 0.2:
            soc_factor = 0.5
        elif self.soc < 0.1:
            soc_factor = 0.2
        
        return self.rated_power * soc_factor
    
    def charge(self, power: float, duration_hours: float = 1.0) -> Dict:
        """
        充电操作
        
        Parameters:
        -----------
        power : float
            充电功率 (kW)
        duration_hours : float
            充电时长 (小时)
        """
        available_power = self.get_available_charge_power()
        actual_power = min(power, available_power)
        
        if actual_power <= 0:
            return {"success": False, "message": "无法充电", "power": 0}
        
        # 计算充电量
        energy = actual_power * duration_hours * self.config["efficiency_charge"]
        
        # 更新SOC
        delta_soc = energy / self.rated_capacity
        new_soc = min(self.config["max_soc"], self.soc + delta_soc)
        actual_energy = (new_soc - self.soc) * self.rated_capacity
        
        self.soc = new_soc
        self.current_power = actual_power
        self.status = "charging"
        self.total_charge_energy += actual_energy
        self.daily_charge_energy += actual_energy
        
        # 更新电池簇状态
        for cluster in self.clusters:
            cluster.soc = self.soc
            cluster.status = "charging"
        
        return {
            "success": True,
            "power": actual_power,
            "energy": actual_energy,
            "soc": self.soc,
            "duration": duration_hours,
        }
    
    def discharge(self, power: float, duration_hours: float = 1.0) -> Dict:
        """
        放电操作
        
        Parameters:
        -----------
        power : float
            放电功率 (kW)
        duration_hours : float
            放电时长 (小时)
        """
        available_power = self.get_available_discharge_power()
        actual_power = min(power, available_power)
        
        if actual_power <= 0:
            return {"success": False, "message": "无法放电", "power": 0}
        
        # 计算放电量
        energy = actual_power * duration_hours / self.config["efficiency_discharge"]
        
        # 更新SOC
        delta_soc = energy / self.rated_capacity
        new_soc = max(self.config["min_soc"], self.soc - delta_soc)
        actual_energy = (self.soc - new_soc) * self.rated_capacity
        
        self.soc = new_soc
        self.current_power = -actual_power
        self.status = "discharging"
        self.total_discharge_energy += actual_energy
        self.daily_discharge_energy += actual_energy
        
        # 更新电池簇状态
        for cluster in self.clusters:
            cluster.soc = self.soc
            cluster.status = "discharging"
        
        return {
            "success": True,
            "power": actual_power,
            "energy": actual_energy,
            "soc": self.soc,
            "duration": duration_hours,
        }
    
    def standby(self):
        """待机"""
        self.status = "standby"
        self.current_power = 0.0
        for cluster in self.clusters:
            cluster.status = "standby"
    
    def get_status(self) -> Dict:
        """获取储能站状态"""
        # 检查BMS状态
        bms_status = self.bms.check_status(self.clusters[0]) if self.clusters else {}
        
        return {
            "station_name": self.config["station_name"],
            "status": self.status,
            "rated_power": self.rated_power,
            "rated_capacity": self.rated_capacity,
            "current_power": self.current_power,
            "soc": round(self.soc, 3),
            "soh": round(self.soh, 3),
            "available_energy": round(self.soc * self.rated_capacity, 2),
            "available_charge_power": round(self.get_available_charge_power(), 2),
            "available_discharge_power": round(self.get_available_discharge_power(), 2),
            "clusters_count": self.clusters_count,
            "pcs_count": self.pcs_count,
            "cycle_count": self.cycle_count,
            "total_charge_energy": round(self.total_charge_energy, 2),
            "total_discharge_energy": round(self.total_discharge_energy, 2),
            "daily_charge_energy": round(self.daily_charge_energy, 2),
            "daily_discharge_energy": round(self.daily_discharge_energy, 2),
            "bms_status": bms_status,
            "efficiency": self.config["efficiency_charge"],
        }
    
    def reset_daily_stats(self):
        """重置每日统计"""
        self.daily_charge_energy = 0.0
        self.daily_discharge_energy = 0.0


class SmallEnergyStorage(EnergyStorageStation):
    """小型储能系统 - 适用于微电网"""
    
    def __init__(self):
        config = {
            "station_name": "微电网储能系统",
            "rated_power": 200,  # 200kW
            "rated_capacity": 500,  # 500kWh
            "clusters_count": 3,
            "pcs_count": 2,
            "pcs_power": 100,
            "efficiency_charge": 0.95,
            "efficiency_discharge": 0.95,
            "min_soc": 0.1,
            "max_soc": 0.9,
            "charge_rate": 0.5,
            "discharge_rate": 0.5,
        }
        super().__init__(config)

















