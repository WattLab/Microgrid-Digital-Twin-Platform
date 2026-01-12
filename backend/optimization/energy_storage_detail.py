"""
储能系统详细模型 - 电芯级别监控
参考中广核储能电站结构：电芯 -> Pack -> 簇 -> 箱（集装箱） -> 站
包含BMS硬件版、DCDC模块和多层级状态聚合算法
"""
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import random


# ==================== 状态聚合算法 ====================

class StateAggregator:
    """
    多层级状态聚合器
    实现从电芯到站级的状态映射方法
    """
    
    # 聚合方法枚举
    METHODS = {
        "mean": "均值法",
        "weighted_mean": "加权均值法", 
        "min": "最小值法（木桶效应）",
        "max": "最大值法",
        "voting": "投票法（多数决）",
        "hierarchical": "层级递推法",
        "kalman": "卡尔曼滤波法",
        "fuzzy": "模糊综合评价法"
    }
    
    @staticmethod
    def aggregate_soc(values: List[float], weights: List[float] = None, method: str = "weighted_mean") -> float:
        """
        SOC聚合 - 默认使用加权均值法
        对于串联电池组，SOC取最小值更保守
        对于并联电池组，SOC取加权均值
        """
        if not values:
            return 0.0
        if method == "mean":
            return float(np.mean(values))
        elif method == "weighted_mean":
            if weights is None:
                weights = [1.0] * len(values)
            return float(np.average(values, weights=weights))
        elif method == "min":
            return float(min(values))
        else:
            return float(np.mean(values))
    
    @staticmethod
    def aggregate_soh(values: List[float], method: str = "min") -> float:
        """
        SOH聚合 - 默认使用最小值法（木桶效应）
        整体健康状态由最差单体决定
        """
        if not values:
            return 0.0
        if method == "min":
            return float(min(values))
        elif method == "mean":
            return float(np.mean(values))
        elif method == "weighted_mean":
            # 对低SOH赋予更高权重
            weights = [1.0 / (v + 0.01) for v in values]
            return float(np.average(values, weights=weights))
        return float(min(values))
    
    @staticmethod
    def aggregate_temperature(values: List[float], method: str = "max") -> Tuple[float, float, float]:
        """
        温度聚合 - 返回(平均, 最高, 最低)
        安全相关参数关注最高温度
        """
        if not values:
            return (0.0, 0.0, 0.0)
        return (float(np.mean(values)), float(max(values)), float(min(values)))
    
    @staticmethod
    def aggregate_status(statuses: List[str], method: str = "voting") -> str:
        """
        状态聚合 - 使用投票法或最严格法
        voting: 多数决定
        strict: 只要有fault就是fault
        """
        if not statuses:
            return "normal"
        
        if method == "strict":
            if "fault" in statuses:
                return "fault"
            elif "warning" in statuses:
                return "warning"
            return "normal"
        
        elif method == "voting":
            # 统计各状态数量
            counts = {"fault": 0, "warning": 0, "normal": 0}
            for s in statuses:
                counts[s] = counts.get(s, 0) + 1
            
            # 故障优先级最高
            if counts["fault"] > 0:
                return "fault"
            # 警告次之
            if counts["warning"] >= len(statuses) * 0.3:  # 30%以上告警
                return "warning"
            return "normal"
        
        return "normal"
    
    @staticmethod
    def aggregate_voltage(values: List[float], connection: str = "series") -> float:
        """
        电压聚合
        series: 串联，电压相加
        parallel: 并联，电压取平均（理论相等）
        """
        if not values:
            return 0.0
        if connection == "series":
            return float(sum(values))
        else:
            return float(np.mean(values))
    
    @staticmethod
    def get_aggregation_info() -> Dict:
        """获取所有聚合方法的说明"""
        return {
            "soc_aggregation": {
                "method": "weighted_mean",
                "description": "SOC使用加权均值法聚合，对于并联组考虑各单元容量权重",
                "formula": "SOC_agg = Σ(SOC_i × W_i) / Σ(W_i)"
            },
            "soh_aggregation": {
                "method": "min", 
                "description": "SOH使用最小值法（木桶效应），整体健康由最差单体决定",
                "formula": "SOH_agg = min(SOH_1, SOH_2, ..., SOH_n)"
            },
            "temperature_aggregation": {
                "method": "max",
                "description": "温度关注最高值，同时记录平均和最低温度用于均衡分析",
                "formula": "T_critical = max(T_1, T_2, ..., T_n)"
            },
            "status_aggregation": {
                "method": "strict",
                "description": "状态使用严格法，任一故障则整体故障，告警比例超30%则告警",
                "formula": "Status = fault if any(fault) else warning if ratio(warning)>0.3 else normal"
            },
            "voltage_aggregation": {
                "series": "串联电压相加: V_total = Σ(V_i)",
                "parallel": "并联电压取平均: V_total = mean(V_i)"
            }
        }


# ==================== BMS硬件组件 ====================

class BMSModule:
    """
    电池管理系统(BMS)模块
    每个Pack配置一个BMS从机板
    """
    
    def __init__(self, bms_id: str):
        self.bms_id = bms_id
        self.version = "BMS-V3.2"  # 硬件版本
        self.firmware = "FW-2024.12"  # 固件版本
        
        # 采样通道
        self.voltage_channels = 16  # 电压采样通道数
        self.temp_channels = 8      # 温度采样通道数
        
        # 采样精度
        self.voltage_accuracy = 0.001  # 1mV
        self.temp_accuracy = 0.5       # 0.5℃
        self.current_accuracy = 0.01   # 10mA
        
        # 通信
        self.can_baudrate = 500000  # 500kbps
        self.communication_status = "online"
        self.last_heartbeat = datetime.now()
        
        # 均衡功能
        self.balance_enabled = True
        self.balance_threshold = 0.03  # 30mV压差启动均衡
        self.balance_current = 0.1     # 100mA均衡电流
        self.balancing_cells = []      # 正在均衡的电芯
        
        # 保护参数
        self.protection_params = {
            "overvoltage": 3.65,       # 过压保护 V
            "undervoltage": 2.5,       # 欠压保护 V
            "overtemp": 55,            # 过温保护 ℃
            "undertemp": -20,          # 低温保护 ℃
            "overcurrent_charge": 1.0,  # 充电过流 C
            "overcurrent_discharge": 2.0  # 放电过流 C
        }
        
        # 状态估计算法
        self.soc_algorithm = "EKF"     # 扩展卡尔曼滤波
        self.soh_algorithm = "Dual-EKF"  # 双卡尔曼滤波
        
    def to_dict(self) -> Dict:
        return {
            "bms_id": self.bms_id,
            "version": self.version,
            "firmware": self.firmware,
            "voltage_channels": self.voltage_channels,
            "temp_channels": self.temp_channels,
            "communication_status": self.communication_status,
            "balance_enabled": self.balance_enabled,
            "balancing_cells": len(self.balancing_cells),
            "soc_algorithm": self.soc_algorithm,
            "soh_algorithm": self.soh_algorithm,
            "protection_params": self.protection_params
        }


class DCDCConverter:
    """
    DC-DC变换器模块
    每个Pack配置一个DCDC，实现Pack级功率控制和电压转换
    """
    
    def __init__(self, dcdc_id: str):
        self.dcdc_id = dcdc_id
        self.model = "DCDC-30kW"
        
        # 电气参数
        self.rated_power = 30.0       # 额定功率 kW
        self.max_power = 36.0         # 最大功率 kW (120%)
        self.efficiency = 0.965       # 转换效率
        
        self.input_voltage_range = (40, 60)   # 输入电压范围 V
        self.output_voltage_range = (45, 58)  # 输出电压范围 V
        
        # 运行状态
        self.status = "standby"       # standby, charging, discharging, fault
        self.current_power = 0.0      # 当前功率 kW
        self.input_voltage = 51.2     # 输入电压 V
        self.output_voltage = 51.2    # 输出电压 V
        self.input_current = 0.0      # 输入电流 A
        self.output_current = 0.0     # 输出电流 A
        
        # 热管理
        self.temperature = random.uniform(30, 40)
        self.fan_speed = 0            # 风扇转速 RPM
        
        # MPPT功能（可选）
        self.mppt_enabled = False
        
    def calculate_efficiency(self, load_ratio: float) -> float:
        """计算不同负载率下的效率"""
        # 效率曲线：低负载效率较低，中负载最优
        if load_ratio < 0.1:
            return 0.90
        elif load_ratio < 0.3:
            return 0.94
        elif load_ratio < 0.8:
            return 0.965
        else:
            return 0.95
    
    def to_dict(self) -> Dict:
        load_ratio = self.current_power / self.rated_power if self.rated_power > 0 else 0
        return {
            "dcdc_id": self.dcdc_id,
            "model": self.model,
            "rated_power": self.rated_power,
            "current_power": round(self.current_power, 2),
            "load_ratio": round(load_ratio * 100, 1),
            "efficiency": round(self.calculate_efficiency(load_ratio) * 100, 1),
            "input_voltage": round(self.input_voltage, 2),
            "output_voltage": round(self.output_voltage, 2),
            "status": self.status,
            "temperature": round(self.temperature, 1)
        }


# ==================== 电池层级结构 ====================

class BatteryCell:
    """电池单体（电芯）"""
    
    def __init__(self, cell_id: str, capacity: float = 280):
        self.cell_id = cell_id
        self.capacity = capacity  # Ah
        self.voltage_nominal = 3.2  # 标称电压 V
        self.voltage = 3.25  # 当前电压 V
        
        # 状态参数
        self.soc = random.uniform(0.45, 0.55)  # 荷电状态
        self.soh = random.uniform(0.95, 1.0)   # 健康状态
        self.soe = self.soc * self.soh         # 能量状态
        self.sop = random.uniform(0.85, 1.0)   # 功率状态
        self.temperature = random.uniform(23, 28)  # 温度 ℃
        
        # 内阻
        self.internal_resistance = random.uniform(0.5, 0.8)  # mΩ
        
        # 累计循环次数
        self.cycle_count = random.randint(0, 500)
        
        # 状态标志
        self.status = "normal"  # normal, warning, fault
        self.alarms = []
    
    def update_status(self):
        """更新电芯状态"""
        self.alarms = []
        
        # 温度检查
        if self.temperature > 45:
            self.status = "fault"
            self.alarms.append("过温告警")
        elif self.temperature > 40:
            self.status = "warning"
            self.alarms.append("温度偏高")
        elif self.temperature < 5:
            self.status = "warning"
            self.alarms.append("温度偏低")
        
        # 电压检查
        if self.voltage > 3.65:
            self.status = "fault"
            self.alarms.append("过压告警")
        elif self.voltage < 2.5:
            self.status = "fault"
            self.alarms.append("欠压告警")
        
        # SOH检查
        if self.soh < 0.8:
            self.status = "warning"
            self.alarms.append("SOH过低")
        
        if not self.alarms:
            self.status = "normal"
        
        return self.status
    
    def to_dict(self) -> Dict:
        self.update_status()
        return {
            "cell_id": self.cell_id,
            "voltage": round(self.voltage, 3),
            "soc": round(self.soc * 100, 1),
            "soh": round(self.soh * 100, 1),
            "soe": round(self.soe * 100, 1),
            "sop": round(self.sop * 100, 1),
            "temperature": round(self.temperature, 1),
            "internal_resistance": round(self.internal_resistance, 2),
            "cycle_count": self.cycle_count,
            "status": self.status,
            "alarms": self.alarms,
        }


class BatteryPack:
    """电池包（Pack）- 多个电芯串联 + BMS从机 + DCDC"""
    
    def __init__(self, pack_id: str, cells_count: int = 16):
        self.pack_id = pack_id
        self.cells_count = cells_count
        self.cells = [BatteryCell(f"{pack_id}-C{i+1:02d}") for i in range(cells_count)]
        
        # Pack级硬件组件
        self.bms = BMSModule(f"{pack_id}-BMS")  # BMS从机板
        self.dcdc = DCDCConverter(f"{pack_id}-DCDC")  # DCDC变换器
        
        # Pack级参数（使用聚合算法）
        self.voltage = StateAggregator.aggregate_voltage(
            [c.voltage for c in self.cells], connection="series"
        )
        self.capacity = self.cells[0].capacity  # 容量（串联容量不变）
        self.energy = self.voltage * self.capacity / 1000  # kWh
        
        # 更新DCDC输入电压
        self.dcdc.input_voltage = self.voltage
        self.dcdc.output_voltage = self.voltage
        
    def aggregate_from_cells(self) -> Dict:
        """从电芯级聚合到Pack级的状态"""
        socs = [c.soc for c in self.cells]
        sohs = [c.soh for c in self.cells]
        temps = [c.temperature for c in self.cells]
        statuses = [c.status for c in self.cells]
        
        return {
            "soc": StateAggregator.aggregate_soc(socs, method="weighted_mean"),
            "soh": StateAggregator.aggregate_soh(sohs, method="min"),
            "temperature": StateAggregator.aggregate_temperature(temps, method="max"),
            "status": StateAggregator.aggregate_status(statuses, method="strict"),
            "methods_used": {
                "soc": "weighted_mean（加权均值法）",
                "soh": "min（木桶效应-最小值法）",
                "temperature": "max（最大值法）",
                "status": "strict（严格法-故障优先）"
            }
        }
        
    def get_summary(self) -> Dict:
        agg = self.aggregate_from_cells()
        socs = [c.soc for c in self.cells]
        temps = [c.temperature for c in self.cells]
        sohs = [c.soh for c in self.cells]
        
        fault_count = sum(1 for c in self.cells if c.status == "fault")
        warning_count = sum(1 for c in self.cells if c.status == "warning")
        
        return {
            "pack_id": self.pack_id,
            "cells_count": self.cells_count,
            "voltage": round(self.voltage, 2),
            "soc_avg": round(agg["soc"] * 100, 1),
            "soc_min": round(min(socs) * 100, 1),
            "soc_max": round(max(socs) * 100, 1),
            "soc_diff": round((max(socs) - min(socs)) * 100, 2),
            "temp_avg": round(agg["temperature"][0], 1),
            "temp_min": round(agg["temperature"][2], 1),
            "temp_max": round(agg["temperature"][1], 1),
            "soh_avg": round(np.mean(sohs) * 100, 1),
            "soh_min": round(agg["soh"] * 100, 1),
            "fault_count": fault_count,
            "warning_count": warning_count,
            "status": agg["status"],
            # 硬件组件状态
            "bms_status": self.bms.communication_status,
            "bms_version": self.bms.version,
            "dcdc_status": self.dcdc.status,
            "dcdc_power": self.dcdc.current_power,
            "aggregation_methods": agg["methods_used"]
        }
    
    def get_all_cells(self) -> List[Dict]:
        return [cell.to_dict() for cell in self.cells]
    
    def get_hardware_info(self) -> Dict:
        """获取Pack硬件信息"""
        return {
            "pack_id": self.pack_id,
            "bms": self.bms.to_dict(),
            "dcdc": self.dcdc.to_dict()
        }


class ClusterBMS:
    """簇级BMS主控 - 管理多个Pack的BMS从机"""
    
    def __init__(self, cluster_id: str):
        self.bms_id = f"{cluster_id}-CBMS"
        self.version = "CBMS-V2.1"
        self.firmware = "FW-2024.12"
        
        # 通信管理
        self.can_nodes = []  # 管理的CAN节点（Pack BMS）
        self.communication_status = "online"
        
        # 簇级保护
        self.cluster_protection = {
            "max_charge_current": 500,   # A
            "max_discharge_current": 600,  # A
            "voltage_balance_threshold": 0.1  # V
        }
        
        # 簇级SOC/SOH估计
        self.soc_fusion_method = "Weighted Average"  # 加权平均法
        self.soh_fusion_method = "Minimum Value"     # 最小值法
        
    def to_dict(self) -> Dict:
        return {
            "bms_id": self.bms_id,
            "version": self.version,
            "communication_status": self.communication_status,
            "soc_fusion_method": self.soc_fusion_method,
            "soh_fusion_method": self.soh_fusion_method
        }


class BatteryCluster:
    """电池簇（Cluster）- 多个Pack串联 + 簇级BMS主控"""
    
    def __init__(self, cluster_id: str, packs_count: int = 4):
        self.cluster_id = cluster_id
        self.packs_count = packs_count
        self.packs = [BatteryPack(f"{cluster_id}-P{i+1:02d}") for i in range(packs_count)]
        
        # 簇级BMS主控
        self.cluster_bms = ClusterBMS(cluster_id)
        
        # 簇级参数（使用聚合算法）
        self.cells_count = sum(p.cells_count for p in self.packs)
        self.voltage = StateAggregator.aggregate_voltage(
            [p.voltage for p in self.packs], connection="series"
        )
        self.capacity = self.packs[0].capacity
        self.energy = self.voltage * self.capacity / 1000
        
        # 运行状态
        self.current = 0.0
        self.power = 0.0
        
    def aggregate_from_packs(self) -> Dict:
        """从Pack级聚合到簇级的状态"""
        pack_summaries = [p.get_summary() for p in self.packs]
        
        socs = [ps["soc_avg"] / 100 for ps in pack_summaries]
        sohs = [ps["soh_min"] / 100 for ps in pack_summaries]
        temps = [ps["temp_max"] for ps in pack_summaries]
        statuses = [ps["status"] for ps in pack_summaries]
        
        # 按能量加权
        energies = [p.energy for p in self.packs]
        total_energy = sum(energies)
        weights = [e / total_energy for e in energies] if total_energy > 0 else None
        
        return {
            "soc": StateAggregator.aggregate_soc(socs, weights=weights, method="weighted_mean"),
            "soh": StateAggregator.aggregate_soh(sohs, method="min"),
            "temperature": StateAggregator.aggregate_temperature(temps, method="max"),
            "status": StateAggregator.aggregate_status(statuses, method="strict"),
            "methods_used": {
                "soc": "weighted_mean（按能量加权）",
                "soh": "min（木桶效应）",
                "temperature": "max（关注最高温度）",
                "status": "strict（故障优先传递）"
            }
        }
        
    def get_summary(self) -> Dict:
        agg = self.aggregate_from_packs()
        pack_summaries = [p.get_summary() for p in self.packs]
        
        fault_count = sum(ps["fault_count"] for ps in pack_summaries)
        warning_count = sum(ps["warning_count"] for ps in pack_summaries)
        
        return {
            "cluster_id": self.cluster_id,
            "packs_count": self.packs_count,
            "cells_count": self.cells_count,
            "voltage": round(self.voltage, 2),
            "capacity": self.capacity,
            "energy": round(self.energy, 2),
            "soc_avg": round(agg["soc"] * 100, 1),
            "soh_min": round(agg["soh"] * 100, 1),
            "temp_avg": round(agg["temperature"][0], 1),
            "temp_max": round(agg["temperature"][1], 1),
            "current": round(self.current, 2),
            "power": round(self.power, 2),
            "cluster_bms_status": self.cluster_bms.communication_status,
            "fault_count": fault_count,
            "warning_count": warning_count,
            "status": agg["status"],
            "aggregation_methods": agg["methods_used"]
        }
    
    def get_all_packs(self) -> List[Dict]:
        return [pack.get_summary() for pack in self.packs]
    
    def get_hardware_topology(self) -> Dict:
        """获取簇的硬件拓扑"""
        return {
            "cluster_id": self.cluster_id,
            "cluster_bms": self.cluster_bms.to_dict(),
            "packs": [p.get_hardware_info() for p in self.packs]
        }


class PCSConverter:
    """储能变流器(PCS) - 箱级DC/AC变换"""
    
    def __init__(self, pcs_id: str):
        self.pcs_id = pcs_id
        self.model = "PCS-500kW"
        
        # 额定参数
        self.rated_power = 500.0      # kW
        self.rated_voltage_dc = 800   # V DC
        self.rated_voltage_ac = 400   # V AC
        self.frequency = 50           # Hz
        
        # 运行状态
        self.status = "standby"       # standby, charging, discharging, fault
        self.current_power = 0.0      # 当前功率 kW
        self.efficiency = 0.96
        
        # 并网参数
        self.grid_connected = True
        self.power_factor = 0.99
        
    def to_dict(self) -> Dict:
        return {
            "pcs_id": self.pcs_id,
            "model": self.model,
            "rated_power": self.rated_power,
            "current_power": round(self.current_power, 2),
            "status": self.status,
            "efficiency": round(self.efficiency * 100, 1),
            "grid_connected": self.grid_connected
        }


class EMS:
    """能量管理系统 - 箱级控制"""
    
    def __init__(self, ems_id: str):
        self.ems_id = ems_id
        self.version = "EMS-V4.0"
        
        # 控制策略
        self.control_mode = "auto"    # auto, manual, schedule
        self.optimization_target = "peak_shaving"  # peak_shaving, arbitrage, frequency_regulation
        
        # 通信
        self.communication_status = "online"
        
    def to_dict(self) -> Dict:
        return {
            "ems_id": self.ems_id,
            "version": self.version,
            "control_mode": self.control_mode,
            "optimization_target": self.optimization_target,
            "communication_status": self.communication_status
        }


class BatteryContainer:
    """储能集装箱（箱）- 多个簇并联 + PCS + EMS"""
    
    def __init__(self, container_id: str, clusters_count: int = 8):
        self.container_id = container_id
        self.clusters_count = clusters_count
        self.clusters = [BatteryCluster(f"{container_id}-CL{i+1:02d}") for i in range(clusters_count)]
        
        # 箱级设备
        self.pcs = PCSConverter(f"{container_id}-PCS")
        self.ems = EMS(f"{container_id}-EMS")
        
        # 箱级参数
        self.packs_count = sum(cl.packs_count for cl in self.clusters)
        self.cells_count = sum(cl.cells_count for cl in self.clusters)
        self.energy = sum(cl.energy for cl in self.clusters)
        
        # 环境监控
        self.ambient_temp = 25.0
        self.humidity = 45.0
        self.fire_alarm = False
        self.door_status = "closed"
        
    def aggregate_from_clusters(self) -> Dict:
        """从簇级聚合到箱级的状态"""
        cluster_summaries = [cl.get_summary() for cl in self.clusters]
        
        socs = [cs["soc_avg"] / 100 for cs in cluster_summaries]
        sohs = [cs["soh_min"] / 100 for cs in cluster_summaries]
        temps = [cs["temp_max"] for cs in cluster_summaries]
        statuses = [cs["status"] for cs in cluster_summaries]
        
        # 并联：按能量加权
        energies = [cl.energy for cl in self.clusters]
        total_energy = sum(energies)
        weights = [e / total_energy for e in energies] if total_energy > 0 else None
        
        return {
            "soc": StateAggregator.aggregate_soc(socs, weights=weights, method="weighted_mean"),
            "soh": StateAggregator.aggregate_soh(sohs, method="min"),
            "temperature": StateAggregator.aggregate_temperature(temps, method="max"),
            "status": StateAggregator.aggregate_status(statuses, method="voting"),
            "methods_used": {
                "soc": "weighted_mean（并联按能量加权）",
                "soh": "min（整体健康由最差决定）",
                "temperature": "max（安全关注最高温）",
                "status": "voting（投票法-30%告警阈值）"
            }
        }
        
    def get_summary(self) -> Dict:
        agg = self.aggregate_from_clusters()
        cluster_summaries = [cl.get_summary() for cl in self.clusters]
        
        fault_count = sum(cs["fault_count"] for cs in cluster_summaries)
        warning_count = sum(cs["warning_count"] for cs in cluster_summaries)
        
        return {
            "container_id": self.container_id,
            "clusters_count": self.clusters_count,
            "packs_count": self.packs_count,
            "cells_count": self.cells_count,
            "total_energy": round(self.energy, 2),
            "soc_avg": round(agg["soc"] * 100, 1),
            "soh_min": round(agg["soh"] * 100, 1),
            "temp_avg": round(agg["temperature"][0], 1),
            "temp_max": round(agg["temperature"][1], 1),
            "pcs_power": self.pcs.rated_power,
            "pcs_status": self.pcs.status,
            "current_power": round(self.pcs.current_power, 2),
            "ems_status": self.ems.communication_status,
            "ambient_temp": round(self.ambient_temp, 1),
            "humidity": round(self.humidity, 1),
            "fire_alarm": self.fire_alarm,
            "door_status": self.door_status,
            "fault_count": fault_count,
            "warning_count": warning_count,
            "status": agg["status"],
            "aggregation_methods": agg["methods_used"]
        }
    
    def get_all_clusters(self) -> List[Dict]:
        return [cluster.get_summary() for cluster in self.clusters]
    
    def get_hardware_topology(self) -> Dict:
        """获取箱的硬件拓扑"""
        return {
            "container_id": self.container_id,
            "pcs": self.pcs.to_dict(),
            "ems": self.ems.to_dict(),
            "clusters": [cl.get_hardware_topology() for cl in self.clusters]
        }


class SCADA:
    """站级监控系统"""
    
    def __init__(self, station_id: str):
        self.scada_id = f"{station_id}-SCADA"
        self.version = "SCADA-V5.0"
        
        # 数据采集
        self.polling_interval = 1.0   # 秒
        self.data_retention = 365     # 天
        
        # 通信
        self.protocols = ["IEC61850", "Modbus TCP", "IEC104"]
        self.communication_status = "online"
        
    def to_dict(self) -> Dict:
        return {
            "scada_id": self.scada_id,
            "version": self.version,
            "protocols": self.protocols,
            "communication_status": self.communication_status
        }


class EnergyStorageSystem:
    """储能电站系统 - 站级"""
    
    def __init__(self, containers_count: int = 2):
        self.station_id = "ESS-001"
        self.station_name = "微电网储能电站"
        self.containers_count = containers_count
        self.containers = [BatteryContainer(f"BOX{i+1:02d}") for i in range(containers_count)]
        
        # 站级设备
        self.scada = SCADA(self.station_id)
        
        # 统计
        self.total_clusters = sum(c.clusters_count for c in self.containers)
        self.total_packs = sum(c.packs_count for c in self.containers)
        self.total_cells = sum(c.cells_count for c in self.containers)
        self.total_energy = sum(c.energy for c in self.containers)
        
        # 系统状态
        self.grid_connected = True
        
    def aggregate_from_containers(self) -> Dict:
        """从箱级聚合到站级的状态"""
        container_summaries = [c.get_summary() for c in self.containers]
        
        socs = [cs["soc_avg"] / 100 for cs in container_summaries]
        sohs = [cs["soh_min"] / 100 for cs in container_summaries]
        temps = [cs["temp_max"] for cs in container_summaries]
        statuses = [cs["status"] for cs in container_summaries]
        
        # 按能量加权
        energies = [c.energy for c in self.containers]
        total_energy = sum(energies)
        weights = [e / total_energy for e in energies] if total_energy > 0 else None
        
        return {
            "soc": StateAggregator.aggregate_soc(socs, weights=weights, method="weighted_mean"),
            "soh": StateAggregator.aggregate_soh(sohs, method="min"),
            "temperature": StateAggregator.aggregate_temperature(temps, method="max"),
            "status": StateAggregator.aggregate_status(statuses, method="voting"),
            "methods_used": {
                "soc": "weighted_mean（全站按能量加权）",
                "soh": "min（站级健康由最差箱决定）",
                "temperature": "max（关注最高温度）",
                "status": "voting（多数决+故障优先）"
            }
        }
        
    def get_overview(self) -> Dict:
        agg = self.aggregate_from_containers()
        container_summaries = [c.get_summary() for c in self.containers]
        
        total_faults = sum(cs["fault_count"] for cs in container_summaries)
        total_warnings = sum(cs["warning_count"] for cs in container_summaries)
        
        return {
            "station_id": self.station_id,
            "station_name": self.station_name,
            "containers_count": self.containers_count,
            "clusters_count": self.total_clusters,
            "packs_count": self.total_packs,
            "cells_count": self.total_cells,
            "total_energy_kwh": round(self.total_energy, 2),
            "total_energy_mwh": round(self.total_energy / 1000, 3),
            "soc_avg": round(agg["soc"] * 100, 1),
            "soh_min": round(agg["soh"] * 100, 1),
            "temp_max": round(agg["temperature"][1], 1),
            "total_faults": total_faults,
            "total_warnings": total_warnings,
            "system_status": agg["status"],
            "grid_connected": self.grid_connected,
            "scada_status": self.scada.communication_status,
            "timestamp": datetime.now().isoformat(),
            "aggregation_methods": agg["methods_used"]
        }
    
    def get_all_containers(self) -> List[Dict]:
        return [container.get_summary() for container in self.containers]
    
    def get_container_detail(self, container_id: str) -> Optional[Dict]:
        for container in self.containers:
            if container.container_id == container_id:
                return {
                    "summary": container.get_summary(),
                    "clusters": container.get_all_clusters(),
                }
        return None
    
    def get_cluster_detail(self, container_id: str, cluster_id: str) -> Optional[Dict]:
        for container in self.containers:
            if container.container_id == container_id:
                for cluster in container.clusters:
                    if cluster.cluster_id == cluster_id:
                        return {
                            "summary": cluster.get_summary(),
                            "packs": cluster.get_all_packs(),
                        }
        return None
    
    def get_pack_cells(self, container_id: str, cluster_id: str, pack_id: str) -> Optional[List[Dict]]:
        for container in self.containers:
            if container.container_id == container_id:
                for cluster in container.clusters:
                    if cluster.cluster_id == cluster_id:
                        for pack in cluster.packs:
                            if pack.pack_id == pack_id:
                                return pack.get_all_cells()
        return None
    
    def get_cell_by_id(self, cell_id: str) -> Optional[Dict]:
        """根据电芯ID获取电芯详情"""
        for container in self.containers:
            for cluster in container.clusters:
                for pack in cluster.packs:
                    for cell in pack.cells:
                        if cell.cell_id == cell_id:
                            return cell.to_dict()
        return None
    
    def get_system_architecture(self) -> Dict:
        """获取系统架构信息"""
        return {
            "station": {
                "id": self.station_id,
                "name": self.station_name,
                "scada": self.scada.to_dict(),
                "total_containers": self.containers_count,
                "total_clusters": self.total_clusters,
                "total_packs": self.total_packs,
                "total_cells": self.total_cells,
                "total_energy_mwh": round(self.total_energy / 1000, 3)
            },
            "hierarchy": {
                "levels": ["站(Station)", "箱(Container)", "簇(Cluster)", "Pack", "电芯(Cell)"],
                "description": "五级层级结构，从站级到电芯级的树形拓扑",
                "structure": {
                    "station": f"1 × 站级SCADA",
                    "container": f"{self.containers_count} × 集装箱 (各含PCS+EMS)",
                    "cluster": f"每箱 8 × 簇 (各含簇级BMS)",
                    "pack": f"每簇 4 × Pack (各含BMS从机+DCDC)",
                    "cell": f"每Pack 16 × 电芯"
                }
            },
            "hardware_per_level": {
                "station": ["SCADA系统", "通信网关", "数据服务器"],
                "container": ["PCS储能变流器(500kW)", "EMS能量管理系统", "消防系统", "空调系统"],
                "cluster": ["簇级BMS主控", "接触器", "熔断器"],
                "pack": ["BMS从机板(V3.2)", "DCDC变换器(30kW)", "电压/温度采样"],
                "cell": ["LFP电芯(280Ah)", "采样线束"]
            },
            "aggregation_methods": StateAggregator.get_aggregation_info(),
            "data_flow": {
                "upward": "电芯→Pack→簇→箱→站（状态聚合）",
                "downward": "站→箱→簇→Pack→电芯（控制指令）",
                "protocols": {
                    "cell_to_pack": "硬件采样",
                    "pack_to_cluster": "CAN总线(500kbps)",
                    "cluster_to_container": "CAN/RS485",
                    "container_to_station": "以太网/光纤"
                }
            }
        }
    
    def get_pack_hardware(self, container_id: str, cluster_id: str, pack_id: str) -> Optional[Dict]:
        """获取Pack硬件详情"""
        for container in self.containers:
            if container.container_id == container_id:
                for cluster in container.clusters:
                    if cluster.cluster_id == cluster_id:
                        for pack in cluster.packs:
                            if pack.pack_id == pack_id:
                                return pack.get_hardware_info()
        return None


# 全局储能系统实例
storage_system = EnergyStorageSystem(containers_count=6)









