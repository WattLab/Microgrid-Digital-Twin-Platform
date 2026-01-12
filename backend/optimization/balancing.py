"""
电池均衡算法模块
包含被动均衡和主动均衡算法
支持单体间、Pack间、簇间、箱间多层级均衡
涵盖SOC不一致、SOH不一致、电压不一致的均衡

参考标准：
- IEC 62619: 锂电池安全要求
- GB/T 34131-2017: 电化学储能电站安全规范
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum
from dataclasses import dataclass


class BalanceType(Enum):
    """均衡类型"""
    PASSIVE = "passive"    # 被动均衡
    ACTIVE = "active"      # 主动均衡


class BalanceLevel(Enum):
    """均衡层级"""
    CELL = "cell"          # 单体间均衡
    PACK = "pack"          # Pack间均衡
    CLUSTER = "cluster"    # 簇间均衡
    CONTAINER = "container"  # 箱间均衡


class BalanceTarget(Enum):
    """均衡目标"""
    SOC = "soc"            # SOC均衡
    SOH = "soh"            # SOH均衡
    VOLTAGE = "voltage"    # 电压均衡


@dataclass
class BalanceCommand:
    """均衡控制命令"""
    target_id: str                    # 目标单元ID
    balance_type: BalanceType         # 均衡类型
    balance_level: BalanceLevel       # 均衡层级
    balance_target: BalanceTarget     # 均衡目标
    action: str                       # 动作: discharge/charge/transfer
    power: float                      # 均衡功率(W)
    duration: float                   # 持续时间(s)
    priority: int                     # 优先级(1-10)
    reason: str                       # 均衡原因


# ==================== 被动均衡算法 ====================

class PassiveBalancer:
    """
    被动均衡器
    
    原理：通过电阻放电的方式将高SOC/高电压单体的能量耗散为热能
    优点：结构简单、成本低、可靠性高
    缺点：能量损耗大、均衡速度慢、发热量大
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Parameters:
        -----------
        config : dict
            均衡器配置参数
        """
        self.config = config or self._default_config()
        
        # 均衡参数
        self.balance_current = self.config.get("balance_current", 0.1)  # 均衡电流 A
        self.balance_resistance = self.config.get("balance_resistance", 33)  # 均衡电阻 Ω
        
        # 触发阈值
        self.voltage_threshold = self.config.get("voltage_threshold", 0.03)  # 电压差阈值 V
        self.soc_threshold = self.config.get("soc_threshold", 0.05)  # SOC差阈值 (5%)
        
        # 保护阈值
        self.max_balance_temp = self.config.get("max_balance_temp", 55)  # 最高均衡温度
        self.min_cell_voltage = self.config.get("min_cell_voltage", 2.8)  # 最低电芯电压
        
        # 状态
        self.is_balancing = False
        self.balancing_cells = []
        
    def _default_config(self) -> Dict:
        return {
            "balance_current": 0.1,      # 100mA
            "balance_resistance": 33,     # 33Ω
            "voltage_threshold": 0.03,    # 30mV
            "soc_threshold": 0.05,        # 5%
            "max_balance_temp": 55,       # 55℃
            "min_cell_voltage": 2.8,      # 2.8V
            "balance_time_limit": 3600,   # 最大均衡时间1小时
        }
    
    def analyze_imbalance(self, cells: List[Dict], target: BalanceTarget) -> Dict:
        """
        分析单体不一致性
        
        Parameters:
        -----------
        cells : list
            电芯数据列表 [{voltage, soc, soh, temperature, cell_id}]
        target : BalanceTarget
            均衡目标(SOC/SOH/电压)
            
        Returns:
        --------
        dict: 不一致性分析结果
        """
        if not cells:
            return {"needs_balance": False}
        
        if target == BalanceTarget.VOLTAGE:
            values = [c.get("voltage", 3.2) for c in cells]
            threshold = self.voltage_threshold
        elif target == BalanceTarget.SOC:
            values = [c.get("soc", 50) / 100 for c in cells]
            threshold = self.soc_threshold
        else:  # SOH
            values = [c.get("soh", 100) / 100 for c in cells]
            threshold = 0.05  # SOH差异阈值5%
        
        min_val = min(values)
        max_val = max(values)
        avg_val = np.mean(values)
        std_val = np.std(values)
        diff = max_val - min_val
        
        # 找出需要均衡的单体（高于平均值）
        cells_to_balance = []
        for i, (cell, val) in enumerate(zip(cells, values)):
            if val > avg_val + threshold / 2:
                cells_to_balance.append({
                    "cell_id": cell.get("cell_id", f"cell_{i}"),
                    "value": val,
                    "deviation": val - avg_val,
                    "balance_needed": val - avg_val
                })
        
        return {
            "target": target.value,
            "min_value": round(min_val, 4),
            "max_value": round(max_val, 4),
            "avg_value": round(avg_val, 4),
            "std_value": round(std_val, 4),
            "difference": round(diff, 4),
            "threshold": threshold,
            "needs_balance": diff > threshold,
            "cells_to_balance": cells_to_balance,
            "balance_count": len(cells_to_balance),
            "severity": self._calculate_severity(diff, threshold),
        }
    
    def _calculate_severity(self, diff: float, threshold: float) -> str:
        """计算不一致性严重程度"""
        ratio = diff / threshold if threshold > 0 else 0
        if ratio < 1:
            return "normal"
        elif ratio < 2:
            return "mild"
        elif ratio < 3:
            return "moderate"
        else:
            return "severe"
    
    def generate_balance_commands(
        self,
        cells: List[Dict],
        target: BalanceTarget,
        level: BalanceLevel = BalanceLevel.CELL
    ) -> List[BalanceCommand]:
        """
        生成被动均衡命令
        
        Parameters:
        -----------
        cells : list
            电芯/单元数据列表
        target : BalanceTarget
            均衡目标
        level : BalanceLevel
            均衡层级
            
        Returns:
        --------
        list: 均衡命令列表
        """
        analysis = self.analyze_imbalance(cells, target)
        
        if not analysis["needs_balance"]:
            return []
        
        commands = []
        for cell_info in analysis["cells_to_balance"]:
            # 计算均衡时间
            if target == BalanceTarget.VOLTAGE:
                # 电压均衡：根据压差和均衡电流计算时间
                delta_v = cell_info["deviation"]
                capacity = 280  # Ah，假设容量
                # 估算需要释放的能量
                delta_soc = delta_v / 0.1  # 简化估算，0.1V对应约1%SOC
                balance_time = (delta_soc * capacity) / self.balance_current * 3600  # 秒
            elif target == BalanceTarget.SOC:
                # SOC均衡
                delta_soc = cell_info["deviation"]
                capacity = 280
                balance_time = (delta_soc * capacity) / self.balance_current * 3600
            else:
                # SOH均衡（通过调整使用策略实现）
                balance_time = 0  # 被动均衡不直接处理SOH
            
            if balance_time > 0:
                commands.append(BalanceCommand(
                    target_id=cell_info["cell_id"],
                    balance_type=BalanceType.PASSIVE,
                    balance_level=level,
                    balance_target=target,
                    action="discharge",
                    power=self.balance_current * cells[0].get("voltage", 3.2),
                    duration=min(balance_time, self.config.get("balance_time_limit", 3600)),
                    priority=5,
                    reason=f"{target.value}偏高，差值: {cell_info['deviation']:.4f}"
                ))
        
        return commands
    
    def execute_balance(self, commands: List[BalanceCommand]) -> Dict:
        """
        执行均衡操作（模拟）
        
        Returns:
        --------
        dict: 执行结果
        """
        results = []
        total_energy_dissipated = 0
        
        for cmd in commands:
            # 计算耗散能量
            energy = cmd.power * cmd.duration / 3600  # Wh
            total_energy_dissipated += energy
            
            results.append({
                "target_id": cmd.target_id,
                "action": cmd.action,
                "power": cmd.power,
                "duration": cmd.duration,
                "energy_dissipated": round(energy, 4),
                "status": "executed"
            })
        
        return {
            "balance_type": "passive",
            "commands_executed": len(commands),
            "total_energy_dissipated_wh": round(total_energy_dissipated, 2),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }


# ==================== 主动均衡算法 ====================

class ActiveBalancer:
    """
    主动均衡器
    
    原理：通过DC-DC变换器在电芯/Pack之间转移能量
    优点：能量效率高(85-95%)、均衡速度快
    缺点：结构复杂、成本高
    
    拓扑结构：
    1. 相邻单体均衡（Adjacent Cell-to-Cell）
    2. 电池组到单体（Pack-to-Cell）
    3. 单体到电池组（Cell-to-Pack）
    4. 多层级均衡（Multi-level）
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Parameters:
        -----------
        config : dict
            均衡器配置参数
        """
        self.config = config or self._default_config()
        
        # 均衡参数
        self.max_balance_current = self.config.get("max_balance_current", 5)  # 最大均衡电流 A
        self.dcdc_efficiency = self.config.get("dcdc_efficiency", 0.92)  # DC-DC效率
        
        # 触发阈值
        self.voltage_threshold = self.config.get("voltage_threshold", 0.02)  # 电压差阈值 V
        self.soc_threshold = self.config.get("soc_threshold", 0.03)  # SOC差阈值 (3%)
        self.soh_threshold = self.config.get("soh_threshold", 0.05)  # SOH差阈值 (5%)
        
        # 均衡策略
        self.topology = self.config.get("topology", "multi_level")  # 均衡拓扑
        
    def _default_config(self) -> Dict:
        return {
            "max_balance_current": 5,     # 5A
            "dcdc_efficiency": 0.92,       # 92%
            "voltage_threshold": 0.02,     # 20mV
            "soc_threshold": 0.03,         # 3%
            "soh_threshold": 0.05,         # 5%
            "topology": "multi_level",     # 多层级均衡
            "balance_power_limit": 500,    # 最大均衡功率 W
        }
    
    def analyze_multilevel_imbalance(
        self,
        station_data: Dict
    ) -> Dict:
        """
        多层级不一致性分析
        
        分析站级内各层级的不一致性：
        - 箱间不一致性
        - 簇间不一致性
        - Pack间不一致性
        - 单体间不一致性
        
        Parameters:
        -----------
        station_data : dict
            储能站数据结构
            
        Returns:
        --------
        dict: 多层级不一致性分析结果
        """
        result = {
            "container_level": {},
            "cluster_level": {},
            "pack_level": {},
            "cell_level": {},
            "summary": {}
        }
        
        # 箱级分析
        containers = station_data.get("containers", [])
        if containers:
            container_socs = [c.get("soc_avg", 50) for c in containers]
            container_voltages = [c.get("voltage", 0) for c in containers]
            
            result["container_level"] = {
                "soc_diff": round(max(container_socs) - min(container_socs), 2),
                "voltage_diff": round(max(container_voltages) - min(container_voltages), 2),
                "needs_balance": (max(container_socs) - min(container_socs)) > self.soc_threshold * 100
            }
        
        # 簇级分析（在每个箱内）
        cluster_analysis = []
        for container in containers:
            clusters = container.get("clusters", [])
            if clusters:
                cluster_socs = [cl.get("soc_avg", 50) for cl in clusters]
                cluster_analysis.append({
                    "container_id": container.get("container_id"),
                    "soc_diff": round(max(cluster_socs) - min(cluster_socs), 2),
                    "needs_balance": (max(cluster_socs) - min(cluster_socs)) > self.soc_threshold * 100
                })
        result["cluster_level"] = cluster_analysis
        
        # 计算总体严重程度
        total_imbalance_score = 0
        if containers:
            total_imbalance_score += result["container_level"].get("soc_diff", 0) * 0.3
        for cl in cluster_analysis:
            total_imbalance_score += cl.get("soc_diff", 0) * 0.2
        
        result["summary"] = {
            "overall_imbalance_score": round(total_imbalance_score, 2),
            "severity": "severe" if total_imbalance_score > 10 else "moderate" if total_imbalance_score > 5 else "mild",
            "recommendation": self._get_balance_recommendation(total_imbalance_score)
        }
        
        return result
    
    def _get_balance_recommendation(self, score: float) -> str:
        """获取均衡建议"""
        if score > 10:
            return "立即启动多层级主动均衡"
        elif score > 5:
            return "建议在下次充放电周期内启动均衡"
        elif score > 2:
            return "可在低谷时段进行均衡维护"
        else:
            return "一致性良好，暂不需要均衡"
    
    def generate_soc_balance_commands(
        self,
        units: List[Dict],
        level: BalanceLevel
    ) -> List[BalanceCommand]:
        """
        生成SOC均衡命令
        
        策略：高SOC单元向低SOC单元转移能量
        
        Parameters:
        -----------
        units : list
            单元数据列表
        level : BalanceLevel
            均衡层级
        """
        if len(units) < 2:
            return []
        
        commands = []
        
        # 计算平均SOC
        socs = [u.get("soc", 50) / 100 for u in units]
        avg_soc = np.mean(socs)
        
        # 找出高SOC和低SOC单元
        high_soc_units = [(i, u, s) for i, (u, s) in enumerate(zip(units, socs)) if s > avg_soc + self.soc_threshold / 2]
        low_soc_units = [(i, u, s) for i, (u, s) in enumerate(zip(units, socs)) if s < avg_soc - self.soc_threshold / 2]
        
        # 配对进行能量转移
        for (hi_idx, hi_unit, hi_soc), (lo_idx, lo_unit, lo_soc) in zip(high_soc_units, low_soc_units):
            delta_soc = hi_soc - lo_soc
            
            if delta_soc > self.soc_threshold:
                # 计算需要转移的能量
                capacity = hi_unit.get("capacity", 280)  # Ah
                voltage = hi_unit.get("voltage", 51.2)  # V
                energy_to_transfer = delta_soc / 2 * capacity * voltage  # Wh
                
                # 考虑效率损失
                actual_transfer = energy_to_transfer * self.dcdc_efficiency
                
                # 计算均衡时间
                balance_power = min(self.config.get("balance_power_limit", 500), 
                                   self.max_balance_current * voltage)
                balance_time = energy_to_transfer * 3600 / balance_power  # 秒
                
                # 生成转移命令
                commands.append(BalanceCommand(
                    target_id=hi_unit.get("id", f"unit_{hi_idx}"),
                    balance_type=BalanceType.ACTIVE,
                    balance_level=level,
                    balance_target=BalanceTarget.SOC,
                    action="transfer",
                    power=balance_power,
                    duration=balance_time,
                    priority=7,
                    reason=f"SOC均衡: 从{hi_unit.get('id')}({hi_soc*100:.1f}%)转移到{lo_unit.get('id')}({lo_soc*100:.1f}%)"
                ))
        
        return commands
    
    def generate_soh_balance_commands(
        self,
        units: List[Dict],
        level: BalanceLevel
    ) -> List[BalanceCommand]:
        """
        生成SOH均衡命令
        
        SOH均衡策略：
        1. 对低SOH单元减少使用强度
        2. 通过调整充放电功率分配实现SOH均衡
        3. 优先使用高SOH单元承担峰值功率
        
        Parameters:
        -----------
        units : list
            单元数据列表
        level : BalanceLevel
            均衡层级
        """
        if len(units) < 2:
            return []
        
        commands = []
        
        # 计算SOH统计
        sohs = [u.get("soh", 100) / 100 for u in units]
        avg_soh = np.mean(sohs)
        min_soh = min(sohs)
        max_soh = max(sohs)
        
        if max_soh - min_soh < self.soh_threshold:
            return []  # SOH差异在阈值内，无需均衡
        
        # 对低SOH单元生成保护命令
        for i, (unit, soh) in enumerate(zip(units, sohs)):
            if soh < avg_soh - self.soh_threshold / 2:
                # 计算功率限制比例
                derating_ratio = soh / avg_soh
                
                commands.append(BalanceCommand(
                    target_id=unit.get("id", f"unit_{i}"),
                    balance_type=BalanceType.ACTIVE,
                    balance_level=level,
                    balance_target=BalanceTarget.SOH,
                    action="derate",  # 降额运行
                    power=derating_ratio * unit.get("rated_power", 100),
                    duration=3600 * 24,  # 持续一天
                    priority=8,
                    reason=f"SOH偏低({soh*100:.1f}%)，降额至{derating_ratio*100:.0f}%功率运行以延长寿命"
                ))
        
        return commands
    
    def generate_voltage_balance_commands(
        self,
        cells: List[Dict],
        level: BalanceLevel = BalanceLevel.CELL
    ) -> List[BalanceCommand]:
        """
        生成电压均衡命令
        
        电压均衡策略：
        1. 快速响应，优先级最高
        2. 防止过充/过放
        
        Parameters:
        -----------
        cells : list
            电芯数据列表
        level : BalanceLevel
            均衡层级
        """
        if len(cells) < 2:
            return []
        
        commands = []
        
        voltages = [c.get("voltage", 3.2) for c in cells]
        avg_voltage = np.mean(voltages)
        min_voltage = min(voltages)
        max_voltage = max(voltages)
        
        if max_voltage - min_voltage < self.voltage_threshold:
            return []
        
        # 高电压单体转移能量到低电压单体
        for i, (cell, voltage) in enumerate(zip(cells, voltages)):
            deviation = voltage - avg_voltage
            
            if abs(deviation) > self.voltage_threshold / 2:
                if deviation > 0:
                    action = "discharge"  # 高电压放电
                else:
                    action = "charge"  # 低电压充电（从其他单体获取能量）
                
                # 计算均衡功率和时间
                balance_current = min(self.max_balance_current, abs(deviation) * 10)  # 简化计算
                balance_power = balance_current * voltage
                balance_time = abs(deviation) / 0.001 * 60  # 估算时间
                
                commands.append(BalanceCommand(
                    target_id=cell.get("cell_id", f"cell_{i}"),
                    balance_type=BalanceType.ACTIVE,
                    balance_level=level,
                    balance_target=BalanceTarget.VOLTAGE,
                    action=action,
                    power=balance_power,
                    duration=balance_time,
                    priority=9,  # 电压均衡优先级最高
                    reason=f"电压偏差: {deviation*1000:.1f}mV，执行{action}操作"
                ))
        
        return commands
    
    def execute_balance(self, commands: List[BalanceCommand]) -> Dict:
        """
        执行主动均衡操作（模拟）
        
        Returns:
        --------
        dict: 执行结果
        """
        results = []
        total_energy_transferred = 0
        total_energy_loss = 0
        
        for cmd in commands:
            # 计算转移能量和损失
            energy = cmd.power * cmd.duration / 3600  # Wh
            if cmd.action == "transfer":
                energy_loss = energy * (1 - self.dcdc_efficiency)
                actual_transferred = energy * self.dcdc_efficiency
            else:
                energy_loss = 0
                actual_transferred = energy
            
            total_energy_transferred += actual_transferred
            total_energy_loss += energy_loss
            
            results.append({
                "target_id": cmd.target_id,
                "balance_target": cmd.balance_target.value,
                "action": cmd.action,
                "power": cmd.power,
                "duration": cmd.duration,
                "energy_transferred": round(actual_transferred, 4),
                "energy_loss": round(energy_loss, 4),
                "status": "executed"
            })
        
        efficiency = (total_energy_transferred / (total_energy_transferred + total_energy_loss) 
                     if total_energy_transferred > 0 else 0)
        
        return {
            "balance_type": "active",
            "commands_executed": len(commands),
            "total_energy_transferred_wh": round(total_energy_transferred, 2),
            "total_energy_loss_wh": round(total_energy_loss, 2),
            "overall_efficiency": round(efficiency * 100, 1),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }


# ==================== 多层级均衡管理器 ====================

class MultilevelBalanceManager:
    """
    多层级均衡管理器
    
    实现单体间、Pack间、簇间、箱间的协调均衡
    支持SOC、SOH、电压三种均衡目标
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # 初始化均衡器
        self.passive_balancer = PassiveBalancer(self.config.get("passive", {}))
        self.active_balancer = ActiveBalancer(self.config.get("active", {}))
        
        # 均衡策略
        self.strategy = self.config.get("strategy", "hybrid")  # passive, active, hybrid
        
        # 均衡优先级
        self.level_priority = {
            BalanceLevel.CELL: 1,       # 最先均衡
            BalanceLevel.PACK: 2,
            BalanceLevel.CLUSTER: 3,
            BalanceLevel.CONTAINER: 4   # 最后均衡
        }
        
        # 目标优先级
        self.target_priority = {
            BalanceTarget.VOLTAGE: 1,   # 最高优先级（安全相关）
            BalanceTarget.SOC: 2,
            BalanceTarget.SOH: 3        # 最低优先级（长期优化）
        }
        
    def analyze_system_imbalance(self, system_data: Dict) -> Dict:
        """
        分析整个储能系统的不一致性
        
        Parameters:
        -----------
        system_data : dict
            储能系统完整数据
            
        Returns:
        --------
        dict: 全系统不一致性分析
        """
        analysis = {
            "container_imbalance": [],
            "cluster_imbalance": [],
            "pack_imbalance": [],
            "cell_imbalance": [],
            "overall_score": 0,
            "recommendations": []
        }
        
        containers = system_data.get("containers", [])
        
        # 箱级分析
        if len(containers) > 1:
            container_socs = [c.get("soc_avg", 50) for c in containers]
            container_sohs = [c.get("soh_min", 100) for c in containers]
            
            soc_diff = max(container_socs) - min(container_socs)
            soh_diff = max(container_sohs) - min(container_sohs)
            
            analysis["container_imbalance"] = {
                "level": "container",
                "soc_difference": round(soc_diff, 2),
                "soh_difference": round(soh_diff, 2),
                "needs_soc_balance": soc_diff > 5,  # 5%阈值
                "needs_soh_balance": soh_diff > 5
            }
            
            if soc_diff > 5:
                analysis["recommendations"].append({
                    "level": "container",
                    "action": "active_balance",
                    "reason": f"箱间SOC差异过大({soc_diff:.1f}%)，建议启动主动均衡"
                })
        
        # 计算综合评分
        scores = []
        if analysis["container_imbalance"]:
            scores.append(analysis["container_imbalance"].get("soc_difference", 0))
        
        analysis["overall_score"] = np.mean(scores) if scores else 0
        analysis["system_status"] = self._get_system_status(analysis["overall_score"])
        
        return analysis
    
    def _get_system_status(self, score: float) -> str:
        """获取系统均衡状态"""
        if score < 2:
            return "excellent"
        elif score < 5:
            return "good"
        elif score < 10:
            return "fair"
        else:
            return "poor"
    
    def generate_balance_plan(
        self,
        system_data: Dict,
        targets: List[BalanceTarget] = None
    ) -> Dict:
        """
        生成多层级均衡计划
        
        Parameters:
        -----------
        system_data : dict
            储能系统数据
        targets : list
            均衡目标列表，默认全部
            
        Returns:
        --------
        dict: 均衡计划
        """
        if targets is None:
            targets = [BalanceTarget.VOLTAGE, BalanceTarget.SOC, BalanceTarget.SOH]
        
        plan = {
            "analysis": self.analyze_system_imbalance(system_data),
            "commands": {
                "cell_level": [],
                "pack_level": [],
                "cluster_level": [],
                "container_level": []
            },
            "schedule": [],
            "estimated_duration": 0,
            "estimated_energy_cost": 0
        }
        
        # 按优先级生成命令
        for target in sorted(targets, key=lambda t: self.target_priority.get(t, 99)):
            # 对每个层级生成均衡命令
            containers = system_data.get("containers", [])
            
            # 箱级均衡
            if len(containers) > 1:
                container_units = [{
                    "id": c.get("container_id"),
                    "soc": c.get("soc_avg", 50),
                    "soh": c.get("soh_min", 100),
                    "voltage": c.get("voltage", 0),
                    "capacity": c.get("capacity", 0),
                    "rated_power": c.get("pcs_power", 500)
                } for c in containers]
                
                if target == BalanceTarget.SOC:
                    cmds = self.active_balancer.generate_soc_balance_commands(
                        container_units, BalanceLevel.CONTAINER
                    )
                    plan["commands"]["container_level"].extend(cmds)
                elif target == BalanceTarget.SOH:
                    cmds = self.active_balancer.generate_soh_balance_commands(
                        container_units, BalanceLevel.CONTAINER
                    )
                    plan["commands"]["container_level"].extend(cmds)
        
        # 计算估计时间和能耗
        all_commands = []
        for level_cmds in plan["commands"].values():
            all_commands.extend(level_cmds)
        
        if all_commands:
            plan["estimated_duration"] = max(cmd.duration for cmd in all_commands)
            plan["estimated_energy_cost"] = sum(cmd.power * cmd.duration / 3600 for cmd in all_commands)
        
        # 生成调度计划
        plan["schedule"] = self._generate_schedule(all_commands)
        
        return plan
    
    def _generate_schedule(self, commands: List[BalanceCommand]) -> List[Dict]:
        """生成均衡调度计划"""
        if not commands:
            return []
        
        # 按优先级排序
        sorted_commands = sorted(commands, key=lambda c: c.priority, reverse=True)
        
        schedule = []
        current_time = 0
        
        for cmd in sorted_commands:
            schedule.append({
                "start_time": current_time,
                "end_time": current_time + cmd.duration,
                "target_id": cmd.target_id,
                "action": cmd.action,
                "power": cmd.power,
                "priority": cmd.priority,
                "reason": cmd.reason
            })
            # 简化：串行执行（实际可并行）
            current_time += cmd.duration
        
        return schedule
    
    def execute_balance_plan(self, plan: Dict) -> Dict:
        """
        执行均衡计划
        
        Returns:
        --------
        dict: 执行结果
        """
        results = {
            "cell_level": [],
            "pack_level": [],
            "cluster_level": [],
            "container_level": [],
            "summary": {}
        }
        
        total_commands = 0
        total_energy = 0
        
        # 执行各层级均衡
        for level in ["cell_level", "pack_level", "cluster_level", "container_level"]:
            commands = plan["commands"].get(level, [])
            if commands:
                if self.strategy in ["active", "hybrid"]:
                    level_result = self.active_balancer.execute_balance(commands)
                else:
                    level_result = self.passive_balancer.execute_balance(commands)
                
                results[level] = level_result
                total_commands += level_result.get("commands_executed", 0)
                total_energy += level_result.get("total_energy_transferred_wh", 0)
        
        results["summary"] = {
            "total_commands_executed": total_commands,
            "total_energy_wh": round(total_energy, 2),
            "execution_time": datetime.now().isoformat(),
            "status": "completed"
        }
        
        return results
    
    def get_balance_status(self) -> Dict:
        """获取均衡器状态"""
        return {
            "strategy": self.strategy,
            "passive_balancer": {
                "balance_current": self.passive_balancer.balance_current,
                "voltage_threshold": self.passive_balancer.voltage_threshold,
                "soc_threshold": self.passive_balancer.soc_threshold,
                "is_balancing": self.passive_balancer.is_balancing
            },
            "active_balancer": {
                "max_balance_current": self.active_balancer.max_balance_current,
                "dcdc_efficiency": self.active_balancer.dcdc_efficiency,
                "topology": self.active_balancer.topology
            },
            "level_priority": {k.value: v for k, v in self.level_priority.items()},
            "target_priority": {k.value: v for k, v in self.target_priority.items()}
        }


# ==================== 均衡效果评估 ====================

class BalanceEvaluator:
    """均衡效果评估器"""
    
    @staticmethod
    def evaluate_balance_effect(before: Dict, after: Dict) -> Dict:
        """
        评估均衡效果
        
        Parameters:
        -----------
        before : dict
            均衡前的状态
        after : dict
            均衡后的状态
            
        Returns:
        --------
        dict: 评估结果
        """
        # SOC一致性改善
        soc_improvement = 0
        if "soc_diff" in before and "soc_diff" in after:
            soc_improvement = before["soc_diff"] - after["soc_diff"]
        
        # 电压一致性改善
        voltage_improvement = 0
        if "voltage_diff" in before and "voltage_diff" in after:
            voltage_improvement = before["voltage_diff"] - after["voltage_diff"]
        
        return {
            "soc_improvement": round(soc_improvement, 2),
            "voltage_improvement_mv": round(voltage_improvement * 1000, 1),
            "balance_efficiency": round((soc_improvement + voltage_improvement * 100) / 2, 1),
            "is_effective": soc_improvement > 0 or voltage_improvement > 0
        }
    
    @staticmethod
    def calculate_balance_kpi(history: List[Dict]) -> Dict:
        """
        计算均衡KPI指标
        
        Parameters:
        -----------
        history : list
            历史均衡记录
            
        Returns:
        --------
        dict: KPI指标
        """
        if not history:
            return {"message": "无均衡历史记录"}
        
        total_balance_count = len(history)
        total_energy_consumed = sum(h.get("energy_consumed", 0) for h in history)
        avg_balance_time = np.mean([h.get("duration", 0) for h in history])
        
        # 成功率
        success_count = sum(1 for h in history if h.get("success", False))
        success_rate = success_count / total_balance_count * 100
        
        return {
            "total_balance_count": total_balance_count,
            "success_rate": round(success_rate, 1),
            "total_energy_consumed_kwh": round(total_energy_consumed / 1000, 2),
            "avg_balance_time_minutes": round(avg_balance_time / 60, 1),
            "balance_frequency_per_day": round(total_balance_count / 30, 2)  # 假设30天
        }


# 创建全局均衡管理器实例
balance_manager = MultilevelBalanceManager()


