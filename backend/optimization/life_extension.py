"""
电池延寿策略模块
包含多维度的电池寿命优化策略

延寿策略包括：
1. 温度优化策略 - 最佳工作温度管理
2. DOD优化策略 - 放电深度优化
3. 倍率优化策略 - 充放电倍率控制
4. SOC管理策略 - 避免极端SOC状态
5. 循环优化策略 - 减少不必要循环
6. 热管理策略 - 预热与冷却控制
7. 充电策略优化 - 阶梯/脉冲充电
8. 电流波纹控制 - 减少纹波损伤
9. 日历老化管理 - 存储条件优化
10. 负载均衡策略 - 功率分配优化

参考标准：
- IEC 62620: 锂电池二次电池性能测试
- GB/T 36276-2018: 电力储能用锂离子电池
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import math


class LifeOptimizationTarget(Enum):
    """寿命优化目标"""
    TEMPERATURE = "temperature"      # 温度优化
    DOD = "dod"                      # 放电深度优化
    C_RATE = "c_rate"               # 倍率优化
    SOC = "soc"                      # SOC管理
    CYCLE = "cycle"                  # 循环优化
    THERMAL = "thermal"              # 热管理
    CHARGING = "charging"            # 充电策略
    RIPPLE = "ripple"               # 电流波纹
    CALENDAR = "calendar"            # 日历老化
    LOAD_BALANCE = "load_balance"   # 负载均衡


@dataclass
class LifeOptimizationCommand:
    """寿命优化命令"""
    target: LifeOptimizationTarget    # 优化目标
    unit_id: str                      # 目标单元ID
    action: str                       # 动作
    parameters: Dict                  # 参数
    priority: int                     # 优先级(1-10)
    expected_life_gain: float        # 预期寿命增益(%)
    reason: str                       # 优化原因


# ==================== 温度优化策略 ====================

class TemperatureOptimizer:
    """
    温度优化器
    
    原理：锂电池的老化速率与温度呈Arrhenius关系
    最佳工作温度范围：20-35℃
    高温加速老化，低温降低容量和功率
    
    策略：
    1. 预热策略：低温时先加热后使用
    2. 冷却策略：高温时降低功率并加强散热
    3. 温度均衡：减少温度梯度
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        
        # 温度阈值
        self.optimal_temp_min = self.config.get("optimal_temp_min", 20)  # 最佳温度下限
        self.optimal_temp_max = self.config.get("optimal_temp_max", 35)  # 最佳温度上限
        self.warning_temp_high = self.config.get("warning_temp_high", 45)  # 高温警告
        self.critical_temp_high = self.config.get("critical_temp_high", 55)  # 高温临界
        self.warning_temp_low = self.config.get("warning_temp_low", 5)  # 低温警告
        self.critical_temp_low = self.config.get("critical_temp_low", 0)  # 低温临界
        
        # Arrhenius老化模型参数
        self.activation_energy = 50000  # J/mol，活化能
        self.gas_constant = 8.314  # J/(mol·K)
        self.reference_temp = 298  # K (25℃)
        
    def _default_config(self) -> Dict:
        return {
            "optimal_temp_min": 20,
            "optimal_temp_max": 35,
            "warning_temp_high": 45,
            "critical_temp_high": 55,
            "warning_temp_low": 5,
            "critical_temp_low": 0,
            "preheat_power": 1000,  # 预热功率 W
            "cooling_power": 2000,  # 冷却功率 W
        }
    
    def calculate_aging_factor(self, temperature: float) -> float:
        """
        计算温度对老化的影响因子
        
        基于Arrhenius方程：k = A * exp(-Ea/RT)
        相对于25℃的老化加速因子
        
        Parameters:
        -----------
        temperature : float
            温度 (℃)
            
        Returns:
        --------
        float: 老化加速因子 (1.0表示正常，>1表示加速老化)
        """
        temp_k = temperature + 273.15
        ref_temp_k = self.reference_temp
        
        # Arrhenius加速因子
        factor = np.exp(
            (self.activation_energy / self.gas_constant) * 
            (1/ref_temp_k - 1/temp_k)
        )
        
        return round(factor, 3)
    
    def analyze_temperature_status(self, temperatures: List[float]) -> Dict:
        """
        分析温度状态
        
        Parameters:
        -----------
        temperatures : list
            各单元温度列表
            
        Returns:
        --------
        dict: 温度分析结果
        """
        if not temperatures:
            return {"status": "unknown"}
        
        avg_temp = np.mean(temperatures)
        max_temp = max(temperatures)
        min_temp = min(temperatures)
        temp_diff = max_temp - min_temp
        
        # 判断状态
        if max_temp >= self.critical_temp_high or min_temp <= self.critical_temp_low:
            status = "critical"
        elif max_temp >= self.warning_temp_high or min_temp <= self.warning_temp_low:
            status = "warning"
        elif self.optimal_temp_min <= avg_temp <= self.optimal_temp_max:
            status = "optimal"
        else:
            status = "suboptimal"
        
        # 计算平均老化因子
        aging_factors = [self.calculate_aging_factor(t) for t in temperatures]
        avg_aging_factor = np.mean(aging_factors)
        
        return {
            "status": status,
            "avg_temperature": round(avg_temp, 1),
            "max_temperature": round(max_temp, 1),
            "min_temperature": round(min_temp, 1),
            "temperature_difference": round(temp_diff, 1),
            "avg_aging_factor": round(avg_aging_factor, 3),
            "optimal_range": f"{self.optimal_temp_min}-{self.optimal_temp_max}℃",
            "recommendations": self._get_temp_recommendations(avg_temp, max_temp, min_temp, temp_diff)
        }
    
    def _get_temp_recommendations(
        self, 
        avg_temp: float, 
        max_temp: float, 
        min_temp: float,
        temp_diff: float
    ) -> List[str]:
        """获取温度优化建议"""
        recommendations = []
        
        if max_temp > self.warning_temp_high:
            recommendations.append(f"高温警告：最高温度{max_temp:.1f}℃，建议降低功率或加强冷却")
        
        if min_temp < self.warning_temp_low:
            recommendations.append(f"低温警告：最低温度{min_temp:.1f}℃，建议预热后再使用")
        
        if temp_diff > 10:
            recommendations.append(f"温差过大：{temp_diff:.1f}℃，建议检查热管理系统")
        
        if avg_temp > self.optimal_temp_max:
            recommendations.append("平均温度偏高，建议提高冷却效率")
        elif avg_temp < self.optimal_temp_min:
            recommendations.append("平均温度偏低，可开启预热功能")
        
        if not recommendations:
            recommendations.append("温度状态良好")
        
        return recommendations
    
    def generate_thermal_commands(self, temperatures: List[Dict]) -> List[LifeOptimizationCommand]:
        """
        生成热管理命令
        
        Parameters:
        -----------
        temperatures : list
            温度数据 [{unit_id, temperature}]
            
        Returns:
        --------
        list: 热管理命令
        """
        commands = []
        
        for unit in temperatures:
            temp = unit.get("temperature", 25)
            unit_id = unit.get("unit_id", "unknown")
            
            if temp > self.warning_temp_high:
                # 高温：开启冷却，降低功率
                derating = max(0.3, 1 - (temp - self.warning_temp_high) * 0.05)
                commands.append(LifeOptimizationCommand(
                    target=LifeOptimizationTarget.TEMPERATURE,
                    unit_id=unit_id,
                    action="cooling",
                    parameters={
                        "cooling_power": self.config.get("cooling_power", 2000),
                        "power_derating": derating,
                        "target_temperature": self.optimal_temp_max
                    },
                    priority=9,
                    expected_life_gain=5.0,
                    reason=f"温度{temp:.1f}℃过高，启动冷却并降额至{derating*100:.0f}%"
                ))
            
            elif temp < self.warning_temp_low:
                # 低温：开启预热
                commands.append(LifeOptimizationCommand(
                    target=LifeOptimizationTarget.TEMPERATURE,
                    unit_id=unit_id,
                    action="preheat",
                    parameters={
                        "preheat_power": self.config.get("preheat_power", 1000),
                        "target_temperature": self.optimal_temp_min,
                        "max_preheat_time": 1800  # 30分钟
                    },
                    priority=8,
                    expected_life_gain=3.0,
                    reason=f"温度{temp:.1f}℃过低，启动预热"
                ))
        
        return commands


# ==================== DOD优化策略 ====================

class DODOptimizer:
    """
    放电深度(DOD)优化器
    
    原理：深度放电会加速电池老化
    DOD vs 循环寿命关系（磷酸铁锂）：
    - 100% DOD: ~2000次循环
    - 80% DOD: ~3500次循环
    - 50% DOD: ~6000次循环
    - 30% DOD: ~10000次循环
    
    策略：
    1. 限制最大DOD
    2. 浅充浅放延长寿命
    3. 根据SOH调整DOD限制
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        
        # DOD限制
        self.max_dod = self.config.get("max_dod", 0.8)  # 默认最大80% DOD
        self.optimal_dod = self.config.get("optimal_dod", 0.6)  # 最佳DOD 60%
        self.min_soc = self.config.get("min_soc", 0.1)  # 最小SOC
        self.max_soc = self.config.get("max_soc", 0.9)  # 最大SOC
        
        # DOD-循环寿命模型参数（指数模型）
        self.cycle_life_factor = 2000  # 100% DOD时的循环次数
        self.dod_exponent = -1.5  # DOD指数
        
    def _default_config(self) -> Dict:
        return {
            "max_dod": 0.8,
            "optimal_dod": 0.6,
            "min_soc": 0.1,
            "max_soc": 0.9,
            "soh_threshold": 0.8,  # SOH低于此值时更保守
        }
    
    def calculate_cycle_life(self, dod: float) -> int:
        """
        计算预期循环寿命
        
        基于经验公式：N = k * DOD^n
        
        Parameters:
        -----------
        dod : float
            放电深度 (0-1)
            
        Returns:
        --------
        int: 预期循环次数
        """
        if dod <= 0 or dod > 1:
            return 0
        
        # 修正的循环寿命公式
        cycles = self.cycle_life_factor * (dod ** self.dod_exponent)
        return int(min(cycles, 50000))  # 上限5万次
    
    def calculate_life_loss_per_cycle(self, dod: float) -> float:
        """
        计算每次循环的寿命损耗
        
        Parameters:
        -----------
        dod : float
            放电深度
            
        Returns:
        --------
        float: 寿命损耗百分比
        """
        cycle_life = self.calculate_cycle_life(dod)
        if cycle_life <= 0:
            return 100
        return 100 / cycle_life
    
    def recommend_dod_limit(self, current_soh: float, usage_scenario: str = "normal") -> Dict:
        """
        根据SOH和使用场景推荐DOD限制
        
        Parameters:
        -----------
        current_soh : float
            当前SOH (0-1)
        usage_scenario : str
            使用场景: normal, peak_shaving, frequency_regulation
            
        Returns:
        --------
        dict: DOD推荐配置
        """
        base_dod = self.max_dod
        
        # 根据SOH调整
        if current_soh < 0.8:
            base_dod = min(base_dod, 0.7)  # SOH低时更保守
        if current_soh < 0.7:
            base_dod = min(base_dod, 0.6)
        
        # 根据场景调整
        scenario_adjustments = {
            "normal": 0,
            "peak_shaving": -0.1,  # 削峰填谷可以更深度使用
            "frequency_regulation": 0.1,  # 调频应用保守使用
            "backup": 0.2  # 备用电源更保守
        }
        
        adjusted_dod = base_dod + scenario_adjustments.get(usage_scenario, 0)
        adjusted_dod = max(0.3, min(0.9, adjusted_dod))
        
        # 计算预期寿命
        expected_cycles = self.calculate_cycle_life(adjusted_dod)
        life_loss_per_cycle = self.calculate_life_loss_per_cycle(adjusted_dod)
        
        return {
            "recommended_max_dod": round(adjusted_dod, 2),
            "recommended_soc_range": {
                "min": round(1 - adjusted_dod + self.min_soc, 2),
                "max": self.max_soc
            },
            "expected_cycle_life": expected_cycles,
            "life_loss_per_cycle_percent": round(life_loss_per_cycle, 4),
            "current_soh": round(current_soh, 3),
            "usage_scenario": usage_scenario,
            "optimization_suggestion": self._get_dod_suggestion(adjusted_dod, current_soh)
        }
    
    def _get_dod_suggestion(self, dod: float, soh: float) -> str:
        """获取DOD优化建议"""
        if soh < 0.7:
            return f"SOH较低({soh*100:.1f}%)，建议限制DOD在{dod*100:.0f}%以下，延长剩余寿命"
        elif dod > 0.8:
            return f"DOD设置较高({dod*100:.0f}%)，可有效利用容量但循环寿命会降低"
        else:
            return f"DOD设置合理({dod*100:.0f}%)，平衡了容量利用和循环寿命"
    
    def generate_dod_commands(self, units: List[Dict]) -> List[LifeOptimizationCommand]:
        """生成DOD优化命令"""
        commands = []
        
        for unit in units:
            soh = unit.get("soh", 100) / 100
            current_dod = unit.get("current_dod", 0.8)
            unit_id = unit.get("unit_id", "unknown")
            
            recommendation = self.recommend_dod_limit(soh)
            recommended_dod = recommendation["recommended_max_dod"]
            
            if current_dod > recommended_dod:
                commands.append(LifeOptimizationCommand(
                    target=LifeOptimizationTarget.DOD,
                    unit_id=unit_id,
                    action="limit_dod",
                    parameters={
                        "max_dod": recommended_dod,
                        "min_soc": recommendation["recommended_soc_range"]["min"],
                        "max_soc": recommendation["recommended_soc_range"]["max"]
                    },
                    priority=6,
                    expected_life_gain=self._estimate_life_gain(current_dod, recommended_dod),
                    reason=f"当前DOD({current_dod*100:.0f}%)过高，建议限制到{recommended_dod*100:.0f}%"
                ))
        
        return commands
    
    def _estimate_life_gain(self, old_dod: float, new_dod: float) -> float:
        """估算DOD优化带来的寿命增益"""
        old_cycles = self.calculate_cycle_life(old_dod)
        new_cycles = self.calculate_cycle_life(new_dod)
        
        if old_cycles <= 0:
            return 0
        
        gain = (new_cycles - old_cycles) / old_cycles * 100
        return round(max(0, gain), 1)


# ==================== 倍率优化策略 ====================

class CRateOptimizer:
    """
    充放电倍率(C-Rate)优化器
    
    原理：高倍率充放电会增加内部热效应和极化，加速老化
    
    影响因素：
    1. 锂离子浓度梯度
    2. 欧姆热和极化热
    3. 锂枝晶生长风险（高倍率充电）
    
    策略：
    1. 限制最大充放电倍率
    2. SOC相关倍率调整
    3. 温度相关倍率调整
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        
        # 倍率限制
        self.max_charge_rate = self.config.get("max_charge_rate", 0.5)  # 最大充电倍率
        self.max_discharge_rate = self.config.get("max_discharge_rate", 1.0)  # 最大放电倍率
        self.optimal_charge_rate = self.config.get("optimal_charge_rate", 0.3)
        self.optimal_discharge_rate = self.config.get("optimal_discharge_rate", 0.5)
        
    def _default_config(self) -> Dict:
        return {
            "max_charge_rate": 0.5,       # 0.5C
            "max_discharge_rate": 1.0,     # 1C
            "optimal_charge_rate": 0.3,    # 0.3C
            "optimal_discharge_rate": 0.5, # 0.5C
            "high_soc_derating": 0.5,      # 高SOC时降额比例
            "low_soc_derating": 0.5,       # 低SOC时降额比例
        }
    
    def calculate_aging_factor_from_crate(self, c_rate: float, is_charging: bool) -> float:
        """
        计算倍率对老化的影响因子
        
        基于经验公式，高倍率会加速老化
        
        Parameters:
        -----------
        c_rate : float
            充放电倍率
        is_charging : bool
            是否为充电
            
        Returns:
        --------
        float: 老化加速因子
        """
        # 参考倍率
        ref_rate = self.optimal_charge_rate if is_charging else self.optimal_discharge_rate
        
        # 老化因子（二次关系）
        factor = 1 + ((c_rate - ref_rate) / ref_rate) ** 2 * 0.5
        
        # 充电时高倍率影响更大
        if is_charging and c_rate > 0.5:
            factor *= 1.2
        
        return round(max(1.0, factor), 3)
    
    def get_soc_based_rate_limit(self, soc: float, is_charging: bool) -> float:
        """
        根据SOC获取倍率限制
        
        充电时：高SOC降低倍率，避免过充
        放电时：低SOC降低倍率，避免过放
        
        Parameters:
        -----------
        soc : float
            当前SOC (0-1)
        is_charging : bool
            是否为充电
            
        Returns:
        --------
        float: 推荐的倍率限制
        """
        if is_charging:
            base_rate = self.max_charge_rate
            if soc > 0.8:
                # 高SOC时降低充电倍率
                derating = 1 - (soc - 0.8) * 2  # 80%-100% 线性降额
                return round(base_rate * max(0.2, derating), 2)
            elif soc > 0.9:
                return round(base_rate * 0.1, 2)  # 涓流充电
        else:
            base_rate = self.max_discharge_rate
            if soc < 0.2:
                # 低SOC时降低放电倍率
                derating = soc * 2.5  # 0%-20% 线性降额
                return round(base_rate * max(0.2, derating), 2)
            elif soc < 0.1:
                return round(base_rate * 0.1, 2)  # 限制放电
        
        return base_rate
    
    def get_temperature_based_rate_limit(
        self, 
        temperature: float, 
        is_charging: bool,
        base_rate: float = None
    ) -> float:
        """
        根据温度获取倍率限制
        
        Parameters:
        -----------
        temperature : float
            温度 (℃)
        is_charging : bool
            是否为充电
        base_rate : float
            基准倍率
            
        Returns:
        --------
        float: 温度修正后的倍率限制
        """
        if base_rate is None:
            base_rate = self.max_charge_rate if is_charging else self.max_discharge_rate
        
        # 温度影响因子
        if temperature < 0:
            # 低温严重限制，特别是充电
            factor = 0.1 if is_charging else 0.3
        elif temperature < 10:
            factor = 0.3 if is_charging else 0.5
        elif temperature < 20:
            factor = 0.7
        elif temperature > 45:
            factor = 0.5
        elif temperature > 40:
            factor = 0.7
        else:
            factor = 1.0
        
        return round(base_rate * factor, 2)
    
    def generate_rate_optimization_plan(
        self,
        units: List[Dict],
        is_charging: bool
    ) -> Dict:
        """
        生成倍率优化计划
        
        Parameters:
        -----------
        units : list
            单元数据列表
        is_charging : bool
            是否为充电操作
            
        Returns:
        --------
        dict: 优化计划
        """
        plan = {
            "mode": "charging" if is_charging else "discharging",
            "unit_limits": [],
            "system_limit": None,
            "recommendations": []
        }
        
        rates = []
        for unit in units:
            soc = unit.get("soc", 50) / 100
            temp = unit.get("temperature", 25)
            unit_id = unit.get("unit_id", "unknown")
            
            # SOC限制
            soc_limit = self.get_soc_based_rate_limit(soc, is_charging)
            
            # 温度限制
            temp_limit = self.get_temperature_based_rate_limit(temp, is_charging, soc_limit)
            
            final_limit = min(soc_limit, temp_limit)
            rates.append(final_limit)
            
            plan["unit_limits"].append({
                "unit_id": unit_id,
                "soc": round(soc * 100, 1),
                "temperature": round(temp, 1),
                "soc_based_limit": soc_limit,
                "temp_based_limit": temp_limit,
                "final_limit": final_limit
            })
        
        # 系统级限制（取最小值，保守策略）
        plan["system_limit"] = min(rates) if rates else 0.5
        
        # 生成建议
        if plan["system_limit"] < 0.3:
            plan["recommendations"].append("当前条件限制较大，建议低功率运行")
        
        return plan
    
    def generate_crate_commands(self, units: List[Dict]) -> List[LifeOptimizationCommand]:
        """生成倍率优化命令"""
        commands = []
        
        for unit in units:
            current_rate = unit.get("current_c_rate", 0.5)
            soc = unit.get("soc", 50) / 100
            temp = unit.get("temperature", 25)
            unit_id = unit.get("unit_id", "unknown")
            is_charging = unit.get("is_charging", True)
            
            optimal_rate = self.get_soc_based_rate_limit(soc, is_charging)
            optimal_rate = self.get_temperature_based_rate_limit(temp, is_charging, optimal_rate)
            
            if current_rate > optimal_rate * 1.1:  # 超过10%时优化
                commands.append(LifeOptimizationCommand(
                    target=LifeOptimizationTarget.C_RATE,
                    unit_id=unit_id,
                    action="limit_rate",
                    parameters={
                        "max_c_rate": optimal_rate,
                        "current_c_rate": current_rate,
                        "is_charging": is_charging
                    },
                    priority=7,
                    expected_life_gain=self._estimate_rate_life_gain(current_rate, optimal_rate),
                    reason=f"{'充电' if is_charging else '放电'}倍率{current_rate}C过高，建议限制到{optimal_rate}C"
                ))
        
        return commands
    
    def _estimate_rate_life_gain(self, old_rate: float, new_rate: float) -> float:
        """估算倍率优化带来的寿命增益"""
        old_factor = self.calculate_aging_factor_from_crate(old_rate, True)
        new_factor = self.calculate_aging_factor_from_crate(new_rate, True)
        
        if new_factor <= 0:
            return 0
        
        gain = (old_factor - new_factor) / old_factor * 100
        return round(max(0, gain), 1)


# ==================== SOC管理策略 ====================

class SOCManager:
    """
    SOC管理器
    
    原理：
    1. 高SOC存储会加速日历老化（电极材料氧化）
    2. 低SOC存储可能导致铜箔溶解
    3. 极端SOC状态会增加内阻
    
    策略：
    1. 长期存储时保持中等SOC(40-60%)
    2. 避免长时间处于满电或空电状态
    3. 根据使用模式调整SOC范围
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        
        # SOC范围
        self.storage_soc_min = self.config.get("storage_soc_min", 0.4)
        self.storage_soc_max = self.config.get("storage_soc_max", 0.6)
        self.operating_soc_min = self.config.get("operating_soc_min", 0.1)
        self.operating_soc_max = self.config.get("operating_soc_max", 0.9)
        
    def _default_config(self) -> Dict:
        return {
            "storage_soc_min": 0.4,
            "storage_soc_max": 0.6,
            "operating_soc_min": 0.1,
            "operating_soc_max": 0.9,
            "high_soc_threshold": 0.95,
            "low_soc_threshold": 0.05,
            "max_high_soc_hours": 24,  # 最大高SOC持续时间
        }
    
    def calculate_calendar_aging_factor(self, soc: float, temperature: float = 25) -> float:
        """
        计算日历老化因子
        
        高SOC和高温会加速日历老化
        
        Parameters:
        -----------
        soc : float
            SOC (0-1)
        temperature : float
            温度 (℃)
            
        Returns:
        --------
        float: 日历老化因子
        """
        # SOC影响因子（高SOC加速老化）
        soc_factor = 1 + (soc - 0.5) ** 2 * 2  # 50% SOC时最小
        
        # 温度影响因子
        temp_factor = np.exp((temperature - 25) * 0.03)  # 每10℃翻倍
        
        return round(soc_factor * temp_factor, 3)
    
    def recommend_soc_management(
        self,
        current_soc: float,
        usage_pattern: str = "daily",
        idle_hours: float = 0
    ) -> Dict:
        """
        推荐SOC管理策略
        
        Parameters:
        -----------
        current_soc : float
            当前SOC
        usage_pattern : str
            使用模式: daily, weekly, seasonal, backup
        idle_hours : float
            预计闲置时间(小时)
            
        Returns:
        --------
        dict: SOC管理建议
        """
        recommendations = []
        target_soc = current_soc
        
        if idle_hours > 168:  # 超过一周
            # 长期存储，调整到存储SOC
            target_soc = (self.storage_soc_min + self.storage_soc_max) / 2
            recommendations.append(f"长期存储建议：调整SOC到{target_soc*100:.0f}%")
        elif idle_hours > 24:
            # 短期闲置
            if current_soc > 0.8:
                target_soc = 0.7
                recommendations.append("建议将SOC降至70%避免高SOC存储")
        
        if current_soc > self.config.get("high_soc_threshold", 0.95):
            recommendations.append("SOC过高，建议放电至90%以下")
        elif current_soc < self.config.get("low_soc_threshold", 0.05):
            recommendations.append("SOC过低，建议充电至20%以上")
        
        # 计算当前老化因子
        aging_factor = self.calculate_calendar_aging_factor(current_soc)
        optimal_aging_factor = self.calculate_calendar_aging_factor(0.5)
        
        return {
            "current_soc": round(current_soc * 100, 1),
            "target_soc": round(target_soc * 100, 1),
            "usage_pattern": usage_pattern,
            "idle_hours": idle_hours,
            "current_aging_factor": aging_factor,
            "optimal_aging_factor": optimal_aging_factor,
            "recommendations": recommendations if recommendations else ["SOC状态良好"],
            "action_needed": target_soc != current_soc
        }
    
    def generate_soc_commands(self, units: List[Dict]) -> List[LifeOptimizationCommand]:
        """生成SOC管理命令"""
        commands = []
        
        for unit in units:
            soc = unit.get("soc", 50) / 100
            idle_hours = unit.get("expected_idle_hours", 0)
            unit_id = unit.get("unit_id", "unknown")
            
            recommendation = self.recommend_soc_management(soc, idle_hours=idle_hours)
            
            if recommendation["action_needed"]:
                target_soc = recommendation["target_soc"] / 100
                action = "discharge" if target_soc < soc else "charge"
                
                commands.append(LifeOptimizationCommand(
                    target=LifeOptimizationTarget.SOC,
                    unit_id=unit_id,
                    action=action,
                    parameters={
                        "current_soc": soc,
                        "target_soc": target_soc,
                        "idle_hours": idle_hours
                    },
                    priority=5,
                    expected_life_gain=2.0,
                    reason=recommendation["recommendations"][0] if recommendation["recommendations"] else "SOC优化"
                ))
        
        return commands


# ==================== 充电策略优化 ====================

class ChargingStrategyOptimizer:
    """
    充电策略优化器
    
    支持的充电策略：
    1. CC-CV (恒流恒压) - 标准策略
    2. 阶梯充电 (Multi-stage CC) - 延长寿命
    3. 脉冲充电 (Pulse Charging) - 减少极化
    4. 自适应充电 - 根据温度和SOH调整
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        
    def _default_config(self) -> Dict:
        return {
            "default_strategy": "adaptive",
            "cc_cv_threshold": 0.8,  # CC-CV切换点
            "stages": [
                {"soc_end": 0.5, "c_rate": 0.5},
                {"soc_end": 0.8, "c_rate": 0.3},
                {"soc_end": 0.9, "c_rate": 0.2},
                {"soc_end": 1.0, "c_rate": 0.1},
            ],
            "pulse_on_time": 5,      # 脉冲开启时间(s)
            "pulse_off_time": 1,     # 脉冲关闭时间(s)
        }
    
    def get_charging_parameters(
        self,
        current_soc: float,
        temperature: float,
        soh: float,
        strategy: str = "adaptive"
    ) -> Dict:
        """
        获取充电参数
        
        Parameters:
        -----------
        current_soc : float
            当前SOC
        temperature : float
            温度
        soh : float
            健康状态
        strategy : str
            充电策略
            
        Returns:
        --------
        dict: 充电参数
        """
        if strategy == "cc_cv":
            return self._get_cc_cv_parameters(current_soc)
        elif strategy == "multi_stage":
            return self._get_multi_stage_parameters(current_soc)
        elif strategy == "pulse":
            return self._get_pulse_parameters(current_soc)
        else:  # adaptive
            return self._get_adaptive_parameters(current_soc, temperature, soh)
    
    def _get_cc_cv_parameters(self, soc: float) -> Dict:
        """CC-CV充电参数"""
        threshold = self.config["cc_cv_threshold"]
        
        if soc < threshold:
            return {
                "strategy": "cc_cv",
                "phase": "CC",
                "c_rate": 0.5,
                "voltage_limit": 3.65
            }
        else:
            return {
                "strategy": "cc_cv",
                "phase": "CV",
                "c_rate": None,  # 由电压控制
                "voltage_limit": 3.65,
                "cutoff_current": 0.05  # 截止电流
            }
    
    def _get_multi_stage_parameters(self, soc: float) -> Dict:
        """阶梯充电参数"""
        stages = self.config["stages"]
        
        for stage in stages:
            if soc < stage["soc_end"]:
                return {
                    "strategy": "multi_stage",
                    "current_stage": stages.index(stage) + 1,
                    "c_rate": stage["c_rate"],
                    "soc_target": stage["soc_end"],
                    "total_stages": len(stages)
                }
        
        return {
            "strategy": "multi_stage",
            "current_stage": len(stages),
            "c_rate": stages[-1]["c_rate"],
            "soc_target": 1.0
        }
    
    def _get_pulse_parameters(self, soc: float) -> Dict:
        """脉冲充电参数"""
        return {
            "strategy": "pulse",
            "c_rate": 0.5,
            "pulse_on_time": self.config["pulse_on_time"],
            "pulse_off_time": self.config["pulse_off_time"],
            "duty_cycle": self.config["pulse_on_time"] / (
                self.config["pulse_on_time"] + self.config["pulse_off_time"]
            )
        }
    
    def _get_adaptive_parameters(self, soc: float, temperature: float, soh: float) -> Dict:
        """自适应充电参数"""
        # 基础倍率
        base_rate = 0.5
        
        # 温度调整
        if temperature < 10:
            base_rate *= 0.5
        elif temperature > 40:
            base_rate *= 0.7
        
        # SOH调整
        if soh < 0.8:
            base_rate *= 0.8
        
        # SOC调整
        if soc > 0.8:
            base_rate *= 0.5
        elif soc > 0.9:
            base_rate *= 0.2
        
        return {
            "strategy": "adaptive",
            "c_rate": round(base_rate, 2),
            "temperature_factor": self._get_temp_factor(temperature),
            "soh_factor": min(1.0, soh / 0.8),
            "soc_factor": 1.0 if soc < 0.8 else 0.5 if soc < 0.9 else 0.2,
            "recommended_for_life": True
        }
    
    def _get_temp_factor(self, temperature: float) -> float:
        """获取温度因子"""
        if temperature < 0:
            return 0.2
        elif temperature < 10:
            return 0.5
        elif temperature > 45:
            return 0.5
        elif temperature > 35:
            return 0.8
        else:
            return 1.0


# ==================== 综合延寿管理器 ====================

class BatteryLifeExtensionManager:
    """
    电池延寿综合管理器
    
    整合所有延寿策略，提供统一的寿命优化接口
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # 初始化各优化器
        self.temp_optimizer = TemperatureOptimizer(self.config.get("temperature", {}))
        self.dod_optimizer = DODOptimizer(self.config.get("dod", {}))
        self.crate_optimizer = CRateOptimizer(self.config.get("c_rate", {}))
        self.soc_manager = SOCManager(self.config.get("soc", {}))
        self.charging_optimizer = ChargingStrategyOptimizer(self.config.get("charging", {}))
        
        # 优化权重
        self.optimization_weights = {
            LifeOptimizationTarget.TEMPERATURE: 1.0,
            LifeOptimizationTarget.DOD: 0.8,
            LifeOptimizationTarget.C_RATE: 0.9,
            LifeOptimizationTarget.SOC: 0.7,
            LifeOptimizationTarget.CHARGING: 0.6,
        }
        
    def analyze_system_health(self, system_data: Dict) -> Dict:
        """
        分析系统健康状态和寿命预测
        
        Parameters:
        -----------
        system_data : dict
            系统数据
            
        Returns:
        --------
        dict: 健康分析结果
        """
        units = system_data.get("units", [])
        
        if not units:
            return {"status": "no_data"}
        
        # 温度分析
        temps = [u.get("temperature", 25) for u in units]
        temp_analysis = self.temp_optimizer.analyze_temperature_status(temps)
        
        # SOH统计
        sohs = [u.get("soh", 100) / 100 for u in units]
        avg_soh = np.mean(sohs)
        min_soh = min(sohs)
        
        # 循环次数统计
        cycles = [u.get("cycle_count", 0) for u in units]
        avg_cycles = np.mean(cycles)
        
        # 预测剩余寿命
        remaining_life_years = self._estimate_remaining_life(avg_soh, avg_cycles)
        
        # 计算综合健康评分
        health_score = self._calculate_health_score(avg_soh, temp_analysis, avg_cycles)
        
        return {
            "health_score": round(health_score, 1),
            "health_status": self._get_health_status(health_score),
            "soh_statistics": {
                "average": round(avg_soh * 100, 1),
                "minimum": round(min_soh * 100, 1),
                "units_below_80": sum(1 for s in sohs if s < 0.8)
            },
            "temperature_analysis": temp_analysis,
            "cycle_statistics": {
                "average": round(avg_cycles),
                "maximum": max(cycles),
                "total": sum(cycles)
            },
            "remaining_life_prediction": {
                "years": round(remaining_life_years, 1),
                "cycles": int(remaining_life_years * 365),  # 假设每天一个循环
                "confidence": "medium"
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def _estimate_remaining_life(self, avg_soh: float, avg_cycles: float) -> float:
        """估算剩余寿命（年）"""
        # 简化模型：基于SOH和循环次数
        remaining_soh = avg_soh - 0.7  # 到达70% SOH时寿命结束
        
        if remaining_soh <= 0:
            return 0
        
        # 估算每年SOH下降率（假设年均500次循环）
        annual_degradation = 0.03  # 3%/年
        
        return remaining_soh / annual_degradation
    
    def _calculate_health_score(
        self, 
        avg_soh: float, 
        temp_analysis: Dict,
        avg_cycles: float
    ) -> float:
        """计算综合健康评分(0-100)"""
        # SOH权重50%
        soh_score = avg_soh * 100 * 0.5
        
        # 温度权重30%
        temp_status = temp_analysis.get("status", "unknown")
        temp_scores = {"optimal": 30, "suboptimal": 20, "warning": 10, "critical": 0}
        temp_score = temp_scores.get(temp_status, 15)
        
        # 循环次数权重20%（假设6000次满寿命）
        cycle_score = max(0, (1 - avg_cycles / 6000)) * 20
        
        return soh_score + temp_score + cycle_score
    
    def _get_health_status(self, score: float) -> str:
        """根据评分获取健康状态"""
        if score >= 85:
            return "excellent"
        elif score >= 70:
            return "good"
        elif score >= 50:
            return "fair"
        elif score >= 30:
            return "poor"
        else:
            return "critical"
    
    def generate_life_extension_plan(self, system_data: Dict) -> Dict:
        """
        生成综合延寿计划
        
        Parameters:
        -----------
        system_data : dict
            系统数据
            
        Returns:
        --------
        dict: 延寿计划
        """
        units = system_data.get("units", [])
        
        plan = {
            "health_analysis": self.analyze_system_health(system_data),
            "commands": {
                "temperature": [],
                "dod": [],
                "c_rate": [],
                "soc": [],
                "charging": []
            },
            "priority_actions": [],
            "expected_life_extension": 0,
            "recommendations": []
        }
        
        # 温度优化命令
        temp_units = [{"unit_id": u.get("id"), "temperature": u.get("temperature", 25)} for u in units]
        plan["commands"]["temperature"] = self.temp_optimizer.generate_thermal_commands(temp_units)
        
        # DOD优化命令
        dod_units = [{
            "unit_id": u.get("id"),
            "soh": u.get("soh", 100),
            "current_dod": u.get("current_dod", 0.8)
        } for u in units]
        plan["commands"]["dod"] = self.dod_optimizer.generate_dod_commands(dod_units)
        
        # 倍率优化命令
        rate_units = [{
            "unit_id": u.get("id"),
            "soc": u.get("soc", 50),
            "temperature": u.get("temperature", 25),
            "current_c_rate": u.get("current_c_rate", 0.5),
            "is_charging": u.get("is_charging", True)
        } for u in units]
        plan["commands"]["c_rate"] = self.crate_optimizer.generate_crate_commands(rate_units)
        
        # SOC管理命令
        soc_units = [{
            "unit_id": u.get("id"),
            "soc": u.get("soc", 50),
            "expected_idle_hours": u.get("expected_idle_hours", 0)
        } for u in units]
        plan["commands"]["soc"] = self.soc_manager.generate_soc_commands(soc_units)
        
        # 收集所有命令
        all_commands = []
        for cmd_list in plan["commands"].values():
            all_commands.extend(cmd_list)
        
        # 按优先级排序，取前5个作为优先行动
        sorted_commands = sorted(all_commands, key=lambda c: c.priority, reverse=True)
        plan["priority_actions"] = [
            {
                "target": cmd.target.value,
                "unit_id": cmd.unit_id,
                "action": cmd.action,
                "priority": cmd.priority,
                "expected_life_gain": cmd.expected_life_gain,
                "reason": cmd.reason
            }
            for cmd in sorted_commands[:5]
        ]
        
        # 计算预期延寿效果
        total_gain = sum(cmd.expected_life_gain for cmd in all_commands)
        plan["expected_life_extension"] = round(min(total_gain, 30), 1)  # 最大30%
        
        # 生成建议
        plan["recommendations"] = self._generate_recommendations(plan)
        
        return plan
    
    def _generate_recommendations(self, plan: Dict) -> List[str]:
        """生成综合建议"""
        recommendations = []
        health = plan["health_analysis"]
        
        if health.get("health_status") == "poor":
            recommendations.append("系统健康状态较差，建议全面检查并执行延寿计划")
        
        temp_status = health.get("temperature_analysis", {}).get("status", "unknown")
        if temp_status in ["warning", "critical"]:
            recommendations.append("温度管理需要优化，建议检查热管理系统")
        
        soh_min = health.get("soh_statistics", {}).get("minimum", 100)
        if soh_min < 80:
            recommendations.append(f"部分单元SOH已低于80%({soh_min}%)，建议更换或重点保护")
        
        if plan["expected_life_extension"] > 10:
            recommendations.append(f"执行延寿计划预计可延长{plan['expected_life_extension']:.1f}%的使用寿命")
        
        if not recommendations:
            recommendations.append("系统状态良好，继续保持当前运维策略")
        
        return recommendations
    
    def get_life_extension_strategies(self) -> Dict:
        """获取所有延寿策略说明"""
        return {
            "temperature_optimization": {
                "name": "温度优化策略",
                "description": "通过预热、冷却和温度均衡控制，将电池维持在最佳工作温度范围(20-35℃)",
                "key_parameters": {
                    "optimal_range": "20-35℃",
                    "high_temp_derating": "45℃以上降额运行",
                    "low_temp_preheat": "5℃以下预热"
                },
                "expected_life_extension": "5-15%"
            },
            "dod_optimization": {
                "name": "放电深度优化策略",
                "description": "通过限制放电深度延长循环寿命",
                "key_parameters": {
                    "max_dod": "建议80%以下",
                    "optimal_dod": "60%左右",
                    "soh_based_adjustment": "SOH<80%时更保守"
                },
                "expected_life_extension": "20-50%"
            },
            "c_rate_optimization": {
                "name": "倍率优化策略",
                "description": "控制充放电倍率，减少热效应和极化损伤",
                "key_parameters": {
                    "max_charge_rate": "0.5C",
                    "max_discharge_rate": "1C",
                    "soc_based_derating": "高SOC时降低充电倍率"
                },
                "expected_life_extension": "10-20%"
            },
            "soc_management": {
                "name": "SOC管理策略",
                "description": "避免长期高SOC或低SOC存储，减少日历老化",
                "key_parameters": {
                    "storage_soc": "40-60%",
                    "operating_range": "10-90%",
                    "high_soc_limit_hours": "24小时内调整"
                },
                "expected_life_extension": "5-10%"
            },
            "charging_optimization": {
                "name": "充电策略优化",
                "description": "采用阶梯充电、脉冲充电等先进充电策略",
                "strategies": {
                    "cc_cv": "标准恒流恒压充电",
                    "multi_stage": "多阶段阶梯充电",
                    "pulse": "脉冲充电（减少极化）",
                    "adaptive": "自适应充电（推荐）"
                },
                "expected_life_extension": "5-15%"
            },
            "ripple_control": {
                "name": "电流波纹控制",
                "description": "减少交流纹波对电池的损伤",
                "key_parameters": {
                    "max_ripple": "<5%",
                    "frequency_range": "关注100Hz-10kHz"
                },
                "expected_life_extension": "3-8%"
            },
            "load_balance": {
                "name": "负载均衡策略",
                "description": "均匀分配各单元的功率负荷，避免局部过载",
                "key_parameters": {
                    "power_deviation": "<10%",
                    "soh_based_allocation": "低SOH单元降额"
                },
                "expected_life_extension": "5-10%"
            }
        }
    
    def calculate_total_life_extension_potential(self, current_state: Dict) -> Dict:
        """
        计算总延寿潜力
        
        Parameters:
        -----------
        current_state : dict
            当前状态
            
        Returns:
        --------
        dict: 延寿潜力分析
        """
        potential = {
            "temperature": 0,
            "dod": 0,
            "c_rate": 0,
            "soc": 0,
            "charging": 0,
            "total": 0
        }
        
        # 温度优化潜力
        temp = current_state.get("avg_temperature", 25)
        if temp > 35 or temp < 20:
            potential["temperature"] = 10
        elif temp > 40 or temp < 10:
            potential["temperature"] = 15
        
        # DOD优化潜力
        dod = current_state.get("current_dod", 0.8)
        if dod > 0.8:
            potential["dod"] = (dod - 0.6) * 100  # 每降低10% DOD约增加10%寿命
        
        # 倍率优化潜力
        c_rate = current_state.get("avg_c_rate", 0.5)
        if c_rate > 0.5:
            potential["c_rate"] = (c_rate - 0.3) * 30
        
        # SOC管理潜力
        high_soc_hours = current_state.get("high_soc_hours_per_day", 0)
        if high_soc_hours > 4:
            potential["soc"] = min(10, high_soc_hours * 1.5)
        
        # 充电策略潜力
        charging_strategy = current_state.get("charging_strategy", "cc_cv")
        if charging_strategy == "cc_cv":
            potential["charging"] = 8
        
        potential["total"] = sum(v for k, v in potential.items() if k != "total")
        potential["total"] = round(min(potential["total"], 50), 1)  # 最大50%
        
        return potential


# 创建全局延寿管理器实例
life_extension_manager = BatteryLifeExtensionManager()


