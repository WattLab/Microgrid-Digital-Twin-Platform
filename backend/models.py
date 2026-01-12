"""数据库模型定义"""
from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from backend.database import Base


class SimulationRun(Base):
    """仿真运行记录"""
    __tablename__ = "simulation_runs"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    status = Column(String(50), default="running")  # running, completed, failed
    parameters = Column(Text)  # JSON格式的仿真参数
    
    # 关联仿真数据
    data_points = relationship("SimulationData", back_populates="simulation_run", cascade="all, delete-orphan")


class SimulationData(Base):
    """仿真数据点"""
    __tablename__ = "simulation_data"
    
    id = Column(Integer, primary_key=True, index=True)
    simulation_run_id = Column(Integer, ForeignKey("simulation_runs.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # 发电数据 (kW)
    solar_power = Column(Float, default=0.0)
    wind_power = Column(Float, default=0.0)
    diesel_power = Column(Float, default=0.0)
    
    # 负荷数据 (kW)
    load_power = Column(Float, default=0.0)
    
    # 储能数据
    battery_soc = Column(Float, default=0.5)  # SOC 0-1
    battery_power = Column(Float, default=0.0)  # 正充电负放电
    
    # 电网交互 (kW)
    grid_power = Column(Float, default=0.0)  # 正购电负售电
    
    # 电能质量
    frequency = Column(Float, default=50.0)  # Hz
    voltage = Column(Float, default=380.0)   # V
    
    simulation_run = relationship("SimulationRun", back_populates="data_points")


class RealData(Base):
    """实际场景数据（用于标定）"""
    __tablename__ = "real_data"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    source = Column(String(100))
    
    solar_power = Column(Float)
    wind_power = Column(Float)
    diesel_power = Column(Float)
    load_power = Column(Float)
    battery_soc = Column(Float)
    battery_power = Column(Float)
    grid_power = Column(Float)
    frequency = Column(Float)
    voltage = Column(Float)
    
    used_for_calibration = Column(Boolean, default=False)


class PredictionResult(Base):
    """预测结果"""
    __tablename__ = "prediction_results"
    
    id = Column(Integer, primary_key=True, index=True)
    prediction_type = Column(String(50))  # load, solar, wind
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    forecast_time = Column(DateTime, index=True)
    
    predicted_value = Column(Float)
    confidence_upper = Column(Float)
    confidence_lower = Column(Float)
    model_name = Column(String(100))


class OptimizationResult(Base):
    """优化结果"""
    __tablename__ = "optimization_results"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    optimization_type = Column(String(50))
    
    solar_dispatch = Column(Float)
    wind_dispatch = Column(Float)
    diesel_dispatch = Column(Float)
    battery_charge = Column(Float)
    battery_discharge = Column(Float)
    grid_purchase = Column(Float)
    grid_sell = Column(Float)
    
    objective_value = Column(Float)
    constraints_satisfied = Column(Boolean, default=True)
    optimization_params = Column(Text)
