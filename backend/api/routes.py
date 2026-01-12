"""API路由定义"""
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
import json
import asyncio

from backend.database import get_db
from backend.models import SimulationRun, SimulationData, RealData, PredictionResult, OptimizationResult
from backend.simulation.simulator import MicrogridSimulator
from backend.prediction.predictor import LoadPredictor, RenewablePredictor
from backend.prediction.price_predictor import ElectricityPricePredictor, MarketPricePredictor
from backend.optimization.scheduler import EnergyScheduler
from backend.optimization.energy_storage import EnergyStorageStation, SmallEnergyStorage
from backend.optimization.smart_scheduler import SmartEnergyScheduler
from backend.optimization.energy_storage_detail import storage_system
from pydantic import BaseModel

router = APIRouter()

# 全局实例
simulator = MicrogridSimulator()
load_predictor = LoadPredictor()
solar_predictor = RenewablePredictor('solar')
wind_predictor = RenewablePredictor('wind')
scheduler = EnergyScheduler()

# 电价预测器
price_predictor = ElectricityPricePredictor(region="guangdong")
market_predictor = MarketPricePredictor()

# 储能系统
energy_storage = SmallEnergyStorage()

# 智能调度器
smart_scheduler = SmartEnergyScheduler(
    storage_config=energy_storage.config,
    price_predictor=price_predictor
)

# WebSocket连接管理
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                pass

manager = ConnectionManager()


# Pydantic模型
class SimulationParams(BaseModel):
    name: str = "默认仿真"
    duration_hours: int = 24
    weather_factor: float = 1.0
    wind_speed: float = 8.0
    solar_capacity: Optional[float] = None
    wind_capacity: Optional[float] = None
    diesel_capacity: Optional[float] = None
    battery_capacity: Optional[float] = None
    base_load: Optional[float] = None


class CalibrationData(BaseModel):
    solar_power: List[float]
    wind_power: List[float]
    load_power: List[float]


class StorageCommand(BaseModel):
    action: str  # charge, discharge, standby
    power: float = 0
    duration: float = 1.0


# ==================== 仿真相关API ====================

@router.post("/simulation/start")
async def start_simulation(params: SimulationParams, db: Session = Depends(get_db)):
    """启动仿真"""
    if params.solar_capacity:
        simulator.parameters['solar_capacity'] = params.solar_capacity
    if params.wind_capacity:
        simulator.parameters['wind_capacity'] = params.wind_capacity
    if params.diesel_capacity:
        simulator.parameters['diesel_capacity'] = params.diesel_capacity
    if params.battery_capacity:
        simulator.parameters['battery_capacity'] = params.battery_capacity
    if params.base_load:
        simulator.parameters['base_load'] = params.base_load
    
    sim_run = SimulationRun(
        name=params.name,
        description=f"仿真时长: {params.duration_hours}小时",
        status="running",
        parameters=simulator.get_parameters_json()
    )
    db.add(sim_run)
    db.commit()
    db.refresh(sim_run)
    
    results = simulator.simulate(
        duration_hours=params.duration_hours,
        weather_factor=params.weather_factor,
        wind_speed=params.wind_speed
    )
    
    for i, data in enumerate(results):
        sim_data = SimulationData(
            simulation_run_id=sim_run.id,
            timestamp=datetime.now(),
            solar_power=data['solar_power'],
            wind_power=data['wind_power'],
            diesel_power=data['diesel_power'],
            load_power=data['load_power'],
            battery_soc=data['battery_soc'],
            battery_power=data['battery_power'],
            grid_power=data['grid_power'],
            voltage=data['voltage'],
            frequency=data['frequency']
        )
        db.add(sim_data)
    
    sim_run.status = "completed"
    sim_run.end_time = datetime.now()
    db.commit()
    
    load_predictor.add_history(results)
    solar_predictor.add_history(results)
    wind_predictor.add_history(results)
    
    return {
        "simulation_id": sim_run.id,
        "status": "completed",
        "data": results
    }


@router.get("/simulation/list")
async def list_simulations(db: Session = Depends(get_db)):
    """获取仿真列表"""
    simulations = db.query(SimulationRun).order_by(SimulationRun.start_time.desc()).limit(20).all()
    return [{
        "id": sim.id,
        "name": sim.name,
        "status": sim.status,
        "start_time": sim.start_time.isoformat() if sim.start_time else None,
        "end_time": sim.end_time.isoformat() if sim.end_time else None
    } for sim in simulations]


@router.get("/simulation/{sim_id}/data")
async def get_simulation_data(sim_id: int, db: Session = Depends(get_db)):
    """获取仿真数据"""
    data = db.query(SimulationData).filter(
        SimulationData.simulation_run_id == sim_id
    ).order_by(SimulationData.timestamp).all()
    
    return [{
        "hour": i,
        "solar_power": d.solar_power,
        "wind_power": d.wind_power,
        "diesel_power": d.diesel_power,
        "load_power": d.load_power,
        "battery_soc": d.battery_soc,
        "battery_power": d.battery_power,
        "grid_power": d.grid_power,
        "voltage": d.voltage,
        "frequency": d.frequency
    } for i, d in enumerate(data)]


# ==================== 标定相关API ====================

@router.post("/calibration/upload")
async def upload_calibration_data(data: CalibrationData, db: Session = Depends(get_db)):
    """上传标定数据"""
    for i in range(len(data.solar_power)):
        real_data = RealData(
            timestamp=datetime.now(),
            source="calibration",
            solar_power=data.solar_power[i] if i < len(data.solar_power) else 0,
            wind_power=data.wind_power[i] if i < len(data.wind_power) else 0,
            load_power=data.load_power[i] if i < len(data.load_power) else 0,
            used_for_calibration=True
        )
        db.add(real_data)
    
    db.commit()
    return {"status": "success", "message": f"已上传 {len(data.solar_power)} 条标定数据"}


@router.post("/calibration/run")
async def run_calibration(db: Session = Depends(get_db)):
    """执行模型标定"""
    real_data = db.query(RealData).filter(RealData.used_for_calibration == True).all()
    
    if len(real_data) < 10:
        return {"status": "error", "message": "标定数据不足，需要至少10条数据"}
    
    solar_powers = [d.solar_power for d in real_data if d.solar_power]
    wind_powers = [d.wind_power for d in real_data if d.wind_power]
    load_powers = [d.load_power for d in real_data if d.load_power]
    
    if solar_powers:
        max_solar = max(solar_powers)
        simulator.parameters['solar_capacity'] = max_solar * 1.2
    
    if wind_powers:
        max_wind = max(wind_powers)
        simulator.parameters['wind_capacity'] = max_wind * 1.2
    
    if load_powers:
        avg_load = sum(load_powers) / len(load_powers)
        simulator.parameters['base_load'] = avg_load
    
    return {
        "status": "success",
        "message": "标定完成",
        "calibrated_params": simulator.parameters
    }


# ==================== 预测相关API ====================

@router.get("/prediction/load")
async def predict_load(hours: int = 24):
    """负荷预测"""
    predictions = load_predictor.predict(hours)
    return {"predictions": predictions}


@router.get("/prediction/solar")
async def predict_solar(hours: int = 24):
    """光伏发电预测"""
    predictions = solar_predictor.predict(hours)
    return {"predictions": predictions}


@router.get("/prediction/wind")
async def predict_wind(hours: int = 24):
    """风电预测"""
    predictions = wind_predictor.predict(hours)
    return {"predictions": predictions}


# ==================== 电价相关API ====================

@router.get("/price/forecast")
async def get_price_forecast(hours: int = 24, region: str = "guangdong"):
    """获取电价预测"""
    global price_predictor
    price_predictor = ElectricityPricePredictor(region=region)
    predictions = price_predictor.predict_prices(hours)
    summary = price_predictor.get_daily_price_summary()
    
    return {
        "predictions": predictions,
        "summary": summary,
        "region": region
    }


@router.get("/price/current")
async def get_current_price():
    """获取当前电价"""
    now = datetime.now()
    hour = now.hour
    
    period_type = price_predictor.get_period_type(hour)
    buy_price = price_predictor.get_buy_price(hour)
    sell_price = price_predictor.get_sell_price(hour)
    
    return {
        "hour": hour,
        "period_type": period_type,
        "buy_price": buy_price,
        "sell_price": sell_price,
        "timestamp": now.isoformat()
    }


@router.get("/price/spot")
async def get_spot_prices(hours: int = 24):
    """获取现货市场电价预测"""
    predictions = market_predictor.predict_spot_prices(hours)
    return {"predictions": predictions}


# ==================== 储能系统API ====================

@router.get("/storage/status")
async def get_storage_status():
    """获取储能系统状态"""
    return energy_storage.get_status()


@router.post("/storage/command")
async def storage_command(cmd: StorageCommand):
    """储能系统控制命令"""
    if cmd.action == "charge":
        result = energy_storage.charge(cmd.power, cmd.duration)
    elif cmd.action == "discharge":
        result = energy_storage.discharge(cmd.power, cmd.duration)
    elif cmd.action == "standby":
        energy_storage.standby()
        result = {"success": True, "message": "已切换至待机状态"}
    else:
        raise HTTPException(status_code=400, detail="无效的命令")
    
    return result


@router.get("/storage/config")
async def get_storage_config():
    """获取储能系统配置"""
    return energy_storage.config


# ==================== 储能详情API ====================

@router.get("/storage/detail/overview")
async def get_storage_overview():
    """获取储能系统总览"""
    return storage_system.get_overview()


@router.get("/storage/detail/containers")
async def get_all_containers():
    """获取所有集装箱"""
    return storage_system.get_all_containers()


@router.get("/storage/detail/container/{container_id}")
async def get_container_detail(container_id: str):
    """获取集装箱详情"""
    result = storage_system.get_container_detail(container_id)
    if not result:
        raise HTTPException(status_code=404, detail="集装箱不存在")
    return result


@router.get("/storage/detail/cluster/{container_id}/{cluster_id}")
async def get_cluster_detail(container_id: str, cluster_id: str):
    """获取电池簇详情"""
    result = storage_system.get_cluster_detail(container_id, cluster_id)
    if not result:
        raise HTTPException(status_code=404, detail="电池簇不存在")
    return result


@router.get("/storage/detail/cells/{container_id}/{cluster_id}/{pack_id}")
async def get_pack_cells(container_id: str, cluster_id: str, pack_id: str):
    """获取Pack内所有电芯"""
    result = storage_system.get_pack_cells(container_id, cluster_id, pack_id)
    if not result:
        raise HTTPException(status_code=404, detail="Pack不存在")
    return result


@router.get("/storage/detail/cell/{cell_id}")
async def get_cell_detail(cell_id: str):
    """获取单个电芯详情"""
    result = storage_system.get_cell_by_id(cell_id)
    if not result:
        raise HTTPException(status_code=404, detail="电芯不存在")
    return result


@router.get("/storage/architecture")
async def get_storage_architecture():
    """获取储能系统架构信息"""
    return storage_system.get_system_architecture()


@router.get("/storage/detail/pack-hardware/{container_id}/{cluster_id}/{pack_id}")
async def get_pack_hardware(container_id: str, cluster_id: str, pack_id: str):
    """获取Pack硬件详情（BMS+DCDC）"""
    result = storage_system.get_pack_hardware(container_id, cluster_id, pack_id)
    if not result:
        raise HTTPException(status_code=404, detail="Pack不存在")
    return result


# ==================== 智能调度API ====================

@router.post("/optimization/smart-schedule")
async def smart_schedule(hours: int = 24):
    """智能调度优化"""
    # 获取预测数据
    load_pred = load_predictor.predict(hours)
    solar_pred = solar_predictor.predict(hours)
    wind_pred = wind_predictor.predict(hours)
    
    load_forecast = [p['predicted_value'] for p in load_pred]
    solar_forecast = [p['predicted_value'] for p in solar_pred]
    wind_forecast = [p['predicted_value'] for p in wind_pred]
    
    # 执行优化
    result = smart_scheduler.optimize_daily_schedule(
        load_forecast, solar_forecast, wind_forecast,
        initial_soc=energy_storage.soc
    )
    
    return result


@router.get("/optimization/realtime-strategy")
async def get_realtime_strategy():
    """获取实时调度策略"""
    # 获取当前状态
    now = datetime.now()
    hour = now.hour
    
    # 模拟当前数据
    current_load = simulator.calculate_load(hour)
    current_solar = simulator.generate_solar_power(hour)
    current_wind = simulator.generate_wind_power(hour)
    
    strategy = smart_scheduler.get_realtime_strategy(
        hour, current_load, current_solar, current_wind, energy_storage.soc
    )
    
    return strategy


@router.post("/optimization/dispatch")
async def optimize_dispatch(hours: int = 24):
    """经济调度优化"""
    load_pred = load_predictor.predict(hours)
    solar_pred = solar_predictor.predict(hours)
    wind_pred = wind_predictor.predict(hours)
    
    load_forecast = [p['predicted_value'] for p in load_pred]
    solar_forecast = [p['predicted_value'] for p in solar_pred]
    wind_forecast = [p['predicted_value'] for p in wind_pred]
    
    result = scheduler.optimize(load_forecast, solar_forecast, wind_forecast)
    
    return result


# ==================== 实时数据WebSocket ====================

@router.websocket("/ws/realtime")
async def websocket_endpoint(websocket: WebSocket):
    """实时数据WebSocket"""
    await manager.connect(websocket)
    try:
        hour = 0
        while True:
            # 模拟实时数据
            data = simulator.simulate_step(hour % 24)
            
            # 获取当前电价
            current_hour = datetime.now().hour
            price_info = price_predictor.predict_prices(1)[0]
            
            # 添加储能状态
            storage_status = energy_storage.get_status()
            
            # 执行智能调度
            strategy = smart_scheduler.get_realtime_strategy(
                current_hour, data['load_power'], data['solar_power'],
                data['wind_power'], energy_storage.soc
            )
            
            # 根据策略控制储能
            if strategy['action'] == 'charge':
                energy_storage.charge(strategy['power'], 1/60)  # 每分钟更新
            elif strategy['action'] == 'discharge':
                energy_storage.discharge(strategy['power'], 1/60)
            
            response_data = {
                **data,
                'timestamp': datetime.now().isoformat(),
                'storage': {
                    'soc': energy_storage.soc,
                    'power': energy_storage.current_power,
                    'status': energy_storage.status,
                    'available_charge': energy_storage.get_available_charge_power(),
                    'available_discharge': energy_storage.get_available_discharge_power(),
                },
                'price': {
                    'period_type': price_info['period_type'],
                    'buy_price': price_info['buy_price'],
                    'sell_price': price_info['sell_price'],
                },
                'strategy': strategy,
            }
            
            await websocket.send_text(json.dumps(response_data))
            hour += 1
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# ==================== 系统状态API ====================

@router.get("/status")
async def get_system_status():
    """获取系统状态"""
    return {
        "status": "running",
        "simulator_params": simulator.parameters,
        "current_soc": simulator.current_soc,
        "storage_status": energy_storage.get_status(),
        "current_price": {
            "period_type": price_predictor.get_period_type(datetime.now().hour),
            "buy_price": price_predictor.get_buy_price(datetime.now().hour),
            "sell_price": price_predictor.get_sell_price(datetime.now().hour),
        },
        "timestamp": datetime.now().isoformat()
    }


@router.get("/parameters")
async def get_parameters():
    """获取当前参数"""
    return simulator.parameters


@router.post("/parameters")
async def update_parameters(params: dict):
    """更新参数"""
    simulator.update_parameters(params)
    scheduler.parameters.update(params)
    return {"status": "success", "parameters": simulator.parameters}
