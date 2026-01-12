# 微电网数字孪生平台

## 项目简介

基于Python和Web技术的微电网数字孪生平台，集成仿真、预测、优化和3D可视化功能。
支持站级储能系统的完整建模，包括多层级电池管理、均衡算法和延寿策略。

## 功能特性

### 核心功能
1. **仿真数据管理**：连接仿真模型，实时保存和管理仿真数据
2. **数据标定**：使用实际场景数据标定仿真模型参数
3. **预测功能**：基于历史数据进行负荷和发电预测
4. **能量调度优化**：实现微电网经济调度优化算法
5. **3D可视化**：提供交互式3D可视化界面

### 储能系统建模
- **五级层级结构**：电芯 → Pack → 簇 → 箱（集装箱） → 站
- **完整硬件建模**：BMS、DCDC、PCS、EMS、SCADA等
- **状态聚合算法**：支持多种聚合方法（均值、加权均值、最小值、投票法等）

### 电池均衡算法
#### 被动均衡
- 电阻放电均衡
- 结构简单、成本低
- 适用于单体间均衡

#### 主动均衡
- DC-DC能量转移
- 高效率(85-95%)
- 支持多层级均衡：
  - **单体间均衡**：电芯级别电压/SOC均衡
  - **Pack间均衡**：Pack级别能量再分配
  - **簇间均衡**：簇级别功率分配优化
  - **箱间均衡**：集装箱级别负载均衡

#### 均衡目标
- **SOC均衡**：消除荷电状态不一致
- **SOH均衡**：通过差异化使用延长整体寿命
- **电压均衡**：快速响应，防止过充/过放

### 电池延寿策略
#### 温度优化
- 最佳工作温度控制(20-35℃)
- 低温预热策略
- 高温降额和冷却
- 温度均衡管理
- 基于Arrhenius方程的老化预测

#### DOD优化
- 限制最大放电深度
- 根据SOH动态调整DOD限制
- 浅充浅放延长循环寿命
- DOD-循环寿命预测模型

#### 倍率优化
- 充放电倍率限制
- SOC相关倍率调整
- 温度相关倍率调整
- 高倍率老化因子计算

#### SOC管理
- 避免长期高SOC存储
- 避免过度放电
- 存储状态SOC优化(40-60%)
- 日历老化管理

#### 充电策略优化
- **CC-CV充电**：标准恒流恒压
- **阶梯充电**：多阶段降低倍率
- **脉冲充电**：减少极化效应
- **自适应充电**：根据温度/SOC/SOH动态调整

#### 其他延寿策略
- **电流波纹控制**：减少交流纹波损伤
- **负载均衡**：均匀分配功率负荷
- **循环优化**：减少不必要的充放电循环
- **热管理**：预热与冷却系统协调控制

## 项目结构

```
├── backend/                 # 后端代码
│   ├── api/                 # API路由
│   ├── models.py            # 数据库模型
│   ├── simulation/          # 仿真模块
│   ├── prediction/          # 预测模块
│   └── optimization/        # 优化模块（调度、储能、均衡、延寿）
├── frontend/                # 前端代码
│   ├── static/              # 静态资源 (CSS/JS)
│   └── templates/           # HTML模板
├── main.py                  # 应用入口
├── requirements.txt         # Python依赖
└── start.bat                # Windows启动脚本
```

## 安装和运行

### 环境要求
- Python 3.8+
- pip 包管理器

### 安装步骤

1. 克隆项目：
```bash
git clone <repository-url>
cd <project-directory>
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 运行应用：
```bash
python main.py
```

4. 打开浏览器访问：`http://localhost:8000`

### Windows快速启动
双击运行 `start.bat`

## API接口

### 储能系统API
- `GET /api/storage/status` - 获取储能系统状态
- `GET /api/storage/overview` - 获取系统概览
- `GET /api/storage/containers` - 获取所有集装箱状态
- `GET /api/storage/container/{id}` - 获取指定集装箱详情
- `GET /api/storage/cluster/{container_id}/{cluster_id}` - 获取簇详情
- `GET /api/storage/pack/{container_id}/{cluster_id}/{pack_id}/cells` - 获取Pack电芯

### 均衡算法API
- `POST /api/balance/analyze` - 分析系统不一致性
- `POST /api/balance/plan` - 生成均衡计划
- `POST /api/balance/execute` - 执行均衡操作
- `GET /api/balance/status` - 获取均衡器状态

### 延寿策略API
- `GET /api/life/health` - 获取系统健康分析
- `POST /api/life/plan` - 生成延寿计划
- `GET /api/life/strategies` - 获取所有延寿策略说明
- `GET /api/life/potential` - 计算延寿潜力

### 调度优化API
- `POST /api/schedule/optimize` - 优化调度计划
- `GET /api/schedule/realtime` - 获取实时调度策略

## 技术栈

### 后端
- **FastAPI**：高性能异步Web框架
- **SQLAlchemy**：ORM数据库操作
- **SQLite**：轻量级数据库
- **NumPy/SciPy**：科学计算

### 前端
- **HTML5/CSS3**：现代Web界面
- **JavaScript**：交互逻辑
- **Three.js**：3D可视化
- **Chart.js**：图表展示

### 仿真与优化
- **NumPy**：数值计算
- **SciPy**：优化算法
- **scikit-learn**：机器学习预测

## 均衡算法详解

### 被动均衡
```python
from backend.optimization.balancing import PassiveBalancer

balancer = PassiveBalancer()
# 分析不一致性
analysis = balancer.analyze_imbalance(cells, BalanceTarget.VOLTAGE)
# 生成均衡命令
commands = balancer.generate_balance_commands(cells, BalanceTarget.SOC)
# 执行均衡
result = balancer.execute_balance(commands)
```

### 主动均衡
```python
from backend.optimization.balancing import ActiveBalancer, MultilevelBalanceManager

# 主动均衡器
balancer = ActiveBalancer()
# SOC均衡命令
soc_commands = balancer.generate_soc_balance_commands(units, BalanceLevel.PACK)
# SOH均衡命令
soh_commands = balancer.generate_soh_balance_commands(units, BalanceLevel.CLUSTER)
# 电压均衡命令
voltage_commands = balancer.generate_voltage_balance_commands(cells)

# 多层级均衡管理
manager = MultilevelBalanceManager()
plan = manager.generate_balance_plan(system_data)
result = manager.execute_balance_plan(plan)
```

## 延寿策略详解

### 温度优化
```python
from backend.optimization.life_extension import TemperatureOptimizer

optimizer = TemperatureOptimizer()
# 分析温度状态
analysis = optimizer.analyze_temperature_status(temperatures)
# 计算老化因子
aging_factor = optimizer.calculate_aging_factor(temperature=35)
# 生成热管理命令
commands = optimizer.generate_thermal_commands(units)
```

### DOD优化
```python
from backend.optimization.life_extension import DODOptimizer

optimizer = DODOptimizer()
# 推荐DOD限制
recommendation = optimizer.recommend_dod_limit(current_soh=0.9, usage_scenario="peak_shaving")
# 计算循环寿命
cycle_life = optimizer.calculate_cycle_life(dod=0.8)
```

### 综合延寿管理
```python
from backend.optimization.life_extension import BatteryLifeExtensionManager

manager = BatteryLifeExtensionManager()
# 系统健康分析
health = manager.analyze_system_health(system_data)
# 生成延寿计划
plan = manager.generate_life_extension_plan(system_data)
# 获取策略说明
strategies = manager.get_life_extension_strategies()
# 计算延寿潜力
potential = manager.calculate_total_life_extension_potential(current_state)
```

## 参考标准

- IEC 62619: 锂电池安全要求
- IEC 62620: 锂电池二次电池性能测试
- GB/T 34131-2017: 电化学储能电站安全规范
- GB/T 36276-2018: 电力储能用锂离子电池

## 更新日志

### v1.2.0 (2025-12)
- 新增电池均衡算法模块
  - 被动均衡器
  - 主动均衡器
  - 多层级均衡管理
  - 均衡效果评估
- 新增电池延寿策略模块
  - 温度优化策略
  - DOD优化策略
  - 倍率优化策略
  - SOC管理策略
  - 充电策略优化
  - 综合延寿管理

### v1.1.0 (2025-12)
- 新增五级储能系统详细模型
- 新增状态聚合算法
- 新增BMS/DCDC/PCS硬件建模

### v1.0.0 (2025-11)
- 初始版本
- 基础仿真和优化功能
- Web可视化界面

## 许可证

MIT License
