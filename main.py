"""微电网数字孪生平台 - 主入口"""
import uvicorn
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.requests import Request
from fastapi.middleware.cors import CORSMiddleware

from backend.database import init_db
from backend.api.routes import router as api_router

# 切换到项目目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化数据库
    init_db()
    yield
    # 关闭时的清理工作


# 创建FastAPI应用
app = FastAPI(
    title="微电网数字孪生平台",
    description="集成仿真、预测、优化和3D可视化的微电网数字孪生平台",
    version="1.0.0",
    lifespan=lifespan
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

# 模板配置
templates = Jinja2Templates(directory="frontend/templates")

# 注册API路由
app.include_router(api_router, prefix="/api")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """主页面"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """仪表板页面"""
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/visualization", response_class=HTMLResponse)
async def visualization(request: Request):
    """3D可视化页面"""
    return templates.TemplateResponse("visualization.html", {"request": request})


@app.get("/storage", response_class=HTMLResponse)
async def storage(request: Request):
    """储能电站详情页面"""
    return templates.TemplateResponse("storage.html", {"request": request})


@app.get("/trading", response_class=HTMLResponse)
async def trading(request: Request):
    """电力交易分析页面"""
    return templates.TemplateResponse("trading.html", {"request": request})


@app.get("/agent", response_class=HTMLResponse)
async def agent(request: Request):
    """瓦特智能体页面"""
    return templates.TemplateResponse("agent.html", {"request": request})


@app.get("/opendata", response_class=HTMLResponse)
async def opendata(request: Request):
    """开源数据页面"""
    return templates.TemplateResponse("opendata.html", {"request": request})


if __name__ == "__main__":
    print("=" * 50)
    print("微电网数字孪生平台")
    print("访问地址: http://localhost:8000")
    print("API文档: http://localhost:8000/docs")
    print("3D可视化: http://localhost:8000/visualization")
    print("=" * 50)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
