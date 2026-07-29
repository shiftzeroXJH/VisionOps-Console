# YOLO Platform

YOLO Platform 是一个面向工业视觉训练的 Web 平台，用来管理 YOLO 实验、调参训练、Trial 对比、远程结果同步、验证预览和 ONNX 导出。

## 功能

- 创建实验并按项目分组管理任务。
- 调整 YOLO 训练参数并启动本地训练。
- 导入本地或远程服务器上的已有训练结果。
- 对比 Trial 指标、训练曲线和可视化图片。
- 用 Trial 权重执行临时验证预览，对比 label / predict 图片。
- 使用本地或训练平台模型执行多图片推理，并按类别筛选叠加框。
- 评估标准 YOLO、Pascal VOC 或 LabelMe 验证集，生成逐类别指标和预测 XML。
- 导出 Trial 权重为 ONNX。
- 在设置里清理验证预览缓存。

## 环境要求

- Python 3.10+
- Node.js 18+，用于构建前端
- PyTorch，请按你的 CPU/CUDA 环境单独安装
- Windows 或 Linux

PyTorch 安装示例请以官方命令为准。平台依赖会安装 `ultralytics`，但不会固定安装 Torch，避免 CUDA 版本装错。

## 安装

```bash
python -m pip install -U pip
python -m pip install -e .
```

开发和测试依赖：

```bash
python -m pip install -r requirements-dev.txt
```

## 构建前端

部署运行前需要构建一次前端：

```bash
cd frontend
npm install
npm run build
cd ..
```

构建产物位于 `frontend/dist`。运行时默认不需要 Node 常驻进程。

## 部署模式

部署模式只启动一个 Python 后端服务，后端会托管 `frontend/dist`。

Windows：

```powershell
.\start.bat
.\stop.bat
```

Linux：

```bash
chmod +x start.sh stop.sh
./start.sh
./stop.sh
```

默认访问地址：

```text
http://127.0.0.1:8765/
```

## 开发模式

开发模式会同时启动后端和 Vite 前端开发服务器。

Windows：

```powershell
.\start-dev.bat
.\stop-dev.bat
```

Linux：

```bash
chmod +x start-dev.sh stop-dev.sh
./start-dev.sh
./stop-dev.sh
```

开发模式访问：

```text
Frontend: http://127.0.0.1:5173/
Backend:  http://127.0.0.1:8765/
```

## 手动启动

```bash
set PYTHONPATH=src
python -m backend.api
```

Linux/macOS：

```bash
export PYTHONPATH=src
python -m backend.api
```

安装为 editable 后也可以直接运行：

```bash
yolo-platform
```

## 配置

| 环境变量 | 说明 | 默认值 |
| --- | --- | --- |
| `YOLO_DB_PATH` | SQLite 数据库路径 | `yolo_state.sqlite` |
| `YOLO_HOST` | 后端监听地址 | `127.0.0.1` |
| `YOLO_PORT` | 后端端口 | `8765` |
| `YOLO_PYTHON` | 训练、验证、导出 worker 使用的 Python | 当前 Python；若存在本机默认 YOLO 环境则优先使用 |
| `YOLO_FRONTEND_DIST` | 前端构建产物目录 | 当前目录下的 `frontend/dist` |

旧的 `openclaw_yolo_state.sqlite` 如果存在，且新的 `yolo_state.sqlite` 不存在，首次启动会自动复制为新数据库。

## API

主要接口：

```text
GET    /health
GET    /api/experiments
POST   /api/experiments
GET    /api/experiments/{experiment_id}
PATCH  /api/experiments/{experiment_id}
DELETE /api/experiments/{experiment_id}
POST   /api/experiments/{experiment_id}/trials/run
POST   /api/experiments/{experiment_id}/trials/import
GET    /api/experiments/{experiment_id}/comparison
GET    /api/experiments/{experiment_id}/curves
GET    /api/trials/{trial_id}/summary
PATCH  /api/trials/{trial_id}
POST   /api/trials/{trial_id}/validate-preview
POST   /api/trials/{trial_id}/export-onnx
GET    /jobs/{job_id}
GET    /api/workbench/models
POST   /api/workbench/sessions
POST   /api/workbench/sessions/{session_id}/images
POST   /api/workbench/sessions/{session_id}/infer
POST   /api/workbench/datasets/inspect
POST   /api/workbench/evaluations
```

## 测试

```bash
python -m pytest
```

前端构建验证：

```bash
cd frontend
npm run build
```

## 项目结构

```text
frontend/          React + Vite 前端
src/backend/       FastAPI API、业务服务、SQLite 仓库、YOLO workers
tests/             后端单元测试
bin/               Windows PowerShell 启停脚本
start*.bat         Windows 入口
start*.sh          Linux 入口
```
