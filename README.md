# YOLO Platform

YOLO Platform 是一个面向工业视觉训练的 Web 平台，用来管理 YOLO 实验、调参训练、Trial 对比、远程结果同步、验证预览和 ONNX 导出。

## 功能

- 创建实验并按项目分组管理任务。
- 调整 YOLO 训练参数并启动本地训练。
- 通过持久化训练队列限制全局并行任务数，并调整或取消等待任务。
- 导入本地或远程服务器上的已有训练结果。
- 对比 Trial 指标、训练曲线和可视化图片。
- 用 Trial 权重执行临时验证预览，对比 label / predict 图片。
- 使用本地或训练平台模型执行多图片推理，并按类别筛选叠加框。
- 评估标准 YOLO、Pascal VOC 或 LabelMe 验证集，生成逐类别指标和预测 XML。
- 导出 Trial 权重为 ONNX。
- 在设置里清理验证预览缓存。

## 环境要求

- Python 3.10+
- Node.js 20.19+，或 22.12+
- 与本机 CPU/CUDA 环境匹配的 PyTorch 和 torchvision
- Windows 或 Linux

当前前端使用 Vite 8，需要 Node.js 20.19+ 或 22.12+。

Ultralytics 会传递依赖 `torch` 和 `torchvision`。建议先按照 PyTorch 官方命令安装与你的 CPU/CUDA 环境匹配的版本，再安装本项目，避免 pip 自动选择不合适的构建。

## 安装

建议使用专门的 YOLO Python 环境。Windows PowerShell 示例：

```powershell
$env:YOLO_PYTHON = "D:\apps\miniforge\envs\yolo_env\python.exe"
& $env:YOLO_PYTHON -m pip install -U pip
```

请先使用 PyTorch 官方安装命令安装匹配的 `torch` 和 `torchvision`，然后安装项目和测试依赖：

```powershell
& $env:YOLO_PYTHON -m pip install -r requirements-dev.txt
```

`requirements-dev.txt` 会以 editable 方式安装当前项目，并安装 pytest 和 httpx。只需要运行环境时，也可以使用示意命令：

```bash
python -m pip install -e .
```

Linux 下请将 `YOLO_PYTHON` 设置为实际 YOLO 环境中的 Python：

```bash
export YOLO_PYTHON=/path/to/yolo_env/bin/python
"$YOLO_PYTHON" -m pip install -r requirements-dev.txt
```

## 前端依赖

开发运行前安装前端依赖：

```bash
cd frontend
npm ci
cd ..
```

开发运行不需要预先构建 `frontend/dist`。只有进行打包验证时才需要执行 `npm run build`。

## 开发运行

默认入口会同时启动 Python 后端和 Vite 前端开发服务器。

Windows：

```powershell
$env:YOLO_PYTHON = "D:\apps\miniforge\envs\yolo_env\python.exe"
.\start.bat
.\stop.bat
```

Linux：

```bash
export YOLO_PYTHON=/path/to/yolo_env/bin/python
./start.sh
./stop.sh
```

访问地址：

```text
Frontend: http://127.0.0.1:5173/
Backend:  http://127.0.0.1:8765/
```

前端使用 Vite 提供热更新。后端当前不启用自动 reload，修改 Python 代码后需要执行 `stop` 和 `start`。

## 打包验证

需要验证后端托管构建后的前端时，先构建前端：

```bash
cd frontend
npm run build
cd ..
```

Windows：

```powershell
.\bin\start-built.bat
.\bin\stop-built.bat
```

Linux：

```bash
./bin/start-built.sh
./bin/stop-built.sh
```

打包验证入口只启动一个 Python 后端服务，并托管 `frontend/dist`。

## 手动启动

Windows CMD：

```bat
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
| `YOLO_PYTHON` | 训练、验证、导出 worker 使用的 Python | 当前 Python；建议显式设置为 YOLO 环境 Python |
| `YOLO_FRONTEND_DIST` | 打包验证时的前端构建产物目录 | 当前目录下的 `frontend/dist` |

旧的 `openclaw_yolo_state.sqlite` 如果存在，且新的 `yolo_state.sqlite` 不存在，首次启动会自动复制为新数据库。

全局设置中的“最大并行模型训练任务”默认是 `1`，可设置为 `1–64`。该限制只统计平台启动的本地调参训练；等待队列保存在 SQLite 中，后端重启后会继续调度。

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
GET    /api/training-tasks
PATCH  /api/training-tasks/{queue_id}
POST   /api/training-tasks/{queue_id}/cancel
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
bin/               打包验证入口与内部启动脚本
start.bat/.sh      Windows 与 Linux 开发入口
```
