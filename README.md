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

ONNX 默认导出到项目工作目录下的 `exports/`。通过仓库中的启动脚本运行时，该目录位于仓库根目录；也可以在项目设置中改为其他绝对或相对路径。

全局设置中的“最大并行模型训练任务”默认是 `1`，可设置为 `1–64`。该限制只统计平台启动的本地调参训练；等待队列保存在 SQLite 中，后端重启后会继续调度。

每台远程服务器拥有独立的训练队列，可在服务器设置中调整并发上限（默认 `1`，范围 `1–64`）。同一服务器内的同一实验始终串行，其他实验可使用剩余名额。左上角“模型训练列表”按本地和服务器分组，支持查看状态、组内调整顺序和取消等待任务；浮层最高占网页可视高度的三分之一，超出后滚动。

远程任务提交时保存参数、模型及数据集路径，轮到执行时才读取路径下的最新数据并上传、生成训练前统计快照。刷新远程结果不会重算历史统计。平台后端每 30 秒核对远程进程，确认完成或失败后启动下一项；关闭网页不影响调度，停止后端会暂停后续派发但不终止远程进程。重新启动后端后先核对状态，连接失败或启动结果不明时保留名额，列表中可点击“重新检查”。只有平台登记的远程训练会占用队列名额，显存监控不作为自动调度条件。

远程执行器面向 Linux 单 GPU（GPU 0），使用进程锁和启动标识防止同一 Trial 重复运行。平台后端应以单进程运行；队列恢复会接管该数据库中的任务。旧版或外部导入任务缺少可核实的进程信息时会显示状态待确认，而不会擅自释放名额。

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
POST   /api/training-tasks/{queue_id}/recheck
POST   /api/experiments/{experiment_id}/trials/remote-run
GET    /api/remote-servers/{remote_server_id}/gpu-status
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
