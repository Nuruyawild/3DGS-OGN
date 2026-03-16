# Object-Goal-Navigation 代码使用说明

本文档详细介绍本项目的训练、评估、论文图片输出以及 Web 可视化系统的使用方法。

---

## 目录

1. [环境准备](#1-环境准备)
2. [训练](#2-训练)
3. [评估](#3-评估)
4. [论文图片输出](#4-论文图片输出)
5. [Web 可视化系统](#5-web-可视化系统)
6. [常用参数速查](#6-常用参数速查)

---

## 1. 环境准备

### 1.1 依赖安装

**基础依赖（Habitat + PyTorch）：**

请参考项目根目录 `README.md` 安装：

- habitat-sim (v0.1.5)
- habitat-lab (v0.1.5)
- PyTorch (建议 1.6+)
- detectron2

**项目依赖：**

```bash
cd Object-Goal-Navigation
pip install -r requirements.txt
# 若使用增强版或 Web 应用，还需：
pip install -r requirements_enhanced.txt
```

### 1.2 数据集准备

**场景数据集（Gibson）：**

- 下载 Gibson 数据集：https://github.com/facebookresearch/habitat-lab#scenes-datasets
- 解压到 `data/scene_datasets/gibson_semantic/`

**Episode 数据集：**

```bash
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1tslnZAkH8m3V5nP8pbtBmaR2XEfr8Rau' -O objectnav_gibson_v1.1.zip
unzip objectnav_gibson_v1.1.zip -d data/datasets/objectnav/gibson/v1.1/
```

**目录结构：**

```
Object-Goal-Navigation/
  data/
    scene_datasets/
      gibson_semantic/
        Adrian.glb
        ...
    datasets/
      objectnav/
        gibson/
          v1.1/
            train/
            val/
```

### 1.3 验证安装

```bash
python test.py --agent random -n 1 --num_eval_episodes 1 --auto_gpu_config 0
```

若运行无报错，则环境配置正确。

---

## 2. 训练

本项目提供两种训练脚本：**基础版**（`main.py`）和**增强版**（`run_enhanced.py`）。

### 2.1 基础版训练（main.py）

基于原始 SemExp 模型，使用语义探索策略进行 Object-Goal Navigation 训练。

**基本命令：**

```bash
python main.py
```

**常用参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-d` / `--dump_location` | `./tmp/` | 模型与日志保存根目录 |
| `--exp_name` | `exp1` | 实验名称 |
| `--split` | `train` | 数据集划分（train / val） |
| `-n` / `--num_processes` | 5 | 并行环境数量 |
| `--num_training_frames` | 10000000 | 总训练帧数 |
| `--save_periodic` | 500000 | 周期性保存间隔 |
| `--lr` | 2.5e-5 | 学习率 |

**示例：**

```bash
# 指定实验名和保存路径
python main.py -d ./experiments/ --exp_name my_exp1

# 减少并行数以节省显存
python main.py -n 2
```

**输出：**

- 模型：`{dump_location}/models/{exp_name}/model_best.pth`
- 周期保存：`{dump_location}/dump/{exp_name}/periodic_*.pth`
- 日志：`{dump_location}/models/{exp_name}/train.log`

### 2.2 增强版训练（run_enhanced.py）

在基础版上增加 3D Gaussian Splatting、场景图、Transformer 策略、多维奖励等模块。

**基本命令：**

```bash
python run_enhanced.py
```

**增强模块开关：**

| 参数 | 默认 | 说明 |
|------|------|------|
| `--use_3dgs` | 1 | 是否使用 3D Gaussian Splatting |
| `--use_scene_graph` | 1 | 是否使用场景图 |
| `--use_transformer` | 1 | 是否使用 Transformer 策略 |
| `--use_enhanced_reward` | 1 | 是否使用多维奖励 |
| `--use_enhanced_policy` | 1 | 是否使用增强策略网络 |

**示例：**

```bash
# 完整增强版训练
python run_enhanced.py -d ./experiments/ --exp_name enhanced_exp

# 关闭 3DGS，仅用基础语义探索（类似 main.py）
python run_enhanced.py --use_3dgs 0 --exp_name baseline_exp

# 自定义奖励权重
python run_enhanced.py --direction_reward_weight 1.2 --collision_reward_weight 2.5
```

**输出：**

与 `main.py` 相同，模型和日志保存在 `{dump_location}/models/{exp_name}/` 和 `{dump_location}/dump/{exp_name}/`。

---

## 3. 评估

### 3.1 基础版评估（main.py）

```bash
python main.py --split val --eval 1 --load pretrained_models/sem_exp.pth
```

**参数说明：**

- `--split val`：使用验证集
- `--eval 1`：评估模式（不更新模型）
- `--load <path>`：加载的模型路径

**可视化评估过程：**

```bash
python main.py --split val --eval 1 --load pretrained_models/sem_exp.pth -v 1
```

`-v 1` 会渲染观测与预测语义地图。

### 3.2 增强版评估（run_enhanced.py）

```bash
python run_enhanced.py --split val --eval 1 --load ./tmp/models/exp1/model_best.pth
```

**评估指标：**

- **Success Rate (SR)**：成功率
- **SPL**：Success weighted by Path Length
- **DTG**：Distance to Goal（平均距离）

评估结束后，结果会写入 `{dump_location}/dump/{exp_name}/`，包括：

- `val_spl_per_cat_pred_thr.json`
- `val_success_per_cat_pred_thr.json`
- `enhanced_metrics.json`（增强版）

### 3.3 预训练模型

```bash
mkdir -p pretrained_models
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=171ZA7XNu5vi3XLpuKs8DuGGZrYyuSjL0' -O pretrained_models/sem_exp.pth
```

该预训练模型在验证集上约达到：Success 0.657，SPL 0.339，DTG 1.474。

---

## 4. 论文图片输出

评估或专用脚本运行时可自动生成论文所需的高分辨率图片（300 DPI）。

### 4.1 图 5-1：轨迹对比（成功/失败案例）

在 2D 语义地图上绘制本文方法轨迹、起点、终点、目标物体位置。

**生成方式：** 评估时启用 `--save_paper_figures 1`

```bash
python run_enhanced.py --split val --eval 1 --load <model_path> \
    --save_paper_figures 1 --paper_figures_dir ./tmp/paper_figures
```

**输出文件：**

- `fig_5_1_trajectory_comparison_scene1.png`（成功案例 1）
- `fig_5_1_trajectory_comparison_scene2.png`（成功案例 2）
- `fig_5_1_trajectory_comparison_failure1.png`（失败案例 1）
- `fig_5_1_trajectory_comparison_failure2.png`（失败案例 2）

### 4.2 图 5-2：渲染质量对比（高/低质量 3DGS）

对比高质量与低质量 3D Gaussian Splatting 渲染（RGB、深度、语义）。

**生成方式：** 运行场景重建评估脚本

```bash
python eval_reconstruction.py --output_dir ./tmp/paper_figures --num_scenes 2
```

**输出文件：** `fig_5_3_rendering_quality_comparison.png`

### 4.3 图 5-3：失败案例改进前后对比

展示改进前（失败）与改进后（成功）的轨迹对比。

**生成方式：** 与图 5-1 相同，评估中若存在失败案例会自动生成。

**输出文件：** `fig_5_4_failure_case_comparison.png`

### 4.4 图 5-4：Web 界面截图

**生成方式：**

1. 启动 Web 服务（见下文）
2. 在浏览器打开 http://localhost:5000
3. 点击导航栏「📷 保存论文图」按钮
4. 图片会下载为 `fig_5_5_web_ui_main.png`，并可选保存到 `./tmp/paper_figures/`

### 4.5 输出目录

默认：`./tmp/paper_figures/`，可通过 `--paper_figures_dir` 或 `--output_dir` 修改。

---

## 5. Web 可视化系统

### 5.1 启动方式

```bash
cd Object-Goal-Navigation
python web_app/app.py
```

**可选参数：**

```bash
python web_app/app.py --port 5000 --host 0.0.0.0 --debug
```

启动后访问：**http://localhost:5000**

### 5.2 功能模块（4.5.1 / 4.5.2 / 4.5.3）

#### （1）3D 可视化（4.5.1）

- **3D 高斯场景实时渲染**：展示 3D 高斯点云
- **多视角观察**：默认、语义着色、俯视
- **交互式浏览**：OrbitControls 旋转/缩放
- **导航路径可视化**：轨迹线、起点/终点标注
- **Agent 实时位置**：蓝色锥体表示
- **目标物体选择**：下拉选择 15 类目标

#### （2）导航监控（4.5.2）

- **性能指标图表**：SR、SPL、DTG
- **导航路径俯视图**：2D 轨迹绘制
- **实时指标**：SR、SPL、DTG 数值

#### （3）语义地图（4.5.1）

- **语义标签可视化**：语义分割网格
- **语义热力图**：不同类别颜色区分

#### （4）训练与评估（4.5.2）— 一键操作

- **一键训练**：选择实验名、并行进程数，点击「开始」
- **一键评估**：选择模型、Episode 数、是否保存论文图，点击「开始」
- **实时输出**：终端日志实时显示在页面
- **实时指标**：SR、SPL、DTG、Steps、FPS 从日志解析并更新
- **停止**：点击「停止」终止当前任务

#### （5）实验管理（4.5.3）

- **实验结果列表**：查看所有实验及 SR/SPL
- **实验详情**：Success、SPL、DTG
- **导出**：导出实验结果为 JSON

#### （6）数据集（4.5.3）

- **Gibson / MP3D 浏览**：场景列表
- **物体类别统计**：柱状图展示各类别观测数

### 5.3 API 接口

| 接口 | 说明 |
|------|------|
| `GET /api/state` | 导航状态 |
| `POST /api/state/update` | 更新状态（训练脚本可调用） |
| `GET /api/categories` | 目标类别 |
| `POST /api/goal/set` | 设置目标 |
| `POST /api/jobs/start` | 启动训练/评估 |
| `POST /api/jobs/stop` | 停止任务 |
| `GET /api/jobs/status` | 任务状态与日志 |
| `GET /api/jobs/models` | 可用模型列表 |
| `GET /api/experiments` | 实验列表 |
| `GET /api/experiments/<name>/export` | 导出实验 |
| `GET /api/datasets` | 数据集信息 |
| `GET /api/datasets/stats` | 类别统计 |

---

## 6. 常用参数速查

### 通用参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `-d` | 输出根目录 | `-d ./experiments/` |
| `--exp_name` | 实验名 | `--exp_name exp2` |
| `--split` | 数据划分 | `--split val` |
| `-n` | 并行进程数 | `-n 3` |
| `--load` | 加载模型 | `--load model.pth` |
| `--eval` | 评估模式 | `--eval 1` |
| `--no_cuda` | 禁用 GPU | `--no_cuda` |

### 训练相关

| 参数 | 说明 | 示例 |
|------|------|------|
| `--num_training_frames` | 总训练帧数 | `10000000` |
| `--save_periodic` | 周期保存间隔 | `500000` |
| `--lr` | 学习率 | `2.5e-5` |

### 评估相关

| 参数 | 说明 | 示例 |
|------|------|------|
| `--num_eval_episodes` | 每场景评估 episode 数 | `200` |
| `-v` | 可视化级别 | `-v 1` |
| `--save_paper_figures` | 保存论文图 | `1` |
| `--paper_figures_dir` | 论文图输出目录 | `./tmp/paper_figures` |

### 增强版特有

| 参数 | 说明 | 示例 |
|------|------|------|
| `--use_3dgs` | 启用 3DGS | `1` / `0` |
| `--use_transformer` | 启用 Transformer | `1` / `0` |
| `--use_enhanced_reward` | 启用多维奖励 | `1` / `0` |

---

## 快速命令汇总

```bash
# 训练（基础版）
python main.py -d ./tmp/ --exp_name exp1

# 训练（增强版）
python run_enhanced.py -d ./tmp/ --exp_name enhanced_exp

# 评估
python run_enhanced.py --split val --eval 1 --load ./tmp/models/exp1/model_best.pth

# 评估并生成论文图
python run_enhanced.py --split val --eval 1 --load ./tmp/models/exp1/model_best.pth --save_paper_figures 1

# 场景重建图（图 5-2）
python eval_reconstruction.py --output_dir ./tmp/paper_figures

# 启动 Web 应用
python web_app/app.py --port 5000
```

---

更多细节可参考：

- `README.md`：项目概述与 Habitat 安装
- `PAPER_FIGURES_README.md`：论文图表生成说明
