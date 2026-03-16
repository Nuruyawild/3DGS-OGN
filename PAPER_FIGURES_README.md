# 论文图表生成说明

本说明描述如何自动生成论文所需的图像文件。

## 图表清单与生成方式

### 图5-1：典型场景下与基线方法的定性对比（轨迹示意）

- **输出文件**：`fig_5_1_trajectory_comparison_scene1.png`、`fig_5_1_trajectory_comparison_scene2.png`、`fig_5_1_trajectory_comparison_failure1.png` 等
- **生成方式**：在导航评估时启用 `--save_paper_figures 1`
- **命令示例**：
  ```bash
  python run_enhanced.py --eval 1 --load <model_path> --save_paper_figures 1 --paper_figures_dir ./tmp/paper_figures
  ```
- **说明**：评估过程中会自动选择 1～2 个成功案例和 1～2 个失败案例，在 2D 语义地图上绘制轨迹、起点、终点、目标物体位置，保存为高分辨率 PNG（300 DPI）。

### 图5-2：低保真场景问题与替换策略示意

- **输出文件**：`fig_5_3_rendering_quality_comparison.png`
- **生成方式**：运行场景重建评估脚本
- **命令示例**：
  ```bash
  python eval_reconstruction.py --output_dir ./tmp/paper_figures --num_scenes 2
  ```
- **说明**：对比高质量（更多 Gaussian 聚类）与低质量（更少聚类）的 3DGS 渲染结果，包含 RGB、深度、语义分割图。

### 图5-3：典型失败案例与改进前后对比

- **输出文件**：`fig_5_4_failure_case_comparison.png`
- **生成方式**：与图5-1 相同，在评估时启用 `--save_paper_figures 1`
- **说明**：当存在失败案例时，自动生成改进前（失败轨迹）与改进后（示意路径）的对比图。

### 图5-4：Web 可视化系统界面示意

- **输出文件**：`fig_5_5_web_ui_main.png`
- **生成方式**：
  1. 启动 Web 服务：`python web_app/app.py`
  2. 在浏览器打开 http://localhost:5000
  3. 点击导航栏中的「📷 保存论文图」按钮
  4. 图片将下载到本地，并可选保存到 `./tmp/paper_figures/`
- **说明**：使用 html2canvas 捕获当前界面，2x 缩放以提高分辨率。

## 输出目录

默认输出目录：`./tmp/paper_figures/`

可通过 `--paper_figures_dir` 或 `--output_dir` 指定。

## 依赖

- matplotlib（用于绘图）
- numpy
- 图5-2 需要：torch, models.gaussian_splatting, models.semantic_utils
- 图5-4 需要：html2canvas（已通过 CDN 引入）

## 注意事项

1. **图5-1 / 图5-3**：需先完成模型训练，并使用 `--eval 1` 进行评估；评估需运行足够 episode 以遇到成功和失败案例。
2. **图5-2**：当前使用合成数据进行演示；若需真实 Habitat 场景，可扩展 `eval_reconstruction.py` 的 `--use_habitat 1` 逻辑。
3. **基线对比**：图5-1 的基线轨迹需分别运行基线方法（如 `use_3dgs=0`）与本文方法，将两次轨迹数据传入 `save_trajectory_comparison` 的 `trajectory_baseline` 参数。
