# 在 GitHub 上运行

本项目推荐使用 **GitHub Codespaces**：浏览器里会启动一台临时 Linux 开发环境，自动安装 Python、项目依赖和命令行工具。仓库中的真实水质数据与模型输出默认不会提交。

## 1. 创建 Codespace

1. 进入仓库主页。
2. 保持默认分支 `main`。
3. 点击 **Code → Codespaces → Create codespace on main**。
4. 等待终端中的依赖安装完成，然后运行：

```bash
snd-control --help
bash scripts/run_demo.sh
```

`run_demo.sh` 使用明确标注的合成数据，依次完成数据生成、校验、R² 门控训练与同日三点校准预测。结果保存在 `outputs/demo/`。

## 2. 在授权环境运行私有数据

不要把真实 CSV 提交到 Git。只在 Codespaces 当前环境中把文件上传到：

```text
data/processed/model_input.csv
```

然后执行：

```bash
bash scripts/run_private_data.sh
```

脚本会校验数据，并运行带可配置 R² 闸门的 V4-C 流程。主要结果位于：

- `outputs/private-v4c/calibrated_evaluation.csv`
- `outputs/private-v4c/calibrated_oof_predictions.csv`
- `outputs/private-v4c/calibrated_rolling_predictions.csv`
- `outputs/private-v4c/calibrated_model_bundle.joblib`

也可以显式指定输入和输出目录：

```bash
bash scripts/run_private_data.sh data/processed/model_input.csv outputs/my-run
```

## 3. 预测同日新工况

复制并修改以下两个示例文件：

- `configs/calibration_three_point.example.csv`：同日低、中、高三个曝气水平的实测 TN 去除率与 SND 率；
- `configs/calibrated_condition.example.json`：同一天、校准曝气范围内的一条待预测工况。

运行：

```bash
snd-control predict-calibrated \
  --model outputs/private-v4c/calibrated_model_bundle.joblib \
  --calibration configs/calibration_three_point.example.csv \
  --condition configs/calibrated_condition.example.json
```

R²>0.8 只对应“目标日期已经取得三个校准点，并在其曝气范围内插值”的场景，不代表完全未见日期的冷启动性能。

## 4. GitHub Actions 自动验收

每次推送或 Pull Request，`CI` 工作流都会：

1. 安装项目依赖；
2. 运行单元测试；
3. 即时生成合成数据并运行端到端演示；
4. 上传演示输出为临时 artifact。

该工作流只使用代码生成的虚构记录，不读取私有 CSV。工作流合并到默认分支后，也可以在 **Actions → CI → Run workflow** 手动触发。

## 5. 数据安全

`.gitignore` 已排除 `data/processed/`、根目录下的 CSV/Excel、`outputs/` 和模型文件。上传真实数据到 Codespaces 后，不要使用 `git add -f` 强制加入这些文件。Codespace 不再使用时可删除；重要的模型评估结果请下载到安全位置保存。
