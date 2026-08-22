# 运行手册

## 1. 安装

建议每个仓库使用独立虚拟环境。Python 版本需为 3.10 或更高。

```bash
python -m venv .venv
```

激活后安装：

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[plot,dev]"
```

如果暂时不需要图和开发工具，可使用：

```bash
python -m pip install -e .
```

检查入口：

```bash
snd-control --help
snd-control schema
```

## 2. 一键本地网页导入

macOS / Linux：

```bash
bash scripts/start_local.sh
```

Windows：双击 `scripts\start_local.cmd`，或在命令提示符运行它。脚本自动建立
虚拟环境、安装网页依赖并打开 `http://127.0.0.1:8501`。左侧导入 CSV 后，
程序先执行与 CLI 相同的数据质量闸门；只有校验通过才会在当前会话内存中建模。

本地导入不会写入仓库或 `outputs/`。关闭网页终端即可结束进程。公开部署默认
没有文件选择器，不要在公网主机上手工设置 `SND_APP_MODE=local`。

## 3. 审计 Excel

```bash
snd-control audit-sources \
  --legacy-water "路径/legacy_water.xlsx" \
  --legacy-our "路径/legacy_our.xlsx" \
  --current-water "路径/current_water.xlsx" \
  --mbr-our "路径/current_reactor_our.xlsx" \
  --output-dir outputs/source-audit
```

先查看 `quality_issues.csv`，再处理 `model_input_fill_template.csv`。处理原则：

1. 温度和曝气量必须来自实验/运行记录，并按 `日期 + source_sample_index` 对齐。
2. 负硝氮或亚硝氮结果根据检测方法转成显式 `<LOQ` 状态；不要直接改成普通零值。
3. 日期错配必须由实验记录确认，不自动采用“前一天 OUR”。
4. 保留独立实测出水 TN；不要只保存由去除率反推的值。

建议把确认后的文件保存到被忽略的 `data/processed/model_input_v1.csv`。

## 4. 数据校验

```bash
snd-control validate --data data/processed/model_input_v1.csv
```

退出码为 0 表示可进入 V4；退出码为 2 表示字段、空值、范围、记录数或日期组数仍未通过。V4 最少要求 40 条完整有效记录和 5 个日期组。

单行缺失或越界会先列为排除警告；只要排除后仍有至少 40 条完整记录和 5 个日期组，训练可以继续。训练目录中的 `input_validation.json` 和 `input_excluded_rows.csv` 会保留汇总与 CSV 原始行号，便于追溯。

## 5. 训练

```bash
snd-control train \
  --data data/processed/model_input_v1.csv \
  --output-dir outputs/model-v4
```

训练不使用随机记录拆分参与选型。程序依次执行固定日期 5 折、10 次重复日期 5 折和留一日期验证，随后保存模型包。

使用任一符合契约的私有 CSV：

```bash
snd-control validate --data data/processed/model_input.csv
snd-control train --data data/processed/model_input.csv --output-dir outputs/private-v4
```

程序自动尝试 UTF-8、UTF-8 BOM 和 GBK 编码。训练后优先查看：

1. `model_evaluation.csv`：模型跨日期总体成绩；
2. `date_error_summary.csv`：哪些日期误差最大；
3. `aeration_effect_diagnostic.csv`：曝气梯度是否足以支持优化；
4. `input_excluded_rows.csv`：哪些原始行未进入训练及原因。

如果需要训练后立即输入新工况：

```bash
snd-control train --data data/processed/model_input_v1.csv --output-dir outputs/model-v4 --interactive
```

## 6. OUR 特征消融

若 CSV 同时包含 AOB/NOB 实时 OUR，可在完全相同的日期分组规则下比较特征组合：

```bash
snd-control ablate-our \
  --data data/processed/model_input.csv \
  --output outputs/private-v4/realtime_our_ablation.csv
```

重点看“重复分组 R² 均值、标准差和稳定得分”，不要因单次固定 5 折略高
就选更复杂的特征组。特征选择结论必须来自授权数据的本地输出。

## 7. 当日三点校准模型

该模式用于满足“当日已有少量实测结果，再预测当天其余曝气工况”的场景：

```bash
snd-control train-calibrated \
  --data data/processed/model_input.csv \
  --output-dir outputs/private-v4c \
  --required-r2 0.8
```

程序同时执行两种验证：一是整日留出；二是按时间滚动、只用更早日期训练。两种方式都在目标日期中选择低、中、高三个不同曝气水平作为校准点，用正则化残差截距和斜率校准，并只在其余记录上计算 R²。任一目标在任一验证方式下没有严格超过 `--required-r2` 时返回退出码 3。

校准 CSV 至少包含三个不同曝气水平、完整模型输入，以及实测 `TN去除率`、`SND率`。待预测工况必须属于同一日期，且曝气量位于校准最小值和最大值之间：

```bash
snd-control predict-calibrated \
  --model outputs/private-v4c/calibrated_model_bundle.joblib \
  --calibration configs/calibration_three_point.example.csv \
  --condition configs/calibrated_condition.example.json
```

三个校准工况应在安全混合下限以上设置，并分别稳定至少 1 个 HRT 后测定。该模式不是完全未见日期的冷启动模型，不能在没有当日 TN/SND 实测值时使用。

## 8. 独立冷启动预测

复制并修改 `configs/prediction_condition.example.json`。至少提供最大 OUR、温度、当前曝气量、进水 TN/COD；异养菌实时 OUR 可省略，此时程序会按训练数据近邻估算并明确提示。

```bash
snd-control predict \
  --model outputs/model-v4/model_bundle.joblib \
  --condition configs/prediction_condition.example.json \
  --tn-standard 15 \
  --minimum-safe-aeration 3.8
```

`--minimum-safe-aeration` 是生化池混合曝气的硬下限。上面的 `3.8 L/min` 只是
命令格式示例，必须替换为经现场确认的数值。有效下限取配置值与训练数据历史下限
中的较大者；若配置值高于历史曝气上限，程序会拒绝给出模型建议并要求补充数据。
该值不包含 MBR 膜擦洗空气需求，二者需分别核算和联锁。

建议先在历史回放或受控小试中验证，再用于任何现场操作。B 级建议执行后至少稳定一个 HRT，并复测 TN、NH₄⁺-N、NO₃⁻-N、NO₂⁻-N 和 DO。

## 9. 程序自检

```bash
python -m unittest discover -s tests -v
coverage run --source=src/wastewater_snd -m unittest discover -s tests -v
coverage report --fail-under=60
```

端到端演示：

```bash
snd-control demo-data --output outputs/demo/demo_model_input.csv
snd-control train --data outputs/demo/demo_model_input.csv --output-dir outputs/demo-model
snd-control predict --model outputs/demo-model/model_bundle.joblib --condition configs/prediction_condition.example.json
```

演示数据是合成数据，只用于软件验收。
