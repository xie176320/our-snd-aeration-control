# Python 源代码

实际包位于 `src/wastewater_snd/`：

- `sources.py`：四类 Excel 解析、日期标准化、质量审计和训练 CSV 校验。
- `model_v4.py`：日期等权、小样本集成、验证、模型保存与曝气推荐。
- `calibrated.py`：整日留出、同日低/中/高三点残差校准及范围内插值预测。
- `ablation.py`：在相同日期分组模型下比较最大 OUR 与实时 OUR 特征组合。
- `cli.py`：`snd-control` 的审计、校验、训练、校准、消融、预测和演示命令。

模块不硬编码用户数据路径或敏感信息。
