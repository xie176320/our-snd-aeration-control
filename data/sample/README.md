# 合成数据说明

本目录不提交任何逐行测量记录。运行下列命令可在被 Git 忽略的 `outputs/`
目录即时生成固定随机种子的虚构数据：

```bash
snd-control demo-data --output outputs/demo/demo_model_input.csv
```

生成逻辑位于 `src/wastewater_snd/synthetic.py`，与任何私有工作簿或现场测量
无关。生成文件只用于验证安装、训练和预测接口，不能用于论文结论或现场控制。
