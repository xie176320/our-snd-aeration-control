# 贡献指南

感谢参与 OUR-SND 数据建模与曝气决策支持项目。提交修改前，请先确认它不会把真实原始实验数据、账号信息或现场控制凭据带入仓库。

## 本地开发

```bash
git clone https://github.com/xie176320/our-snd-aeration-control.git
cd our-snd-aeration-control
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -e ".[web,plot,dev]"
python -m unittest discover -s tests -v
```

## 提交流程

1. 从 `main` 创建单一目的的分支，例如 `feat/add-do-feature`。
2. 新增行为必须配套测试；涉及模型评价时必须继续按日期/批次分组。
3. 只提交合成或明确脱敏的数据，不提交 `data/raw/`、`data/processed/` 和 `outputs/`。
4. 在 Pull Request 中说明研究假设、验证方法、结果变化和适用边界。
5. 确认 `python -m unittest discover -s tests -v` 与 `bash scripts/run_demo.sh` 均通过。

## 科研可信度要求

- 不使用随机记录拆分成绩替代跨日期验证。
- 不把由 TN 去除率推导的出水 TN 当作独立实测目标。
- 不把合成数据结果写成真实工程效果。
- 不删除外推警告、误差裕量或 A/B/C 级控制边界。
- 新增特征时报告同一分组规则下的消融结果。

发现安全问题请按 [SECURITY.md](SECURITY.md) 私下报告，不要公开创建 Issue。
