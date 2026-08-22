# 部署与运行手册

## 1. 最快验收：GitHub Codespaces

1. 打开仓库，选择 **Code → Codespaces → Create codespace**。
2. 等待开发容器自动安装项目。
3. 在终端运行：

```bash
bash scripts/run_demo.sh
streamlit run streamlit_app.py
```

端口面板会显示 `8501`，点击即可打开网页。演示只使用公开合成数据。

## 2. 本地 Python

要求 Python 3.10–3.12。

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[web,plot,dev]"
python -m unittest discover -s tests -v
streamlit run streamlit_app.py
```

浏览器访问 `http://localhost:8501`。

## 3. Docker

```bash
docker compose up --build
```

容器以非 root 用户运行，并提供 `/_stcore/health` 健康检查。访问 `http://localhost:8501`。停止服务：

```bash
docker compose down
```

## 4. Streamlit Community Cloud

仓库公开后可使用免费社区部署：

1. 登录 Streamlit Community Cloud 并连接 GitHub。
2. 选择 `xie176320/our-snd-aeration-control`。
3. 分支选择 `main`，入口选择 `streamlit_app.py`。
4. 不配置任何密钥，点击 Deploy。

`requirements.txt` 会安装网页与绘图依赖。公开实例只提供合成演示；不要上传包含个人、实验室或污水厂敏感信息的数据。

## 5. 真实数据离线运行

真实文件只放在被 Git 忽略的 `data/processed/`：

```bash
snd-control validate --data data/processed/model_input_v1.csv
snd-control train --data data/processed/model_input_v1.csv --output-dir outputs/model-v4
snd-control predict \
  --model outputs/model-v4/model_bundle.joblib \
  --condition configs/prediction_condition.example.json
```

模型文件基于 pickle/joblib 机制，只加载自己训练或可信来源生成的文件。

## 6. CI 验收标准

每次推送或 Pull Request 自动执行：

- Python 3.10 与 3.12 单元测试；
- Python 源码编译检查；
- 合成数据端到端训练与预测；
- Streamlit 健康检查；
- Docker 镜像构建。

只有这些检查通过，才建议合并到 `main`。
