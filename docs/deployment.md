# 部署与运行手册

## 1. 公开在线演示

公开部署使用故障安全默认配置：不设置任何环境变量或密钥时，应用固定运行在
`public` 模式。页面不会创建 CSV 上传控件，只使用运行时生成的合成数据。

在 Streamlit Community Cloud 中部署：

1. 登录 Community Cloud 并授权读取 GitHub 公共仓库。
2. 选择 `xie176320/our-snd-aeration-control`。
3. 分支选择 `main`，入口选择 `streamlit_app.py`。
4. **Secrets 保持为空**，不要设置 `SND_APP_MODE` 或 `SND_LOCAL_IMPORT`。
5. 点击 Deploy，健康检查通过后复制生成的公开网址。

`requirements.txt` 会安装网页与绘图依赖。线上界面会明确显示“公开合成演示 ·
线上文件上传已关闭”。访客可以使用四种典型工况并下载 Markdown/JSON 报告，
报告仍只包含合成工况与计算结果。公开主机上不得开启本地导入双开关。

## 2. 一键本地导入

要求 Python 3.10–3.12。脚本会自动创建 `.venv`、安装缺少的依赖、同时设置
两个本地开关，并把服务绑定到 `127.0.0.1`。

macOS / Linux：

```bash
bash scripts/start_local.sh
```

Windows：双击 `scripts\start_local.cmd`，或运行：

```bat
scripts\start_local.cmd
```

访问 `http://127.0.0.1:8501`，从侧栏选择标准 CSV。文件在当前 Streamlit
会话内存中解析和建模，不写入工作目录、仓库或公开实例。关闭运行脚本的终端
即可结束服务。

如需其他端口，可在启动前设置 `SND_PORT`。例如：

```bash
SND_PORT=8601 bash scripts/start_local.sh
```

## 3. Docker 本地导入

```bash
docker compose up --build
```

Compose 同时完成三件事：绑定宿主机 `127.0.0.1:8501`、开启本地导入双开关、
配置容器健康检查。浏览器访问 `http://127.0.0.1:8501`。停止服务：

```bash
docker compose down
```

Dockerfile 本身不设置本地开关。因此将镜像部署到其他平台时，仍会安全回退到
关闭上传的公开合成模式。

## 4. GitHub Codespaces 验收

1. 打开仓库，选择 **Code → Codespaces → Create codespace**。
2. 等待开发容器自动安装项目。
3. 在终端运行：

```bash
bash scripts/run_demo.sh
streamlit run streamlit_app.py
```

端口面板会显示 `8501`。Codespaces 默认只运行公开合成演示；它不是本机导入
入口，不建议把敏感原始数据上传到公开演示或临时云环境。

## 5. 真实数据命令行运行

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
- 典型工况与报告导出数据边界测试；
- 合成数据端到端训练与预测；
- `public` 与 `local` 两种网页模式健康检查；
- Docker 镜像构建。

模式单元测试还会验证：默认值、未知值或只设置一个开关时，文件导入始终关闭。
只有这些检查通过，才建议合并到 `main`。
