# Study Guardian AI（学习监督桌面 AI）

当前版本包含以下核心能力：
- 番茄钟模式（学习/休息循环）
- 每日学习报告图表（自动汇总今日日志）
- 多日趋势图（近14天 + 近8周）
- 玩手机监督检测（本地规则 / 本地Qwen模型）
- 可配置提醒策略（连续玩手机/离开达到阈值再提醒）

## 1. 安装依赖

```powershell
python -m pip install -r requirements.txt
```

## 2. 启动

```powershell
python app.py
```

## 3. 使用

1. 选择模式：`番茄钟` 或 `单次学习`
2. 设置学习分钟、休息分钟、循环次数
3. 设置 `提醒阈值(秒)`（例如 30）
4. 选择检测方式：`本地规则` 或 `本地Qwen模型`
5. 点击“开始监督”
6. 任何时候可点“结束监督”
7. 点击“生成今日日报”生成当天柱状图
8. 点击“生成趋势图”生成近14天和近8周趋势图

## 4. 本地Qwen模型接入（不走云端）

程序使用 OpenAI 兼容接口调用本地多模态模型。

### Ollama 示例

```powershell
# 先拉取可用的 Qwen VL 小模型（以你本机实际可用tag为准）
ollama pull qwen2.5vl:3b

# 设置环境变量（当前终端生效）
$env:LOCAL_VLM_BASE_URL="http://127.0.0.1:11434/v1"
$env:LOCAL_VLM_MODEL="qwen2.5vl:3b"

python app.py
```

### LM Studio 示例

```powershell
$env:LOCAL_VLM_BASE_URL="http://127.0.0.1:1234/v1"
$env:LOCAL_VLM_MODEL="你的已加载模型名"
python app.py
```

可选环境变量：
- `LOCAL_VLM_INTERVAL_SECONDS`：请求间隔秒数（默认 2）
- `LOCAL_VLM_API_KEY`：如果你的本地服务要求 token 再设置

## 5. 提醒策略

- 学习阶段下，若 `玩手机/离开` 连续达到阈值秒数才触发提醒
- 专注后连续计时清零
- 休息阶段不累计该阈值计时

## 6. 输出文件

- 会话日志：`logs/session_YYYYmmdd_HHMMSS.json`
- 日报图表：`reports/daily_YYYYmmdd.png`
- 日报汇总：`reports/daily_YYYYmmdd.json`
- 趋势图：`reports/trend_YYYYmmdd.png`
- 趋势汇总：`reports/trend_YYYYmmdd.json`

## 7. 隐私

- 默认仅本地运行与本地存储
- 默认不上传云端
- 默认不保存视频流，仅保存统计数据
