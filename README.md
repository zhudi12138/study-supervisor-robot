# Study Guardian AI（学习监督桌面 AI）

当前版本核心能力：
- 番茄钟模式（学习/休息循环）
- YOLO 识别人和手机（玩手机监督）
- 悬浮小狗助手（状态提示）
- 语音提醒（更自然口吻）
- 每日/趋势统计图

## 1. 安装依赖

```powershell
python -m pip install -r requirements.txt
```

## 2. 启动

```powershell
python app.py
```

## 3. 使用

1. 设置学习时长参数
2. 点击“开始监督”
3. 摄像头页会显示：阶段/状态/依据/时间（中文）
4. 点击“结束监督”结束并保存日志

## 4. YOLO 判定逻辑

- `离开座位`：YOLO 未检测到 person
- `疑似玩手机`：YOLO 检测到 person + cell phone，且手机在监督对象附近
- `专注`：检测到人但未满足玩手机条件

## 5. 提醒策略

- 仅对“疑似玩手机”触发语音提醒
- 人物离开画面只计入离开时长，不提醒

## 6. macOS 说明

- 若摄像头打不开：系统设置 -> 隐私与安全性 -> 相机，允许 Python/终端
- 语音提醒使用系统 `say`

## 7. 输出文件

- 会话日志：`logs/session_YYYYmmdd_HHMMSS.json`
- 日报图表：`reports/daily_YYYYmmdd.png`
- 趋势图：`reports/trend_YYYYmmdd.png`

## 8. 依赖说明

YOLO 使用 `ultralytics`，默认模型为 `yolov8n.pt`，可通过环境变量覆盖：

- `YOLO_MODEL`（默认 `yolov8n.pt`）
- `YOLO_CONF`（默认 `0.35`）
- `YOLO_EVERY_N_FRAMES`（默认 `2`）
