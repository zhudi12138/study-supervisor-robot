import datetime as dt
import json
import os
import platform
import random
import shutil
import subprocess
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

os.environ.setdefault("MPLCONFIGDIR", str((Path(__file__).resolve().parent / ".mplcache")))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


APP_NAME = "Study Guardian AI"
LOG_DIR = Path("logs")
REPORT_DIR = Path("reports")
LOG_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


@dataclass
class SessionStats:
    started_at: str
    ended_at: str | None = None
    mode: str = "single"
    total_seconds: int = 0
    study_seconds: int = 0
    break_seconds: int = 0
    focused_seconds: int = 0
    phone_seconds: int = 0
    distracted_seconds: int = 0
    absent_seconds: int = 0
    alerts: int = 0
    alert_threshold_seconds: int = 30


class StudyGuardianApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_NAME)
        self.root.geometry("1180x900")

        self.video_label = ttk.Label(root)
        self.video_label.pack(padx=12, pady=10)

        controls = ttk.Frame(root)
        controls.pack(fill="x", padx=12, pady=6)

        ttk.Label(controls, text="模式:").pack(side="left")
        self.mode_var = tk.StringVar(value="番茄钟")
        mode_box = ttk.Combobox(
            controls,
            textvariable=self.mode_var,
            values=["番茄钟", "单次学习"],
            width=10,
            state="readonly",
        )
        mode_box.pack(side="left", padx=6)

        ttk.Label(controls, text="学习(分钟):").pack(side="left")
        self.focus_min_var = tk.StringVar(value="50")
        ttk.Entry(controls, textvariable=self.focus_min_var, width=6).pack(side="left", padx=6)

        ttk.Label(controls, text="休息(分钟):").pack(side="left")
        self.break_min_var = tk.StringVar(value="10")
        ttk.Entry(controls, textvariable=self.break_min_var, width=6).pack(side="left", padx=6)

        ttk.Label(controls, text="循环次数:").pack(side="left")
        self.cycles_var = tk.StringVar(value="4")
        ttk.Entry(controls, textvariable=self.cycles_var, width=5).pack(side="left", padx=6)

        ttk.Label(controls, text="提醒阈值(秒):").pack(side="left")
        self.alert_threshold_var = tk.StringVar(value="30")
        ttk.Entry(controls, textvariable=self.alert_threshold_var, width=5).pack(side="left", padx=6)

        ttk.Label(controls, text="检测方式:").pack(side="left")
        self.detector_mode_var = tk.StringVar(value="YOLO(人+手机)")
        ttk.Combobox(
            controls,
            textvariable=self.detector_mode_var,
            values=["YOLO(人+手机)"],
            width=12,
            state="disabled",
        ).pack(side="left", padx=6)

        self.start_btn = ttk.Button(controls, text="开始监督", command=self.start_session)
        self.start_btn.pack(side="left", padx=6)
        self.stop_btn = ttk.Button(controls, text="结束监督", command=self.stop_session, state="disabled")
        self.stop_btn.pack(side="left", padx=6)
        self.report_btn = ttk.Button(controls, text="生成今日日报", command=self.generate_daily_report)
        self.report_btn.pack(side="left", padx=6)
        self.trend_btn = ttk.Button(controls, text="生成趋势图", command=self.generate_trend_report)
        self.trend_btn.pack(side="left", padx=6)

        self.status_var = tk.StringVar(value="状态：待机")
        ttk.Label(root, textvariable=self.status_var, font=("Microsoft YaHei UI", 12)).pack(anchor="w", padx=12)

        self.phase_var = tk.StringVar(value="阶段：--")
        ttk.Label(root, textvariable=self.phase_var, font=("Microsoft YaHei UI", 11)).pack(anchor="w", padx=12)

        self.timer_var = tk.StringVar(value="剩余：--:--")
        ttk.Label(root, textvariable=self.timer_var, font=("Consolas", 12)).pack(anchor="w", padx=12, pady=(0, 8))

        self.stats_text = tk.Text(root, height=10, wrap="word")
        self.stats_text.pack(fill="x", padx=12, pady=8)
        self.stats_text.insert("1.0", "统计：\n")
        self.stats_text.configure(state="disabled")

        self.tip_var = tk.StringVar(value="提示：保持目光在屏幕和笔记区，专注会被计入有效学习时长。")
        ttk.Label(root, textvariable=self.tip_var, foreground="#1f4e79").pack(anchor="w", padx=12, pady=(0, 8))

        self.report_preview = ttk.Label(root)
        self.report_preview.pack(padx=12, pady=(0, 10))

        self.cap = None
        self.running = False
        self.stop_event = threading.Event()
        self.worker = None

        self.session_end_ts = 0.0
        self.last_tick = 0.0
        self.alert_cooldown = 0.0
        self.alert_threshold_seconds = 30
        self.bad_streak_seconds = 0
        self.last_voice_ts = 0.0
        self.voice_cooldown_seconds = 10
        self.yolo_model_name = os.getenv("YOLO_MODEL", "yolov8n.pt")
        self.yolo_conf = float(os.getenv("YOLO_CONF", "0.35"))
        self.yolo_every_n_frames = int(os.getenv("YOLO_EVERY_N_FRAMES", "2"))
        self.yolo_model = None
        self.yolo_frame_count = 0
        self.yolo_last_result = ("专注", "YOLO待命", (0, 180, 0), None)

        self.plan: list[tuple[str, int]] = []
        self.plan_index = 0
        self.current_phase_kind = "study"

        self.stats = SessionStats(started_at="")
        self.face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.profile_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
        self.eye_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

        self.side_face_frames = 0
        self.head_down_frames = 0
        self.phone_suspect_frames = 0
        self.pet_window = None
        self.pet_canvas = None
        self.pet_message_var = tk.StringVar(value="汪~准备好学习了吗？")
        self.overlay_font = self._load_overlay_font(30)

    def start_session(self):
        if self.running:
            return
        try:
            focus_minutes = int(self.focus_min_var.get())
            break_minutes = int(self.break_min_var.get())
            cycles = int(self.cycles_var.get())
            alert_threshold = int(self.alert_threshold_var.get())
            if focus_minutes <= 0 or break_minutes <= 0 or cycles <= 0 or alert_threshold <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("输入错误", "学习/休息/循环/提醒阈值都需要正整数。")
            return

        mode = "pomodoro" if self.mode_var.get() == "番茄钟" else "single"
        if mode == "single":
            self.plan = [("study", focus_minutes * 60)]
        else:
            self.plan = []
            for i in range(cycles):
                self.plan.append(("study", focus_minutes * 60))
                if i != cycles - 1:
                    self.plan.append(("break", break_minutes * 60))

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            if platform.system() == "Darwin":
                messagebox.showerror(
                    "摄像头错误",
                    "无法打开摄像头。请到 系统设置 -> 隐私与安全性 -> 相机，允许 Python/终端 访问后重试。",
                )
            else:
                messagebox.showerror("摄像头错误", "无法打开摄像头，请检查权限或是否被其他程序占用。")
            return

        self.running = True
        self.stop_event.clear()
        self.plan_index = 0
        self.current_phase_kind = self.plan[0][0]
        self.session_end_ts = time.time() + self.plan[0][1]
        now_iso = dt.datetime.now().isoformat(timespec="seconds")
        self.stats = SessionStats(started_at=now_iso, mode=mode, alert_threshold_seconds=alert_threshold)
        self.last_tick = time.time()
        self.side_face_frames = 0
        self.head_down_frames = 0
        self.phone_suspect_frames = 0
        self.bad_streak_seconds = 0
        self.alert_threshold_seconds = alert_threshold
        self.yolo_frame_count = 0
        self.yolo_last_result = ("专注", "YOLO待命", (0, 180, 0), None)
        if self.yolo_model is None:
            try:
                self.yolo_model = YOLO(self.yolo_model_name)
            except Exception as e:
                messagebox.showerror("YOLO加载失败", f"无法加载模型 {self.yolo_model_name}:\n{e}")
                self.cap.release()
                self.cap = None
                return

        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status_var.set("状态：监督中")
        self._show_pet_assistant()
        self._set_pet_message("汪！我来监督你，别玩手机哦。")

        self.worker = threading.Thread(target=self._camera_loop, daemon=True)
        self.worker.start()
        self.root.after(200, self._ui_tick)

    def stop_session(self):
        if not self.running:
            return
        self.running = False
        self.stop_event.set()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")

        self.stats.ended_at = dt.datetime.now().isoformat(timespec="seconds")
        self._save_session()
        self._render_stats(final=True)
        self.status_var.set("状态：已结束")
        self.phase_var.set("阶段：--")
        self.timer_var.set("剩余：00:00")
        self._hide_pet_assistant()

    def _camera_loop(self):
        while not self.stop_event.is_set() and self.cap is not None:
            ok, frame = self.cap.read()
            if not ok:
                self.status_var.set("状态：读取摄像头失败")
                break

            frame = cv2.flip(frame, 1)
            h, _ = frame.shape[:2]
            state, reason, color, box = self._infer_attention_state_yolo(frame)

            if box is not None:
                x, y, fw, fh = box
                cv2.rectangle(frame, (x, y), (x + fw, y + fh), color, 2)

            phase_cn = "学习" if self.current_phase_kind == "study" else "休息"
            ts = dt.datetime.now().strftime("%H:%M:%S")
            frame = self._draw_overlay_text(
                frame,
                [
                    (f"阶段: {phase_cn}", (20, 20), (235, 235, 235)),
                    (f"状态: {state}", (20, 62), color),
                    (f"依据: {reason}", (20, 102), color),
                    (f"时间: {ts}", (20, h - 50), (220, 220, 220)),
                ],
            )

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.imencode(".png", frame_rgb)[1].tobytes()
            photo = tk.PhotoImage(data=img)
            self.video_label.photo = photo
            self.video_label.configure(image=photo)

            self._tick_state(state)
            time.sleep(0.07)

    def _load_overlay_font(self, size: int):
        font_candidates = [
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/msyhbd.ttc",
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        ]
        for path in font_candidates:
            if Path(path).exists():
                try:
                    return ImageFont.truetype(path, size=size)
                except Exception:
                    continue
        return ImageFont.load_default()

    def _draw_overlay_text(self, frame, lines):
        # OpenCV uses BGR; Pillow uses RGB
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        for text, (x, y), color_bgr in lines:
            color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
            draw.text((x, y), text, font=self.overlay_font, fill=color_rgb)
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def _infer_attention_state_yolo(self, frame):
        if self.yolo_model is None:
            return "专注", "YOLO未加载", (0, 180, 0), None

        self.yolo_frame_count += 1
        if self.yolo_frame_count % max(1, self.yolo_every_n_frames) != 0:
            return self.yolo_last_result

        try:
            result = self.yolo_model.predict(
                source=frame,
                conf=self.yolo_conf,
                iou=0.5,
                max_det=20,
                imgsz=640,
                verbose=False,
            )[0]
        except Exception:
            return "专注", "YOLO推理异常", (0, 180, 0), None

        persons = []
        phones = []
        if result.boxes is not None:
            for b in result.boxes:
                cls_id = int(b.cls[0].item())
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                if cls_id == 0:
                    persons.append((x1, y1, x2, y2))
                elif cls_id == 67:
                    phones.append((x1, y1, x2, y2))

        if not persons:
            self.yolo_last_result = ("离开座位", "YOLO未检测到人", (0, 0, 255), None)
            return self.yolo_last_result

        main_person = max(persons, key=lambda p: (p[2] - p[0]) * (p[3] - p[1]))
        person_box = (main_person[0], main_person[1], main_person[2] - main_person[0], main_person[3] - main_person[1])

        if not phones:
            self.yolo_last_result = ("专注", "YOLO检测到人，未检测到手机", (0, 180, 0), person_box)
            return self.yolo_last_result

        phone_use = any(self._phone_belongs_to_person(main_person, ph) for ph in phones)
        if phone_use:
            self.yolo_last_result = ("疑似玩手机", "YOLO检测到人+手机", (0, 165, 255), person_box)
        else:
            self.yolo_last_result = ("专注", "YOLO手机不在本人附近", (0, 180, 0), person_box)
        return self.yolo_last_result

    def _phone_belongs_to_person(self, person_xyxy, phone_xyxy):
        px1, py1, px2, py2 = person_xyxy
        fx1, fy1, fx2, fy2 = phone_xyxy
        pw = max(1, px2 - px1)
        ph = max(1, py2 - py1)

        ex1 = px1 - int(pw * 0.25)
        ex2 = px2 + int(pw * 0.25)
        ey1 = py1 + int(ph * 0.15)
        ey2 = py2 + int(ph * 0.20)

        cx = (fx1 + fx2) / 2
        cy = (fy1 + fy2) / 2
        if not (ex1 <= cx <= ex2 and ey1 <= cy <= ey2):
            return False

        f_area = max(1, (fx2 - fx1) * (fy2 - fy1))
        inter_x1 = max(ex1, fx1)
        inter_y1 = max(ey1, fy1)
        inter_x2 = min(ex2, fx2)
        inter_y2 = min(ey2, fy2)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return False
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        return inter_area / f_area > 0.2

    def _tick_state(self, state: str):
        now = time.time()
        dt_seconds = max(0, int(now - self.last_tick))
        if dt_seconds == 0:
            return
        self.last_tick = now

        self.stats.total_seconds += dt_seconds

        if self.current_phase_kind == "break":
            self.stats.break_seconds += dt_seconds
            self.bad_streak_seconds = 0
            self.tip_var.set("提示：休息阶段，活动一下肩颈。")
            self._set_pet_message("汪~休息时间到，伸个懒腰再回来。")
            return

        self.stats.study_seconds += dt_seconds

        if state == "专注":
            self.stats.focused_seconds += dt_seconds
            self.bad_streak_seconds = 0
            self.tip_var.set("提示：状态良好，继续保持。")
            self._set_pet_message("汪！状态很好，继续冲。")
        elif state == "疑似玩手机":
            self.stats.phone_seconds += dt_seconds
            self.stats.distracted_seconds += dt_seconds
            self.bad_streak_seconds += dt_seconds
            self.tip_var.set(
                f"提示：检测到疑似玩手机，连续 {self.bad_streak_seconds}s（阈值 {self.alert_threshold_seconds}s）。"
            )
            self._set_pet_message(f"汪汪！检测到你可能在看手机 ({self.bad_streak_seconds}s)")
            if self.bad_streak_seconds >= self.alert_threshold_seconds:
                self._maybe_alert(reason="phone")
                self.bad_streak_seconds = 0
        else:
            self.stats.absent_seconds += dt_seconds
            self.bad_streak_seconds = 0
            self.tip_var.set("提示：你已离开，离开时长会计入统计，不触发提醒。")
            self._set_pet_message("我会帮你记录离开时长，回来后继续加油。")

    def _maybe_alert(self, reason: str = "phone"):
        now = time.time()
        if now - self.alert_cooldown < 12:
            return
        self.alert_cooldown = now
        self.stats.alerts += 1
        if reason == "phone":
            phrases = [
                "先把手机放一边，我们回到当前任务。",
                "注意力有点飘了，先回到学习节奏。",
                "休息刷手机可以，学习阶段先专注一下。",
                "我们再坚持几分钟，手机稍后再看。",
            ]
            text = random.choice(phrases)
            self._set_pet_message(text)
            self._speak_alert(text)

    def _speak_alert(self, text: str):
        now = time.time()
        if now - self.last_voice_ts < self.voice_cooldown_seconds:
            return
        self.last_voice_ts = now
        threading.Thread(target=self._speak_worker, args=(text,), daemon=True).start()

    def _speak_worker(self, text: str):
        try:
            system = platform.system()
            if system == "Darwin":
                subprocess.run(["say", text], check=False, timeout=6)
                return
            if system == "Windows":
                escaped = text.replace("'", "''")
                ps = (
                    "Add-Type -AssemblyName System.Speech; "
                    "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                    "$s.Rate=0; "
                    f"$s.Speak('{escaped}')"
                )
                subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=False, timeout=8)
                return
            if shutil.which("spd-say"):
                subprocess.run(["spd-say", text], check=False, timeout=6)
                return
            if shutil.which("espeak"):
                subprocess.run(["espeak", text], check=False, timeout=6)
                return
        except Exception:
            return

    def _show_pet_assistant(self):
        if self.pet_window is not None and self.pet_window.winfo_exists():
            return

        self.pet_window = tk.Toplevel(self.root)
        self.pet_window.overrideredirect(True)
        self.pet_window.attributes("-topmost", True)
        width, height = 250, 260
        x = max(20, self.root.winfo_screenwidth() - width - 36)
        y = max(20, self.root.winfo_screenheight() - height - 88)
        self.pet_window.geometry(f"{width}x{height}+{x}+{y}")
        self.pet_window.configure(bg="#fff6dd", bd=2, relief="solid")

        self.pet_canvas = tk.Canvas(self.pet_window, width=250, height=190, bg="#fff6dd", highlightthickness=0)
        self.pet_canvas.pack(fill="x", padx=4, pady=(6, 0))
        self._draw_dog_avatar()

        msg = tk.Label(
            self.pet_window,
            textvariable=self.pet_message_var,
            bg="#fff6dd",
            fg="#4a3b2d",
            font=("Microsoft YaHei UI", 10),
            wraplength=220,
            justify="left",
        )
        msg.pack(fill="both", expand=True, padx=10, pady=(4, 8))

    def _hide_pet_assistant(self):
        if self.pet_window is not None and self.pet_window.winfo_exists():
            self.pet_window.destroy()
        self.pet_window = None
        self.pet_canvas = None

    def _set_pet_message(self, text: str):
        def _apply():
            self.pet_message_var.set(text)

        self.root.after(0, _apply)

    def _draw_dog_avatar(self):
        if self.pet_canvas is None:
            return
        c = self.pet_canvas
        c.delete("all")
        c.create_oval(26, 12, 224, 182, fill="#f2d3a0", outline="#7a5a36", width=3)
        c.create_polygon(55, 58, 20, 28, 80, 46, fill="#c99053", outline="#7a5a36", width=2)
        c.create_polygon(195, 58, 230, 28, 170, 46, fill="#c99053", outline="#7a5a36", width=2)
        c.create_oval(72, 84, 98, 110, fill="black")
        c.create_oval(152, 84, 178, 110, fill="black")
        c.create_oval(114, 100, 136, 116, fill="#5a3b1f", outline="#5a3b1f")
        c.create_arc(90, 116, 125, 146, start=210, extent=110, style="arc", outline="#5a3b1f", width=3)
        c.create_arc(125, 116, 160, 146, start=220, extent=110, style="arc", outline="#5a3b1f", width=3)
        c.create_oval(96, 126, 154, 172, fill="#f8e7cc", outline="#cda67a", width=2)
        c.create_text(125, 176, text="Study Dog", fill="#7a5a36", font=("Consolas", 10, "bold"))

    def _ui_tick(self):
        if not self.running:
            return

        remain = max(0, int(self.session_end_ts - time.time()))
        mm, ss = divmod(remain, 60)
        phase_cn = "学习" if self.current_phase_kind == "study" else "休息"
        phase_no = self.plan_index + 1
        self.phase_var.set(f"阶段：{phase_cn} (第 {phase_no}/{len(self.plan)} 段)")
        self.timer_var.set(f"剩余：{mm:02d}:{ss:02d}")
        self._render_stats()

        if remain <= 0:
            if not self._next_phase():
                self.status_var.set("状态：计划完成")
                self._speak_alert("学习计划完成，做得很好。")
                self.stop_session()
                messagebox.showinfo("学习完成", "本次计划已完成，日志和统计图可在 reports 中查看。")
                return

        self.root.after(300, self._ui_tick)

    def _next_phase(self):
        self.plan_index += 1
        if self.plan_index >= len(self.plan):
            return False

        self.current_phase_kind = self.plan[self.plan_index][0]
        phase_seconds = self.plan[self.plan_index][1]
        self.session_end_ts = time.time() + phase_seconds
        self.bad_streak_seconds = 0

        if self.current_phase_kind == "break":
            self.status_var.set("状态：休息阶段")
            self.tip_var.set("提示：休息中，喝水或起身活动一下。")
            self._set_pet_message("汪~先休息，等会继续学习。")
            self._speak_alert("辛苦了，先休息一下。")
        else:
            self.status_var.set("状态：学习阶段")
            self.tip_var.set("提示：学习阶段开始，进入专注状态。")
            self._set_pet_message("汪！新一轮学习开始，专注起来。")
            self._speak_alert("新的学习阶段开始了，我们继续。")
        return True

    def _render_stats(self, final: bool = False):
        study_total = max(1, self.stats.study_seconds)
        focus_ratio = self.stats.focused_seconds / study_total

        text = (
            "统计：\n"
            f"总监测时长: {self.stats.total_seconds} 秒\n"
            f"学习时长: {self.stats.study_seconds} 秒\n"
            f"休息时长: {self.stats.break_seconds} 秒\n"
            f"专注时长: {self.stats.focused_seconds} 秒\n"
            f"手机使用时长: {self.stats.phone_seconds} 秒\n"
            f"离开时长: {self.stats.absent_seconds} 秒\n"
            f"提醒次数: {self.stats.alerts} 次\n"
            f"提醒阈值: {self.stats.alert_threshold_seconds} 秒\n"
            f"学习阶段专注率: {focus_ratio:.1%}\n"
        )
        if final:
            text += "会话已保存到 logs 目录。\n"

        self.stats_text.configure(state="normal")
        self.stats_text.delete("1.0", "end")
        self.stats_text.insert("1.0", text)
        self.stats_text.configure(state="disabled")

    def _save_session(self):
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = LOG_DIR / f"session_{stamp}.json"
        path.write_text(json.dumps(asdict(self.stats), ensure_ascii=False, indent=2), encoding="utf-8")

    def _aggregate_logs_by_day(self):
        per_day = defaultdict(
            lambda: {
                "study_seconds": 0,
                "focused_seconds": 0,
                "phone_seconds": 0,
                "distracted_seconds": 0,
                "absent_seconds": 0,
                "break_seconds": 0,
                "alerts": 0,
                "sessions": 0,
            }
        )

        for path in sorted(LOG_DIR.glob("session_*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                started = str(data.get("started_at", ""))
                if len(started) < 10:
                    continue
                day = started[:10]
                per_day[day]["study_seconds"] += int(data.get("study_seconds", 0))
                per_day[day]["focused_seconds"] += int(data.get("focused_seconds", 0))
                per_day[day]["phone_seconds"] += int(data.get("phone_seconds", data.get("distracted_seconds", 0)))
                per_day[day]["distracted_seconds"] += int(data.get("distracted_seconds", 0))
                per_day[day]["absent_seconds"] += int(data.get("absent_seconds", 0))
                per_day[day]["break_seconds"] += int(data.get("break_seconds", 0))
                per_day[day]["alerts"] += int(data.get("alerts", 0))
                per_day[day]["sessions"] += 1
            except Exception:
                continue
        return per_day

    def generate_daily_report(self):
        today = dt.date.today().isoformat()
        all_days = self._aggregate_logs_by_day()
        totals = all_days.get(today, {})

        if not totals or totals["sessions"] == 0:
            messagebox.showinfo("无数据", "今天还没有可用学习日志。")
            return

        labels = ["专注", "玩手机", "离开", "休息"]
        values = [
            totals["focused_seconds"] / 60,
            totals["phone_seconds"] / 60,
            totals["absent_seconds"] / 60,
            totals["break_seconds"] / 60,
        ]

        fig, ax = plt.subplots(figsize=(8, 4.6))
        colors = ["#3b8f50", "#f0a500", "#d9534f", "#4d7ea8"]
        bars = ax.bar(labels, values, color=colors)
        ax.set_title(f"{today} 学习监督日报")
        ax.set_ylabel("分钟")

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, f"{val:.1f}", ha="center", va="bottom")

        focus_ratio = totals["focused_seconds"] / max(1, totals["study_seconds"])
        subtitle = (
            f"场次: {totals['sessions']} | 学习: {totals['study_seconds'] // 60} 分钟 | "
            f"专注率: {focus_ratio:.1%} | 提醒: {totals['alerts']} 次"
        )
        ax.text(0.0, -0.18, subtitle, transform=ax.transAxes, fontsize=10)

        fig.tight_layout()
        day_stamp = dt.datetime.now().strftime("%Y%m%d")
        img_path = REPORT_DIR / f"daily_{day_stamp}.png"
        json_path = REPORT_DIR / f"daily_{day_stamp}.json"
        fig.savefig(img_path, dpi=150)
        plt.close(fig)

        json_path.write_text(json.dumps(totals, ensure_ascii=False, indent=2), encoding="utf-8")

        photo = tk.PhotoImage(file=str(img_path))
        self.report_preview.photo = photo
        self.report_preview.configure(image=photo)

        messagebox.showinfo("日报已生成", f"图表: {img_path}\n汇总: {json_path}")

    def generate_trend_report(self):
        per_day = self._aggregate_logs_by_day()
        if not per_day:
            messagebox.showinfo("无数据", "还没有可用学习日志，无法生成趋势图。")
            return

        today = dt.date.today()
        day_list = [today - dt.timedelta(days=i) for i in range(13, -1, -1)]
        day_keys = [d.isoformat() for d in day_list]
        day_labels = [d.strftime("%m-%d") for d in day_list]

        daily_study_min = [per_day.get(k, {}).get("study_seconds", 0) / 60 for k in day_keys]
        daily_focus_min = [per_day.get(k, {}).get("focused_seconds", 0) / 60 for k in day_keys]
        daily_focus_ratio = []
        for k in day_keys:
            study = per_day.get(k, {}).get("study_seconds", 0)
            focus = per_day.get(k, {}).get("focused_seconds", 0)
            daily_focus_ratio.append((focus / study) * 100 if study > 0 else 0)

        week_buckets = defaultdict(lambda: {"study_seconds": 0, "focused_seconds": 0, "alerts": 0})
        for day_key, rec in per_day.items():
            d = dt.date.fromisoformat(day_key)
            iso = d.isocalendar()
            key = f"{iso.year}-W{iso.week:02d}"
            week_buckets[key]["study_seconds"] += rec["study_seconds"]
            week_buckets[key]["focused_seconds"] += rec["focused_seconds"]
            week_buckets[key]["alerts"] += rec["alerts"]

        week_keys = sorted(week_buckets.keys())[-8:]
        week_study_hours = [week_buckets[k]["study_seconds"] / 3600 for k in week_keys]
        week_focus_ratio = []
        for k in week_keys:
            study = week_buckets[k]["study_seconds"]
            focus = week_buckets[k]["focused_seconds"]
            week_focus_ratio.append((focus / study) * 100 if study > 0 else 0)

        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        axes[0].plot(day_labels, daily_study_min, marker="o", label="学习分钟", color="#2b7a78")
        axes[0].plot(day_labels, daily_focus_min, marker="o", label="专注分钟", color="#3aafa9")
        axes[0].set_title("近14天学习趋势")
        axes[0].set_ylabel("分钟")
        axes[0].tick_params(axis="x", rotation=35)
        axes[0].grid(alpha=0.25)
        axes[0].legend()

        axes[1].bar(week_keys, week_study_hours, color="#f5a623", label="学习小时")
        axes[1].set_title("近8周学习趋势")
        axes[1].set_ylabel("小时")
        axes[1].tick_params(axis="x", rotation=20)
        axes[1].grid(alpha=0.2)

        ax2b = axes[1].twinx()
        ax2b.plot(week_keys, week_focus_ratio, color="#d64541", marker="o", label="专注率%")
        ax2b.set_ylabel("专注率(%)")

        fig.tight_layout()
        stamp = dt.datetime.now().strftime("%Y%m%d")
        img_path = REPORT_DIR / f"trend_{stamp}.png"
        json_path = REPORT_DIR / f"trend_{stamp}.json"
        fig.savefig(img_path, dpi=150)
        plt.close(fig)

        trend_payload = {
            "generated_on": dt.datetime.now().isoformat(timespec="seconds"),
            "daily_14": [
                {
                    "date": k,
                    "study_minutes": round(daily_study_min[i], 2),
                    "focused_minutes": round(daily_focus_min[i], 2),
                    "focus_ratio_percent": round(daily_focus_ratio[i], 2),
                }
                for i, k in enumerate(day_keys)
            ],
            "weekly_8": [
                {
                    "week": k,
                    "study_hours": round(week_study_hours[i], 2),
                    "focus_ratio_percent": round(week_focus_ratio[i], 2),
                }
                for i, k in enumerate(week_keys)
            ],
        }
        json_path.write_text(json.dumps(trend_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        photo = tk.PhotoImage(file=str(img_path))
        self.report_preview.photo = photo
        self.report_preview.configure(image=photo)
        messagebox.showinfo("趋势图已生成", f"图表: {img_path}\n汇总: {json_path}")


def main():
    root = tk.Tk()
    style = ttk.Style()
    if "vista" in style.theme_names():
        style.theme_use("vista")
    app = StudyGuardianApp(root)

    def on_close():
        if app.running:
            app.stop_session()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()


