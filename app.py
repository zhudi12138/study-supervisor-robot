import datetime as dt
import json
import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import tkinter as tk
from tkinter import ttk, messagebox

os.environ.setdefault("MPLCONFIGDIR", str((Path(__file__).resolve().parent / ".mplcache")))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import winsound
except ImportError:
    winsound = None


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
            state, reason, color, box = self._infer_attention_state(frame)

            if box is not None:
                x, y, fw, fh = box
                cv2.rectangle(frame, (x, y), (x + fw, y + fh), color, 2)

            phase_cn = "学习" if self.current_phase_kind == "study" else "休息"
            cv2.putText(frame, f"Phase: {phase_cn}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (230, 230, 230), 2)
            cv2.putText(frame, f"State: {state}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame, f"Reason: {reason}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            ts = dt.datetime.now().strftime("%H:%M:%S")
            cv2.putText(frame, ts, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.imencode(".png", frame_rgb)[1].tobytes()
            photo = tk.PhotoImage(data=img)
            self.video_label.photo = photo
            self.video_label.configure(image=photo)

            self._tick_state(state)
            time.sleep(0.07)

    def _detect_phone_like_object(self, gray_frame, face_box):
        h, w = gray_frame.shape[:2]
        x, y, fw, fh = face_box
        roi_top = min(h - 1, y + int(fh * 0.45))
        roi_bottom = min(h, y + int(fh * 2.1))
        roi_left = max(0, x - int(fw * 0.7))
        roi_right = min(w, x + fw + int(fw * 0.7))
        if roi_bottom - roi_top < 30 or roi_right - roi_left < 30:
            return False

        roi = gray_frame[roi_top:roi_bottom, roi_left:roi_right]
        blur = cv2.GaussianBlur(roi, (5, 5), 0)
        edges = cv2.Canny(blur, 60, 130)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            x2, y2, w2, h2 = cv2.boundingRect(c)
            area = w2 * h2
            if area < 1200:
                continue
            ratio = w2 / max(h2, 1)
            if not (0.35 <= ratio <= 0.95):
                continue
            perimeter = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
            if len(approx) < 4:
                continue
            rect_area = cv2.contourArea(approx)
            fill_rate = rect_area / max(area, 1)
            if fill_rate < 0.5:
                continue
            return True
        return False

    def _infer_attention_state(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]

        frontal_faces = self.face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        profile_faces = self.profile_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        flipped = cv2.flip(gray, 1)
        right_profiles_raw = self.profile_classifier.detectMultiScale(flipped, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        right_profiles = []
        for (x, y, fw, fh) in right_profiles_raw:
            right_profiles.append((w - x - fw, y, fw, fh))

        all_faces = [("frontal", tuple(f)) for f in frontal_faces]
        all_faces.extend(("profile", tuple(f)) for f in profile_faces)
        all_faces.extend(("profile", tuple(f)) for f in right_profiles)

        if not all_faces:
            self.side_face_frames = max(0, self.side_face_frames - 1)
            self.head_down_frames = max(0, self.head_down_frames - 1)
            self.phone_suspect_frames = max(0, self.phone_suspect_frames - 1)
            return "离开座位", "未检测到人脸", (0, 0, 255), None

        face_type, box = max(all_faces, key=lambda item: item[1][2] * item[1][3])
        x, y, fw, fh = box
        cx = x + fw / 2
        cy = y + fh / 2

        roi_y2 = max(y + int(fh * 0.6), y + 1)
        roi = gray[y:roi_y2, x : x + fw]
        eyes = self.eye_classifier.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=6, minSize=(16, 16)) if roi.size else []

        head_down = (len(eyes) == 0 and cy > h * 0.55)
        off_center = abs(cx - w / 2) > w * 0.24 or abs(cy - h / 2) > h * 0.25
        side_face = face_type == "profile" or (fw / max(fh, 1) < 0.72)
        phone_like_object = self._detect_phone_like_object(gray, box)

        if side_face:
            self.side_face_frames += 1
        else:
            self.side_face_frames = max(0, self.side_face_frames - 1)

        if head_down:
            self.head_down_frames += 1
        else:
            self.head_down_frames = max(0, self.head_down_frames - 1)

        if head_down and phone_like_object:
            self.phone_suspect_frames += 1
        else:
            self.phone_suspect_frames = max(0, self.phone_suspect_frames - 1)

        if self.phone_suspect_frames >= 3:
            return "疑似玩手机", "低头+手机形态", (0, 165, 255), box
        if self.head_down_frames >= 10 and off_center:
            return "疑似玩手机", "持续低头偏离", (0, 165, 255), box
        if self.side_face_frames >= 8 and self.head_down_frames >= 6:
            return "疑似玩手机", "侧脸并持续低头", (0, 165, 255), box
        return "专注", "姿态稳定", (0, 180, 0), box

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
                self._maybe_alert()
                self.bad_streak_seconds = 0
        else:
            self.stats.absent_seconds += dt_seconds
            self.bad_streak_seconds += dt_seconds
            self.tip_var.set(
                f"提示：你已离开，连续 {self.bad_streak_seconds}s（阈值 {self.alert_threshold_seconds}s）。"
            )
            self._set_pet_message("汪？你去哪里啦，快回来学习。")
            if self.bad_streak_seconds >= self.alert_threshold_seconds:
                self._maybe_alert()
                self.bad_streak_seconds = 0

    def _maybe_alert(self):
        now = time.time()
        if now - self.alert_cooldown < 12:
            return
        self.alert_cooldown = now
        self.stats.alerts += 1
        self._set_pet_message("汪！提醒一下，先把手机放下吧。")
        if winsound:
            winsound.Beep(950, 250)

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
                self._maybe_alert()
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
            self._maybe_alert()
        else:
            self.status_var.set("状态：学习阶段")
            self.tip_var.set("提示：学习阶段开始，进入专注状态。")
            self._set_pet_message("汪！新一轮学习开始，专注起来。")
            self._maybe_alert()
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


