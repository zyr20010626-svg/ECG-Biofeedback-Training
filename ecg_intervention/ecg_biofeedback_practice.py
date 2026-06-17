#!/usr/bin/env python3
"""
ECG Biofeedback Training — 练习版 (Practice Demo)
内感受视觉训练干预方案 - 被试展示用

基于 PsychoPy 原生 API（visual, event, gui），可直接在 PsychoPy Coder 中 F5 运行。

与正式版差异：
- TRIALS_PER_PHASE = 3（正式版 24）
- REST_DURATION_S = 10 秒（正式版 300 秒 = 5 分钟）
- 数据保存至 recordings_practice/
- 说明文字标明「练习模式」
- 其余参数（组块数、GUI、架构）与正式版完全一致

其余架构、功能、GUI 布局与正式版完全一致。
"""

import sys
import os
import random
import time
import csv
import json
import math
import threading
import queue
from datetime import datetime
from collections import deque

import numpy as np

# ──────────────────────────────────────────────────────────────
# scipy import (optional — fallback to raw signal if missing)
# ──────────────────────────────────────────────────────────────
try:
    from scipy import signal as scipy_signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ──────────────────────────────────────────────────────────────
# pyserial import (optional)
# ──────────────────────────────────────────────────────────────
try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

# ──────────────────────────────────────────────────────────────
# PsychoPy imports
# ──────────────────────────────────────────────────────────────
from psychopy import visual, event, core, gui
from psychopy.clock import Clock


# ══════════════════════════════════════════════════════════════
#  Constants — 练习版（精简参数，其余与正式版一致）
# ══════════════════════════════════════════════════════════════
FS = 512                          # BMD101 sampling rate (Hz)
R_PEAK_REFRACTORY_MS = 200
RESP_WIN_START = 0                # ms after target R-wave (实验说明: 0–500ms)
RESP_WIN_END = 500                # ms after target R-wave
TRIALS_PER_PHASE = 3              # 每阶段 3 试次（练习版）
BLOCKS_PER_SESSION = 3            # 3 个组块（与正式版一致）
REST_DURATION_S = 10              # 10 秒休息（练习版，正式版 300）
TIMEOUT_AFTER_TARGET_S = 3.0      # wait for keypress after response window
HEART_APPEAR_DELAY_MS = 0         # R波后立即呈现
N_MIN = 5
N_MAX = 10

# Heart animation keyframes: (elapsed_ms, scale)
HEART_KEYFRAMES = [
    (0, 1.0), (60, 1.35), (140, 0.9), (260, 1.1), (400, 1.0)
]

# Red pulse period (ms)
RED_PULSE_PERIOD = 800


# ══════════════════════════════════════════════════════════════
#  R-Peak Detector
# ══════════════════════════════════════════════════════════════
class RPeakDetector:
    """Real-time R-peak detection using bandpass filter + adaptive threshold."""

    def __init__(self, fs=FS):
        self.fs = fs
        self.has_filter = False

        if HAS_SCIPY:
            nyq = fs / 2.0
            self.b, self.a = scipy_signal.butter(
                4, [0.5 / nyq, 30 / nyq], btype='band'
            )
            self.zi = scipy_signal.lfilter_zi(self.b, self.a) * 0
            self.has_filter = True

        self.refractory_samples = int(R_PEAK_REFRACTORY_MS * fs / 1000)
        self.since_last_peak = self.refractory_samples
        self.signal_history = deque(maxlen=fs)
        self.threshold = 0.0
        self.filtered_out = 0.0

    def process(self, raw_sample):
        """Process one raw ECG sample. Returns (filtered_value, is_r_peak)."""
        if self.has_filter:
            filtered, self.zi = scipy_signal.lfilter(
                self.b, self.a, [raw_sample], zi=self.zi
            )
            self.filtered_out = filtered[0]
        else:
            self.filtered_out = raw_sample

        self.signal_history.append(self.filtered_out)
        self.since_last_peak += 1

        if len(self.signal_history) > int(0.1 * self.fs):
            recent = np.array(list(self.signal_history))
            self.threshold = float(np.mean(recent) + 1.5 * np.std(recent))

        is_peak = False
        if (self.filtered_out > self.threshold
                and self.since_last_peak > self.refractory_samples
                and len(self.signal_history) >= 5):
            buf = list(self.signal_history)
            if self.filtered_out >= max(buf[-5:]):
                is_peak = True
                self.since_last_peak = 0

        return self.filtered_out, is_peak

    def reset(self):
        if self.has_filter:
            self.zi = scipy_signal.lfilter_zi(self.b, self.a) * 0
        self.since_last_peak = self.refractory_samples
        self.signal_history.clear()
        self.threshold = 0.0


# ══════════════════════════════════════════════════════════════
#  Heart shape vertices (parametric equation)
# ══════════════════════════════════════════════════════════════
def _make_heart_vertices(scale=1.0):
    """Return list of [x, y] vertices for a heart shape (normalized ~ -1..1)."""
    verts = []
    steps = 80
    for i in range(steps):
        t = 2 * math.pi * i / steps
        x = 16 * math.sin(t) ** 3
        y = (13 * math.cos(t) - 5 * math.cos(2 * t)
             - 2 * math.cos(3 * t) - math.cos(4 * t))
        # Normalize to roughly [-1, 1] range
        verts.append([x / 16 * scale, y / 16 * scale])
    return verts


# ══════════════════════════════════════════════════════════════
#  BMD101 Worker Thread
# ══════════════════════════════════════════════════════════════
class BMD101WorkerThread(threading.Thread):
    """Serial reader thread — puts events into a queue.

    Queue items: (event_type, payload)
        'r_peak'    -> ('r_peak', None)
        'raw_data'  -> ('raw_data', int_value)
        'heart_rate'-> ('heart_rate', float_bpm)
        'error'     -> ('error', message_string)
        'connected' -> ('connected', True/False)
    """

    def __init__(self, port, baud=57600):
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self.queue = queue.Queue()
        self._running = False
        self._serial = None
        self.detector = RPeakDetector(FS)
        self._last_r_peak_time = 0.0
        self._rr_intervals = deque(maxlen=10)

    def run(self):
        if not HAS_SERIAL:
            self.queue.put(('error', 'pyserial not installed.'))
            self.queue.put(('connected', False))
            return

        try:
            self._serial = serial.Serial(
                self.port, baudrate=self.baud, timeout=0.5
            )
            self.queue.put(('connected', True))
        except Exception as e:
            self.queue.put(('error', f"无法打开串口 {self.port}: {e}"))
            self.queue.put(('connected', False))
            return

        self._running = True
        while self._running:
            try:
                self._parse_packet()
            except serial.SerialException as e:
                self.queue.put(('error', f"串口错误: {e}"))
                break
            except Exception:
                pass

        if self._serial and self._serial.is_open:
            self._serial.close()
        self.queue.put(('connected', False))

    def stop(self):
        self._running = False

    def _parse_packet(self):
        if not self._serial or not self._serial.is_open:
            return

        if self._serial.read(1) != b'\xaa':
            return
        if self._serial.read(1) != b'\xaa':
            return

        pkt_len_byte = self._serial.read(1)
        if not pkt_len_byte:
            return
        pkt_len = pkt_len_byte[0]

        payload = self._serial.read(pkt_len + 1)
        if len(payload) < pkt_len + 1:
            return

        received_checksum = payload[-1]
        data_bytes = payload[:-1]
        # ThinkGear checksum: one's complement of sum
        calc_checksum = (~sum(data_bytes)) & 0xFF

        if calc_checksum != received_checksum:
            return

        # Iterate through all data rows in the packet
        i = 0
        while i < len(data_bytes):
            code = data_bytes[i]
            if i + 1 >= len(data_bytes):
                break
            row_len = data_bytes[i + 1]
            if i + 2 + row_len > len(data_bytes):
                break

            if code == 0x80 and row_len == 2:
                # RAW_WAVE: 16-bit signed value
                raw_val = (data_bytes[i + 2] << 8) | data_bytes[i + 3]
                if raw_val >= 32768:
                    raw_val -= 65536
                self.queue.put(('raw_data', raw_val))

                _, is_peak = self.detector.process(raw_val)
                if is_peak:
                    now = time.perf_counter()
                    _valid = True
                    if self._last_r_peak_time > 0:
                        rr = now - self._last_r_peak_time
                        # Reject false-positive peaks (T-wave, noise) whose RR interval
                        # is physiologically implausible relative to the running median.
                        if len(self._rr_intervals) >= 3:
                            median_rr = np.median(list(self._rr_intervals))
                            if rr < 0.4 * median_rr:
                                _valid = False
                        if _valid:
                            self._rr_intervals.append(rr)
                    if _valid:
                        self._last_r_peak_time = now
                        self.queue.put(('r_peak', None))

            elif code == 0x03:
                # HEART_RATE: BMD101 chip's internal DSP-computed HR (BPM)
                # CardioChip uses the same value via ThinkGear.dll → accurate!
                if row_len >= 1:
                    hr_val = data_bytes[i + 2]
                    if 20 <= hr_val <= 250:  # physiological range
                        self.queue.put(('heart_rate', float(hr_val)))

            i += 2 + row_len

    def reset_detector(self):
        self.detector.reset()
        self._last_r_peak_time = 0.0
        self._rr_intervals.clear()


# ══════════════════════════════════════════════════════════════
#  Simulated Worker Thread  (for testing without hardware)
# ══════════════════════════════════════════════════════════════
class SimulatedWorkerThread(threading.Thread):
    """Generates synthetic R-peaks for testing."""

    def __init__(self):
        super().__init__(daemon=True)
        self.queue = queue.Queue()
        self._running = False
        self._current_hr = 70.0  # start at 70 BPM, drift gradually

    def run(self):
        self._running = True
        self.queue.put(('connected', True))

        while self._running:
            # Gradually drift HR, not random jumps
            self._current_hr += random.uniform(-2.0, 2.0)
            self._current_hr = max(60, min(80, self._current_hr))
            hr = self._current_hr
            interval = 60.0 / hr * (1 + random.uniform(-0.03, 0.03))

            t0 = time.perf_counter()
            while time.perf_counter() - t0 < interval:
                time.sleep(0.05)
                if not self._running:
                    self.queue.put(('connected', False))
                    return
                if random.random() < 0.1:
                    self.queue.put(('raw_data', random.randint(-300, 300)))

            self.queue.put(('r_peak', None))
            self.queue.put(('heart_rate', round(60.0 / interval, 1)))

        self.queue.put(('connected', False))

    def stop(self):
        self._running = False

    def reset_detector(self):
        pass


# ══════════════════════════════════════════════════════════════
#  Experiment State Machine
# ══════════════════════════════════════════════════════════════
class ExperimentController:
    """State machine for the experiment.  No GUI toolkit dependencies.

    The main PsychoPy loop calls process_worker_queue(), update(dt),
    and handle_keys() each frame, then reads state variables to draw.
    """

    def __init__(self):
        self._worker = None
        self._simulate = False

        # ── state machine ──
        self.state = 'idle'          # idle | starting | wait_key | block_active
                                     # | phase_active | trial_active | trial_feedback
                                     # | phase_summary | rest | complete
        self._wait_callback = None   # callable — invoked on any key in wait_key

        # ── trial / block tracking ──
        self.current_block = 0
        self.current_phase = ''      # 'feedback' | 'nofeedback'
        self.current_trial = 0
        self.current_target_n = 0
        self.trial_r_wave_count = 0
        self.trial_start_time = 0.0
        self.target_r_peak_time = 0.0
        self.pressed = False
        self.phase_correct = 0
        self.phase_total = 0

        self.trials = []
        self.trial_idx = 0

        # ── timing (set / read by main loop) ──
        self.feedback_start_time = 0.0   # when trial_feedback began
        self.rest_remaining = 0
        self.rest_start_time = 0.0
        self.timeout_after_target = 0.0  # absolute time after which trial times out
        self.max_trial_deadline = 0.0    # absolute trial timeout (fallback)
        self.red_signal_trigger_time = 0.0  # when to activate red signal

        # ── display state (read by draw loop) ──
        self.instruction_text = ""
        self.phase_info_text = ""
        self.trial_progress_text = ""
        self.feedback_type = ''       # 'correct' | 'incorrect' | 'timeout' | ''
        self.feedback_text = ""
        self.accuracy = 0.0
        self.heart_rate = 0.0
        self._hr_from_chip = False   # True when 0x03 chip HR has been received
        self.heart_beat_time = 0.0   # last beat time (for animation)
        self.heartbeat_count_text = ""
        self.red_signal_active = False
        self.red_signal_pulse_start = 0.0  # when pulse animation started
        self.wait_key_text = ""       # text shown during wait_key state
        self.show_rest = False
        self.rest_display_text = ""

        # ── participant data ──
        self.participant_id = ""
        self.session = "1"
        self.session_data_path = ""

        # ── data recording ──
        self._r_peak_times = []
        self._trial_results = []
        self._ecg_file = None
        self._ecg_writer = None
        self._hr_log_file = None       # ecg_log.csv (timestamp, raw, hr, hr_4s, hr_30s)
        self._hr_log_writer = None
        self._hr_timestamps = deque()  # (timestamp, hr_bpm) for running avg computation

        # ── results page guard ──
        self._results_start_time = 0.0  # time.perf_counter when results page appeared
                                        # used to enforce minimum display time (prevent flash)

        # ── init timer references ──
        self._exp_start_time = 0.0   # set when experiment starts (time.perf_counter)
        self._global_clock = Clock()  # PsychoPy clock for dt in main loop

    # ── public setup ──

    def set_participant_info(self, pid, session="1"):
        self.participant_id = pid
        self.session = session

    # ── trial generation ──

    def _generate_trials(self):
        random.seed(int(time.time()))
        self.trials = []
        for block in range(1, BLOCKS_PER_SESSION + 1):
            for phase in ['feedback', 'nofeedback']:
                for t in range(TRIALS_PER_PHASE):
                    n = random.randint(N_MIN, N_MAX)
                    self.trials.append({
                        'block': block,
                        'phase': phase,
                        'trial': t + 1,
                        'target_n': n,
                        'r_peak_times': [],
                        'press_time': None,         # relative perf_counter
                        'press_abs_time': None,      # absolute unix timestamp
                        'press_delay_ms': None,
                        'correct': None,
                        'response_time': None,
                    })
        self.trial_idx = 0

    # ── data recording ──

    def _setup_data_recording(self):
        data_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'recordings_practice'
        )
        os.makedirs(data_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        pid = self.participant_id or 'unknown'
        session = self.session or '1'
        session_dir = os.path.join(data_dir, f'{pid}_session{session}_{ts}')
        os.makedirs(session_dir, exist_ok=True)
        self.session_data_path = session_dir

        ecg_path = os.path.join(session_dir, 'ecg_raw.csv')
        self._ecg_file = open(ecg_path, 'w', newline='')
        self._ecg_writer = csv.writer(self._ecg_file)
        self._ecg_writer.writerow(['timestamp', 'raw_value'])

        # ecg_log.csv — matches CardioChip ECGLog format: timestamp, raw, HR, HR_4s_avg, HR_30s_avg
        ecg_log_path = os.path.join(session_dir, 'ecg_log.csv')
        self._hr_log_file = open(ecg_log_path, 'w', newline='')
        self._hr_log_writer = csv.writer(self._hr_log_file)
        self._hr_log_writer.writerow(['timestamp', 'raw_value', 'hr_bpm', 'hr_4s_avg', 'hr_30s_avg'])

    # ── worker management ──

    def start_worker(self, port=None, simulate=False):
        self._simulate = simulate
        if simulate or not HAS_SERIAL:
            self._worker = SimulatedWorkerThread()
        else:
            self._worker = BMD101WorkerThread(port)
        self._worker.start()
        return self._worker

    def get_worker_queue(self):
        return self._worker.queue if self._worker else None

    def stop_worker(self):
        if self._worker:
            self._worker.stop()

    # ── experiment lifecycle ──

    def start_experiment(self, now):
        """Called once to begin the experiment."""
        self._exp_start_time = now
        self._exp_start_wall_time = time.time()  # absolute wall clock for export
        self._generate_trials()
        self._setup_data_recording()

        self.instruction_text = "正在连接..."
        self.state = 'starting'

    def begin_experiment(self, now):
        """Called after the initial 2s connection wait."""
        self.current_block = 1
        self.current_phase = 'feedback'
        self.current_trial = 0
        self.phase_correct = 0
        self.phase_total = 0
        self.trial_idx = 0
        self.state = 'ready'
        self._show_instruction_wait(
            "欢迎参加本次训练（练习模式）！\n\n"
            "训练分为两个阶段：\n"
            "  • 第一阶段（有反馈阶段）\n"
            "    您需要根据提示按键，屏幕上会根据您的实时心率呈现跃动的心脏图标。\n"
            "  • 第二阶段（无反馈阶段）\n"
            "    根据提示按键，但屏幕上不会再出现跃动的心脏图标。\n\n"
            "每次训练结束后会生成本次训练的得分。\n"
            "加油，希望我们在练习中提高成绩！\n\n"
            f"本练习共 {BLOCKS_PER_SESSION} 个组块，每阶段 {TRIALS_PER_PHASE} 个试次。\n\n"
            "准备好后，按任意键开始实验",
            self._welcome_to_phase1_prompt
        )

    def _show_instruction_wait(self, text, callback):
        self._wait_callback = callback
        self.state = 'wait_key'
        self.wait_key_text = text
        self.instruction_text = text

    def _welcome_to_phase1_prompt(self, now):
        """After welcome screen: prompt to enter phase 1."""
        self._show_instruction_wait(
            "说明已阅读完毕。\n\n"
            "按任意键进入第一阶段（有反馈阶段）",
            self._phase1_first_trial_prompt
        )

    def _phase1_first_trial_prompt(self, now):
        """Before first trial of phase 1: final ready prompt."""
        first_n = self.trials[0]['target_n'] if self.trials else N_MIN
        self._show_instruction_wait(
            "即将开始第一阶段第一次训练。\n\n"
            "请将注意力集中在心跳上，\n"
            "感受心脏的跳动节奏。\n\n"
            f"您需要在数到第 {first_n} 次心跳时按键。\n\n"
            "按任意键开始",
            self._start_block
        )

    # ── main loop hooks (called each frame) ──

    def _compute_hr_avg(self, window_sec):
        """Compute running average HR over the last N seconds."""
        if not self._hr_timestamps or self.heart_rate <= 0:
            return 0
        latest_ts = self._hr_timestamps[-1][0]
        recent = [hr for ts, hr in self._hr_timestamps if latest_ts - ts <= window_sec]
        return round(sum(recent) / len(recent), 1) if recent else 0

    def process_worker_event(self, event_type, payload):
        """Handle a single event from the worker queue."""
        if event_type == 'r_peak':
            now = time.perf_counter()
            self._r_peak_times.append(now)
            # RR-interval HR fallback: only when chip 0x03 HR is NOT available
            if not self._hr_from_chip and len(self._r_peak_times) >= 2:
                rr = now - self._r_peak_times[-2]
                if 0.24 <= rr <= 3.0:  # 20~250 BPM physiological range
                    self.heart_rate = round(60.0 / rr, 1)
                    # Also add to hr_timestamps so the running avg works
                    ts = self._exp_start_wall_time + (now - self._exp_start_time)
                    self._hr_timestamps.append((ts, self.heart_rate))
                    # Prune entries older than 30s
                    while len(self._hr_timestamps) > 2 and ts - self._hr_timestamps[0][0] > 30:
                        self._hr_timestamps.popleft()
            self._on_r_peak_experiment(now)
        elif event_type == 'raw_data':
            now = time.perf_counter()
            ts = round(now, 4)
            if self._ecg_writer:
                self._ecg_writer.writerow([ts, payload])
            # Write to ecg_log.csv with HR info
            if self._hr_log_writer:
                hr_4s = self._compute_hr_avg(4)
                hr_30s = self._compute_hr_avg(30)
                self._hr_log_writer.writerow([
                    ts, payload,
                    round(self.heart_rate, 1) if self.heart_rate > 0 else '',
                    hr_4s if hr_4s > 0 else '',
                    hr_30s if hr_30s > 0 else '',
                ])
        elif event_type == 'heart_rate':
            self.heart_rate = payload
            self._hr_from_chip = True  # mark that chip HR data is available
            now = time.perf_counter()
            ts = self._exp_start_wall_time + (now - self._exp_start_time)
            self._hr_timestamps.append((ts, payload))
            # Prune entries older than 30s
            while len(self._hr_timestamps) > 2 and ts - self._hr_timestamps[0][0] > 30:
                self._hr_timestamps.popleft()
        elif event_type == 'error':
            self.instruction_text = f"错误: {payload}"
        elif event_type == 'connected':
            pass  # handled separately

    def update(self, dt, now):
        """Called every frame — handles time-based state transitions.

        dt: seconds since last frame (PsychoPy clock delta)
        now: current time.perf_counter() value
        """
        if self.state == 'trial_active':
            # Check max trial duration (15s fallback)
            if now > self.max_trial_deadline and self.max_trial_deadline > 0:
                self._on_trial_timeout(now)

            # Check response window timeout (after target R-wave)
            if (self.target_r_peak_time > 0
                    and now > self.timeout_after_target
                    and self.timeout_after_target > 0
                    and not self.pressed):
                self._on_trial_timeout(now)

            # Activate red signal immediately after target R-wave
            if (self.current_phase == 'feedback'
                    and self.target_r_peak_time > 0
                    and not self.red_signal_active
                    and (now - self.target_r_peak_time) * 1000 >= HEART_APPEAR_DELAY_MS):
                self._activate_red_signal(now)

        elif self.state == 'trial_feedback':
            if now - self.feedback_start_time >= 1.5:
                self._deactivate_red_signal()
                # 检查下一个试次是否还在当前阶段（否则是阶段过渡，不显示 N 值）
                next_trial = self.trials[self.trial_idx] if self.trial_idx < len(self.trials) else None
                if next_trial is not None and next_trial['phase'] == self.current_phase:
                    next_text = f"下一次练习需要在第 {next_trial['target_n']} 次心跳时按键。\n\n按任意键继续进行练习"
                else:
                    next_text = "按任意键继续进行练习"
                self._show_instruction_wait(next_text, self._start_trial)

        elif self.state == 'rest':
            elapsed = now - self.rest_start_time
            remaining = REST_DURATION_S - elapsed
            self.rest_remaining = max(0, int(remaining))
            mins = self.rest_remaining // 60
            secs = self.rest_remaining % 60
            self.rest_display_text = (
                f"休息时间\n"
                f"{mins:02d}:{secs:02d}"
            )
            if remaining <= 0:
                self._on_rest_done(now)

    # ── keyboard handling ──

    def handle_any_key(self, now):
        """Called from main loop when any key is pressed."""
        if self.state == 'wait_key' and self._wait_callback:
            cb = self._wait_callback
            self._wait_callback = None
            cb(now)

    def handle_space(self, now):
        """Called when space bar is pressed."""
        if self.state == 'trial_active' and not self.pressed:
            self._handle_keypress(now)

    # ── internal state transitions ──

    def _on_r_peak_experiment(self, now):
        if self.state == 'trial_active':
            self.trial_r_wave_count += 1
            self.heartbeat_count_text = f"心跳计数: {self.trial_r_wave_count}/{self.current_target_n}"
            # 实验说明：R波后立即呈现动态心脏图标
            self.heart_beat_time = now + HEART_APPEAR_DELAY_MS / 1000.0

            idx = self.trial_idx - 1
            if 0 <= idx < len(self.trials):
                self.trials[idx]['r_peak_times'].append(now)

            # Target R-wave?
            if self.trial_r_wave_count == self.current_target_n:
                self.target_r_peak_time = now
                timeout_delay = (RESP_WIN_END + int(TIMEOUT_AFTER_TARGET_S * 1000)) / 1000
                self.timeout_after_target = now + timeout_delay

    def _activate_red_signal(self, now):
        self.red_signal_active = True
        self.red_signal_pulse_start = now

    def _deactivate_red_signal(self):
        self.red_signal_active = False

    def _start_block(self, now):
        if self.current_block > BLOCKS_PER_SESSION:
            self._finish_experiment(now)
            return

        self.current_phase = 'feedback'
        self.current_trial = 0
        self.phase_correct = 0
        self.phase_total = 0

        self.phase_info_text = f"组块 {self.current_block}/{BLOCKS_PER_SESSION} — 反馈阶段（练习模式）"
        self.state = 'block_active'
        self._start_phase(now)

    def _start_phase(self, now):
        self.current_trial = 0
        self.phase_correct = 0
        self.phase_total = 0
        self.accuracy = 0.0

        if self.current_phase == 'feedback':
            self.phase_info_text = f"组块 {self.current_block}/{BLOCKS_PER_SESSION} — 反馈阶段（练习模式）"
        else:
            self.phase_info_text = f"组块 {self.current_block}/{BLOCKS_PER_SESSION} — 无反馈阶段（练习模式）"

        self.state = 'phase_active'
        self._start_trial(now)

    def _start_trial(self, now):
        if self.trial_idx >= len(self.trials):
            self._finish_phase(now)
            return

        trial = self.trials[self.trial_idx]
        if trial['phase'] != self.current_phase:
            self._finish_phase(now)
            return

        self.trial_idx += 1
        self.current_trial += 1
        self.current_target_n = trial['target_n']
        self.trial_r_wave_count = 0
        self.target_r_peak_time = 0.0
        self.pressed = False
        self.timeout_after_target = 0.0
        self.max_trial_deadline = 0.0
        self.red_signal_active = False
        self.heart_beat_time = -1.0  # sentinel: no real R-wave yet
        self.feedback_type = ''
        self.feedback_text = ''

        self.trial_progress_text = f"试次 {self.current_trial}/{TRIALS_PER_PHASE}"
        self.instruction_text = f"请在 第 {self.current_target_n} 次 心跳时按键"
        self.heartbeat_count_text = f"心跳计数: 0/{self.current_target_n}"

        self.trial_start_time = now
        self.state = 'trial_active'

        # Max trial duration safety: 15s
        self.max_trial_deadline = now + 15.0

    def _handle_keypress(self, now):
        self.pressed = True
        self.timeout_after_target = 0.0
        self.max_trial_deadline = 0.0

        idx = self.trial_idx - 1
        if 0 <= idx < len(self.trials):
            self.trials[idx]['press_time'] = now
            # Absolute wall-clock timestamp (Unix time, seconds)
            self.trials[idx]['press_abs_time'] = (
                self._exp_start_wall_time + (now - self._exp_start_time)
            )

        correct = False
        delay_ms = None
        if self.target_r_peak_time > 0:
            delay_ms = (now - self.target_r_peak_time) * 1000
            self.trials[idx]['press_delay_ms'] = delay_ms
            self.trials[idx]['response_time'] = now - self.trial_start_time
            if RESP_WIN_START <= delay_ms <= RESP_WIN_END:
                correct = True

        if correct:
            self.phase_correct += 1
            self.feedback_type = 'correct'
            self.feedback_text = '正确！'
        else:
            self.feedback_type = 'incorrect'
            self.feedback_text = '错误'

        self.phase_total += 1
        accuracy = (self.phase_correct / self.phase_total) * 100
        self.trials[idx]['correct'] = correct
        self.accuracy = round(accuracy, 1)
        self._deactivate_red_signal()

        self.state = 'trial_feedback'
        self.feedback_start_time = now

    def _on_trial_timeout(self, now):
        if self.pressed:
            return
        self.pressed = True
        self.timeout_after_target = 0.0
        self.max_trial_deadline = 0.0

        self.phase_total += 1
        accuracy = (self.phase_correct / self.phase_total) * 100
        self.accuracy = round(accuracy, 1)

        idx = self.trial_idx - 1
        if 0 <= idx < len(self.trials):
            self.trials[idx]['correct'] = False
            self.trials[idx]['press_time'] = None
            self.trials[idx]['press_delay_ms'] = None

        self._deactivate_red_signal()
        self.feedback_type = 'timeout'
        self.feedback_text = '超时'
        self.state = 'trial_feedback'
        self.feedback_start_time = now

    def _finish_phase(self, now):
        self._deactivate_red_signal()
        accuracy = 0
        if self.phase_total > 0:
            accuracy = (self.phase_correct / self.phase_total) * 100
        self.accuracy = round(accuracy, 1)
        self.state = 'phase_summary'

        if self.current_phase == 'feedback':
            self.current_phase = 'nofeedback'
            self.phase_info_text = (
                f"组块 {self.current_block}/{BLOCKS_PER_SESSION} — 无反馈阶段（练习模式）"
            )
            self._show_instruction_wait(
                f"反馈阶段结束！正确率: {accuracy:.1f}%\n\n"
                "即将进入无反馈阶段。\n"
                "在该阶段，屏幕不再显示跃动的心脏图标，\n"
                "但仍会有 ✓ 和 ✗ 的按键正确性反馈。\n"
                "请继续保持专注，感受心跳时按键。\n\n"
                "准备好后，按任意键继续",
                self._start_phase
            )
        else:
            if self.current_block < BLOCKS_PER_SESSION:
                self._show_instruction_wait(
                    f"组块 {self.current_block} 无反馈阶段结束！正确率: {accuracy:.1f}%\n\n"
                    "即将进入休息时间。\n\n"
                    "准备好后，按任意键开始休息",
                    self._start_rest
                )
            else:
                self._show_instruction_wait(
                    f"所有组块完成！最终正确率: {accuracy:.1f}%\n\n"
                    "实验即将结束。\n\n"
                    "按任意键查看结果",
                    self._finish_experiment
                )

    def _start_rest(self, now):
        self.state = 'rest'
        self.rest_start_time = now
        self.show_rest = True

    def _on_rest_done(self, now):
        self.show_rest = False
        self.current_block += 1
        self._show_instruction_wait(
            f"休息结束！\n\n"
            f"即将进入组块 {self.current_block}/{BLOCKS_PER_SESSION}\n"
            "请准备好，保持专注。\n\n"
            "按任意键继续",
            self._start_block
        )

    def _build_results_text(self):
        """Build detailed results summary from completed trials."""
        completed = [t for t in self.trials if t['correct'] is not None]
        total_completed = len(completed)
        total_correct = sum(1 for t in completed if t['correct'])
        overall_acc = (total_correct / total_completed * 100) if total_completed > 0 else 0

        lines = []
        lines.append(f"被试编号: {self.participant_id}    第 {self.session} 次训练（练习版）")
        lines.append("")
        lines.append(f"═══  总体正确率: {overall_acc:.1f}% ({total_correct}/{total_completed})  ═══")
        lines.append("")

        for block in range(1, BLOCKS_PER_SESSION + 1):
            block_trials = [t for t in completed if t['block'] == block]
            block_total = len(block_trials)
            if block_total == 0:
                continue
            block_correct = sum(1 for t in block_trials if t['correct'])
            block_acc = (block_correct / block_total * 100)

            lines.append(f"── 组块 {block} ──")
            for phase in ['feedback', 'nofeedback']:
                phase_label = "有反馈" if phase == 'feedback' else "无反馈"
                p_trials = [t for t in block_trials if t['phase'] == phase]
                p_total = len(p_trials)
                if p_total == 0:
                    continue
                p_correct = sum(1 for t in p_trials if t['correct'])
                p_acc = (p_correct / p_total * 100)
                lines.append(f"  {phase_label}: {p_acc:.1f}% ({p_correct}/{p_total})")
            lines.append(f"  小计: {block_acc:.1f}% ({block_correct}/{block_total})")
            lines.append("")

        lines.append("按任意键退出")
        return "\n".join(lines)

    def _finish_experiment(self, now):
        self._save_all_data()
        self.state = 'complete'
        self.instruction_text = self._build_results_text()
        self.feedback_type = ''
        self.feedback_text = ''
        self._results_start_time = now  # mark when results first appeared
        # Clear event buffer: the key that triggered this transition
        # should NOT also immediately exit the results screen.
        event.clearEvents()

    # ── data saving ──

    def _save_all_data(self):
        if not self.session_data_path:
            return

        trial_file = os.path.join(self.session_data_path, 'trials.csv')
        with open(trial_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'trial_idx', 'block', 'phase', 'trial_num',
                'target_n', 'correct', 'press_abs_time',
                'press_delay_ms', 'response_time_s'
            ])
            for i, t in enumerate(self.trials, 1):
                writer.writerow([
                    i, t['block'], t['phase'], t['trial'],
                    t['target_n'], t['correct'],
                    round(t['press_abs_time'], 4) if t['press_abs_time'] is not None else '',
                    round(t['press_delay_ms'], 2) if t['press_delay_ms'] is not None else '',
                    round(t['response_time'], 3) if t['response_time'] is not None else ''
                ])

        rpeak_file = os.path.join(self.session_data_path, 'r_peaks.csv')
        with open(rpeak_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp'])
            for t in self._r_peak_times:
                writer.writerow([round(t, 4)])

        total_correct = sum(1 for t in self.trials if t['correct'])
        total_trials = len(self.trials)
        summary = {
            'participant_id': self.participant_id,
            'session': self.session,
            'timestamp': datetime.now().isoformat(),
            'mode': 'practice',
            'total_trials': total_trials,
            'total_correct': total_correct,
            'overall_accuracy': round(
                (total_correct / total_trials * 100), 1
            ) if total_trials > 0 else 0,
            'blocks': BLOCKS_PER_SESSION,
            'trials_per_phase': TRIALS_PER_PHASE,
        }

        summary_file = os.path.join(self.session_data_path, 'summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        # Close ECG file
        if self._ecg_file:
            self._ecg_file.close()
            self._ecg_file = None

        # Close HR log file
        if self._hr_log_file:
            self._hr_log_file.close()
            self._hr_log_file = None

        print(f"数据已保存至: {self.session_data_path}")

    # ── helper ──

    def reset_detector(self):
        if self._worker:
            self._worker.reset_detector()


# ══════════════════════════════════════════════════════════════
#  Heart beat animation
# ══════════════════════════════════════════════════════════════
def get_heart_scale(elapsed_ms):
    """Return the heart scale factor for the given elapsed ms since beat."""
    if elapsed_ms > HEART_KEYFRAMES[-1][0]:
        return 1.0
    for i in range(len(HEART_KEYFRAMES) - 1):
        t0, s0 = HEART_KEYFRAMES[i]
        t1, s1 = HEART_KEYFRAMES[i + 1]
        if t0 <= elapsed_ms <= t1:
            frac = (elapsed_ms - t0) / (t1 - t0) if t1 != t0 else 0
            return s0 + (s1 - s0) * frac
    return 1.0


# ══════════════════════════════════════════════════════════════
#  Red pulse overlay opacity
# ══════════════════════════════════════════════════════════════
def get_red_pulse_opacity(elapsed_ms):
    """Return opacity [0..1] for the red border pulse."""
    phase = (elapsed_ms % RED_PULSE_PERIOD) / RED_PULSE_PERIOD
    # Smooth oscillation: 0.3 -> 1.0 -> 0.3
    return 0.3 + 0.7 * (0.5 - 0.5 * math.cos(phase * 2 * math.pi))


# ══════════════════════════════════════════════════════════════
#  Config Dialog（练习版：无对话框，直接模拟模式）
# ══════════════════════════════════════════════════════════════
def show_config_dialog():
    """Show PsychoPy GUI dialog for experiment configuration."""
    port_options = ["模拟模式"]
    if HAS_SERIAL:
        try:
            for p in serial.tools.list_ports.comports():
                port_options.append(f"{p.device} - {p.description}")
        except Exception:
            pass

    dlg = gui.Dlg(title="ECG 生物反馈训练 — 练习模式")
    dlg.addField("被试编号:", "P001")
    dlg.addField("训练次数:", "1")
    dlg.addField("串口:", choices=port_options)

    dlg_data = dlg.show()
    if dlg.OK:
        pid = dlg_data[0].strip() or "unknown"
        session = dlg_data[1].strip() or "1"
        port_choice = dlg_data[2]
        simulate = (port_choice == "模拟模式" or not port_choice)
        port = None if simulate else port_choice.split(" - ")[0]
        return pid, session, port, simulate
    return None


# ══════════════════════════════════════════════════════════════
#  Main Experiment Class (PsychoPy)
# ══════════════════════════════════════════════════════════════
class PsychopyExperiment:
    """Top-level experiment runner using PsychoPy for display and input."""

    def __init__(self):
        self.win = None
        self.controller = ExperimentController()
        self.clock = Clock()
        self._running = False
        self._started = False
        self._connection_timer = 3.0  # seconds to wait for connection

        # Stimuli (created in _create_stimuli)
        self.stim = {}

        # Worker queue reference
        self._worker_queue = None

        # Heart beat tracking
        self._last_heart_beat_time = 0.0  # time.perf_counter

        # Dialog config
        self.config = None

    # ── setup ──

    def setup(self, config):
        """Create window and stimuli from dialog config."""
        self.config = config
        pid, session, port, simulate = config

        self.controller.set_participant_info(pid, session)

        # Create window (fullscreen, dark theme)
        self.win = visual.Window(
            fullscr=True,
            color='#1a1a2e',
            units='norm',
            allowStencil=False,
        )

        # Start worker thread
        self._worker_queue = self.controller.start_worker(port, simulate)
        self._running = True

        # Create all stimuli
        self._create_stimuli()

        # Start experiment
        now = time.perf_counter()
        self.controller.start_experiment(now)
        self._started = True
        self._connection_start_time = now

        # Hide mouse cursor
        self.win.mouseVisible = False

    def _find_chinese_font(self):
        """Find an available Chinese font on the current system."""
        if sys.platform == 'darwin':
            return 'PingFang SC'
        elif sys.platform == 'linux':
            return 'WenQuanYi Micro Hei'
        # Windows: try common Chinese fonts
        candidates = ['Microsoft YaHei', 'SimHei', 'DengXian',
                      'Microsoft JhengHei', 'SimSun', 'Arial Unicode MS']
        try:
            import pyglet.font
            for name in candidates:
                try:
                    pyglet.font.load(name)
                    return name
                except Exception:
                    continue
        except ImportError:
            pass
        return 'Microsoft YaHei'  # fallback default

    def _create_stimuli(self):
        """Create all PsychoPy stimuli."""
        # Font settings: try to use Chinese-capable font
        chinese_font = self._find_chinese_font()

        # ── Phase info (top-left) ──
        self.stim['phase_info'] = visual.TextStim(
            self.win, text="准备开始",
            pos=(-0.85, 0.92), alignText='left', anchorHoriz='left',
            height=0.045, font=chinese_font,
            color='white',
        )

        # ── Heart rate (top-right) ──
        self.stim['hr_label'] = visual.TextStim(
            self.win, text="心率:",
            pos=(0.55, 0.92), alignText='right', anchorHoriz='right',
            height=0.04, font=chinese_font,
            color='white',
        )
        self.stim['hr_value'] = visual.TextStim(
            self.win, text="-- BPM",
            pos=(0.80, 0.92), alignText='right', anchorHoriz='right',
            height=0.05, font=chinese_font,
            color='#ff6b6b', bold=True,
        )

        # ── Accuracy (top-center area) ──
        self.stim['accuracy'] = visual.TextStim(
            self.win, text="正确率: 0%",
            pos=(0, 0.35), alignText='center', anchorHoriz='center',
            height=0.05, font=chinese_font,
            color='#4ecdc4', bold=True,
        )

        # ── Instruction / main text ──
        self.stim['instruction'] = visual.TextStim(
            self.win, text="准备开始",
            pos=(0, 0.05), alignText='center', anchorHoriz='center',
            height=0.09, font=chinese_font,
            color='white', bold=True,
            wrapWidth=1.6,
        )

        # ── Heartbeat count ──
        self.stim['heartbeat_count'] = visual.TextStim(
            self.win, text="",
            pos=(0, -0.18), alignText='center', anchorHoriz='center',
            height=0.065, font=chinese_font,
            color='#a0a0c0',
        )

        # ── Feedback (big ✓ / ✗ / —) ──
        self.stim['feedback'] = visual.TextStim(
            self.win, text="",
            pos=(0, -0.35), alignText='center', anchorHoriz='center',
            height=0.16, font=chinese_font,
            color='white', bold=True,
        )

        # ── Heart icon (ShapeStim) ──
        heart_verts = _make_heart_vertices(1.0)
        self.stim['heart'] = visual.ShapeStim(
            self.win, vertices=heart_verts,
            fillColor='#f05050',
            lineColor='#dc2828',
            lineWidth=2,
            pos=(0, 0),
            size=(0.15, 0.15),  # scaled size
            ori=0,
            opacity=0,  # hidden by default
        )
        self.stim['heart'].setAutoDraw(False)

        # ── Red overlay (border glow) ──
        self.stim['red_overlay_1'] = visual.Rect(
            self.win, width=2, height=2,
            pos=(0, 0),
            lineColor='red',
            lineWidth=30,
            fillColor=None,
            opacity=0,
        )
        self.stim['red_overlay_2'] = visual.Rect(
            self.win, width=2, height=2,
            pos=(0, 0),
            lineColor='red',
            lineWidth=20,
            fillColor=None,
            opacity=0,
        )
        self.stim['red_overlay_3'] = visual.Rect(
            self.win, width=2, height=2,
            pos=(0, 0),
            lineColor='red',
            lineWidth=10,
            fillColor=None,
            opacity=0,
        )

        # ── Trial progress (bottom-left) ──
        self.stim['trial_progress'] = visual.TextStim(
            self.win, text="",
            pos=(-0.85, -0.92), alignText='left', anchorHoriz='left',
            height=0.04, font=chinese_font,
            color='#a0a0c0',
        )

        # ── Status (bottom-right) ──
        self.stim['status'] = visual.TextStim(
            self.win, text="就绪",
            pos=(0.85, -0.92), alignText='right', anchorHoriz='right',
            height=0.04, font=chinese_font,
            color='#a0a0c0',
        )

        # ── Rest screen text (overrides instruction during rest) ──
        self.stim['rest_text'] = visual.TextStim(
            self.win, text="",
            pos=(0, 0.05), alignText='center', anchorHoriz='center',
            height=0.09, font=chinese_font,
            color='white', bold=True,
            wrapWidth=1.6,
            autoDraw=False,
        )

    # ── main run loop ──

    def run(self):
        """Main experiment loop."""
        if not self._running:
            return

        # Wait for initial connection briefly, then begin
        while self._running:
            now_time = time.perf_counter()

            # ── 1. Process worker queue ──
            self._drain_worker_queue()

            # ── 2. Update state machine ──
            dt = self.clock.reset()
            self.controller.update(dt, now_time)

            # ── 3. Handle connection timer ──
            if self.controller.state == 'starting':
                if now_time - self._connection_start_time > 2.0:
                    self.controller.begin_experiment(now_time)

            # ── 4. Handle keyboard ──
            keys = event.getKeys()
            self._handle_keys(keys, now_time)

            # ── 5. Check exit ──
            if 'escape' in keys:
                # Experiment already complete — no confirm needed
                if self.controller.state == 'complete':
                    break
                if self._confirm_exit():
                    break

            # ── 6. Draw ──
            self._draw(now_time)

            # ── 7. Flip ──
            self.win.flip()

        # ── cleanup ──
        self._cleanup()

    def _drain_worker_queue(self):
        """Process all pending worker events."""
        q = self.controller.get_worker_queue()
        if q is None:
            return
        while True:
            try:
                event_type, payload = q.get_nowait()
                self.controller.process_worker_event(event_type, payload)
            except queue.Empty:
                break

    def _handle_keys(self, keys, now):
        """Process keyboard input."""
        if not keys:
            return

        # Escape is handled in main loop for exit
        # Filter out escape for experiment logic
        non_escape = [k for k in keys if k != 'escape']

        if not non_escape:
            return

        # Any key in complete state — clean exit (experiment is done)
        # but only after minimum display time to prevent key-repeat flash
        if self.controller.state == 'complete':
            # Enforce minimum 2-second display so the results page doesn't
            # flash away due to OS-level key repeat events.
            if now - self.controller._results_start_time >= 2.0:
                self._running = False
            return

        # Any key in wait_key state
        if self.controller.state == 'wait_key':
            self.controller.handle_any_key(now)
            return

        # Space bar in trial_active
        if 'space' in keys:
            self.controller.handle_space(now)

    def _confirm_exit(self):
        """Ask user to confirm exit. Returns True if should exit."""
        # In PsychoPy, we show a simple text prompt and wait for y/n
        confirm_text = visual.TextStim(
            self.win,
            text="实验正在进行中，确定要退出吗？\n\n按 Y 退出，按 N 继续",
            pos=(0, 0), alignText='center', anchorHoriz='center',
            height=0.065, color='yellow',
            font=self.stim['instruction'].font,
        )
        confirm_text.draw()
        self.win.flip()

        # Wait for Y or N
        event.clearEvents()
        while True:
            resp = event.waitKeys(keyList=['y', 'n', 'escape'])
            if resp:
                if resp[0] == 'y' or resp[0] == 'escape':
                    return True
                else:
                    return False

    def _draw(self, now):
        """Draw all stimuli for the current frame."""
        ctrl = self.controller

        # ── Rest screen overrides everything ──
        if ctrl.state == 'rest' and ctrl.show_rest:
            self.stim['rest_text'].text = ctrl.rest_display_text
            self.stim['rest_text'].setAutoDraw(True)
            # Hide other elements
            for name in ['phase_info', 'hr_label', 'hr_value', 'accuracy',
                         'instruction', 'heartbeat_count', 'feedback', 'heart',
                         'trial_progress', 'status']:
                if name in self.stim:
                    self.stim[name].setAutoDraw(False)
            # Hide red overlays
            for i in [1, 2, 3]:
                self.stim[f'red_overlay_{i}'].opacity = 0
                self.stim[f'red_overlay_{i}'].setAutoDraw(False)

            # Also show phase info during rest
            self.stim['phase_info'].setAutoDraw(True)
            self.stim['phase_info'].text = ctrl.phase_info_text
            return

        # ── Normal display ──
        self.stim['rest_text'].setAutoDraw(False)

        # ── Wait key screen (居中左对齐说明) ──
        if ctrl.state == 'wait_key':
            # 隐藏试次相关元素，避免重叠
            self.stim['hr_label'].setAutoDraw(False)
            self.stim['hr_value'].setAutoDraw(False)
            self.stim['trial_progress'].setAutoDraw(False)
            self.stim['feedback'].setAutoDraw(False)
            self.stim['heartbeat_count'].setAutoDraw(False)
            self.stim['heart'].setAutoDraw(False)
            # Phase info
            self.stim['phase_info'].setAutoDraw(True)
            self.stim['phase_info'].text = ctrl.phase_info_text
            # Accuracy (top-center, useful for inter-trial feedback)
            self.stim['accuracy'].setAutoDraw(True)
            self.stim['accuracy'].pos = (0, 0.40)
            self.stim['accuracy'].text = f"正确率: {ctrl.accuracy:.1f}%"
            # Instruction: centered on screen, text left-aligned within
            self.stim['instruction'].setAutoDraw(True)
            self.stim['instruction'].text = ctrl.wait_key_text
            self.stim['instruction'].height = 0.065
            self.stim['instruction'].color = 'white'
            self.stim['instruction'].pos = (0, 0)
            self.stim['instruction'].alignText = 'left'
            self.stim['instruction'].anchorHoriz = 'center'
            self.stim['instruction'].wrapWidth = 1.4
            # Status
            self.stim['status'].setAutoDraw(True)
            self.stim['status'].text = "等待按键 — 按任意键继续"
            return

        # ── Complete / Results screen ──
        if ctrl.state == 'complete':
            # Hide all trial-specific elements
            for name in ['phase_info', 'hr_label', 'hr_value', 'accuracy',
                         'feedback', 'heart', 'heartbeat_count',
                         'trial_progress', 'status']:
                if name in self.stim:
                    self.stim[name].setAutoDraw(False)
            for i in [1, 2, 3]:
                self.stim[f'red_overlay_{i}'].setAutoDraw(False)
                self.stim[f'red_overlay_{i}'].opacity = 0
            # Show results text with smaller font
            self.stim['instruction'].setAutoDraw(True)
            self.stim['instruction'].text = ctrl.instruction_text
            self.stim['instruction'].height = 0.045
            self.stim['instruction'].color = 'white'
            self.stim['instruction'].pos = (0, 0.25)
            self.stim['instruction'].alignText = 'left'
            self.stim['instruction'].anchorHoriz = 'center'
            self.stim['instruction'].wrapWidth = 1.5
            return

        # Phase info (reset alignText for non-wait_key, non-complete states)
        self.stim['phase_info'].setAutoDraw(True)
        self.stim['phase_info'].text = ctrl.phase_info_text

        # HR — use 4s moving average for stable display
        self.stim['hr_label'].setAutoDraw(True)
        if ctrl.heart_rate > 0:
            smoothed = ctrl._compute_hr_avg(4)
            if smoothed > 0:
                self.stim['hr_value'].text = f"{smoothed:.0f} BPM"
            else:
                self.stim['hr_value'].text = f"{ctrl.heart_rate:.0f} BPM"
        self.stim['hr_value'].setAutoDraw(True)

        # Accuracy (reset pos after wait_key)
        self.stim['accuracy'].pos = (0, 0.35)
        self.stim['accuracy'].setAutoDraw(True)
        self.stim['accuracy'].text = f"正确率: {ctrl.accuracy:.1f}%"

        # Instruction / main text (reset pos/align after wait_key)
        self.stim['instruction'].pos = (0, 0.20)
        self.stim['instruction'].alignText = 'center'
        self.stim['instruction'].anchorHoriz = 'center'
        self.stim['instruction'].setAutoDraw(True)
        if ctrl.state == 'trial_active':
            self.stim['instruction'].text = ctrl.instruction_text
            self.stim['instruction'].height = 0.09
            self.stim['instruction'].color = 'white'
        elif ctrl.state == 'trial_feedback':
            self.stim['instruction'].text = ctrl.instruction_text
            self.stim['instruction'].height = 0.075
            self.stim['instruction'].color = '#cccccc'
        else:
            self.stim['instruction'].text = ctrl.instruction_text
            self.stim['instruction'].height = 0.09
            self.stim['instruction'].color = 'white'

        # Heartbeat count — 已隐藏（被试不需要看到计数）
        self.stim['heartbeat_count'].setAutoDraw(False)

        # Feedback symbol
        if ctrl.state == 'trial_feedback' or (
            ctrl.state == 'complete' and ctrl.feedback_type
        ):
            self.stim['feedback'].setAutoDraw(True)
            if ctrl.feedback_type == 'correct':
                self.stim['feedback'].text = '✓'
                self.stim['feedback'].color = '#2ecc71'
            elif ctrl.feedback_type == 'incorrect':
                self.stim['feedback'].text = '✗'
                self.stim['feedback'].color = '#e74c3c'
            elif ctrl.feedback_type == 'timeout':
                self.stim['feedback'].text = '—'
                self.stim['feedback'].color = '#f39c12'
            else:
                self.stim['feedback'].setAutoDraw(False)
        else:
            self.stim['feedback'].setAutoDraw(False)

        # Heart icon (feedback phase only)
        if (ctrl.current_phase == 'feedback' and ctrl.state in ('trial_active', 'trial_feedback')
                and ctrl.heart_beat_time > 0):
            self.stim['heart'].setAutoDraw(True)
            # Animate heart scale based on time since last beat
            elapsed = (now - ctrl.heart_beat_time) * 1000
            scale = get_heart_scale(elapsed)
            base_size = 0.15
            self.stim['heart'].size = (base_size * scale, base_size * scale)
            self.stim['heart'].opacity = 1.0
        else:
            self.stim['heart'].setAutoDraw(False)
            self.stim['heart'].opacity = 0

        # Red signal overlay (feedback phase, after target R-wave)
        if ctrl.red_signal_active and ctrl.current_phase == 'feedback':
            elapsed_red = (now - ctrl.red_signal_pulse_start) * 1000
            opacity = get_red_pulse_opacity(elapsed_red)

            for i in [1, 2, 3]:
                overlay = self.stim[f'red_overlay_{i}']
                overlay.setAutoDraw(True)
            # Layer 1 (outermost, thickest)
            self.stim['red_overlay_1'].lineWidth = 30
            self.stim['red_overlay_1'].opacity = opacity * 0.15
            self.stim['red_overlay_1'].lineColor = '#ff0000'
            # Layer 2 (middle)
            self.stim['red_overlay_2'].lineWidth = 20
            self.stim['red_overlay_2'].opacity = opacity * 0.35
            self.stim['red_overlay_2'].lineColor = '#ff0000'
            # Layer 3 (inner, brightest)
            self.stim['red_overlay_3'].lineWidth = 10
            self.stim['red_overlay_3'].opacity = opacity * 0.7
            self.stim['red_overlay_3'].lineColor = '#ff0000'
        else:
            for i in [1, 2, 3]:
                self.stim[f'red_overlay_{i}'].setAutoDraw(False)

        # Trial progress
        self.stim['trial_progress'].setAutoDraw(True)
        self.stim['trial_progress'].text = ctrl.trial_progress_text

        # Status
        self.stim['status'].setAutoDraw(True)
        if ctrl.state in ('trial_active', 'trial_feedback'):
            self.stim['status'].text = "运行中 — 数到指定心跳时按 Space"
        elif ctrl.state == 'rest':
            self.stim['status'].text = "休息中..."
        elif ctrl.state == 'complete':
            self.stim['status'].text = "练习完成 — 按 Esc 退出"
        else:
            self.stim['status'].text = ""

    def _cleanup(self):
        """Clean up resources."""
        self.controller.stop_worker()
        self.controller._save_all_data()
        if self.win:
            self.win.close()
        self._running = False


# ══════════════════════════════════════════════════════════════
#  Entry Point
# ══════════════════════════════════════════════════════════════
def main():
    # 练习版：无对话框，直接以模拟模式启动
    config = show_config_dialog()
    if config is None:
        print("用户取消了实验")
        return

    # Create and run experiment
    exp = PsychopyExperiment()
    exp.setup(config)

    try:
        exp.run()
    except Exception as e:
        import traceback
        traceback.print_exc()
        # Try to save data on crash
        try:
            exp.controller._save_all_data()
        except Exception:
            pass
    finally:
        core.quit()


if __name__ == '__main__':
    main()
