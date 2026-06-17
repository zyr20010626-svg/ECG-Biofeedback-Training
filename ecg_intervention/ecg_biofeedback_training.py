#!/usr/bin/env python3
"""
ECG Biofeedback Training Program
内感受视觉训练干预方案 - BMD101 心电传感器版本

实验流程：
- 3个训练组块，每块包含反馈/无反馈两个阶段
- 每阶段24个试次，共144试次（约30分钟）
- 反馈阶段：R波后立即呈现动态心脏图标 + 红色脉冲边框
- 按键窗口：指定R波后0ms-500ms
"""

import sys
import os
import random
import time
import csv
import json
from datetime import datetime
from collections import deque

import numpy as np

from PyQt5 import QtWidgets, QtCore, QtGui

# ============================================================
# Constants
# ============================================================
FS = 512                # BMD101 sampling rate (Hz)
R_PEAK_REFRACTORY_MS = 200
RESP_WIN_START = 0      # ms after target R-wave (实验说明: 0–500ms)
RESP_WIN_END = 500      # ms after target R-wave
TRIALS_PER_PHASE = 24  # 每阶段试次数
BLOCKS_PER_SESSION = 3
REST_DURATION_S = 300  # 5 minutes
TIMEOUT_AFTER_TARGET_S = 3.0  # wait for keypress after response window closes
HEART_APPEAR_DELAY_MS = 0    # delay before heart icon and red signal appear after R-wave (实验说明: 0ms 立即呈现)
N_MIN = 5
N_MAX = 10

# ============================================================
# scipy import (separate to handle ImportError gracefully)
# ============================================================
try:
    from scipy import signal as scipy_signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ============================================================
# Serial import
# ============================================================
try:
    import serial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

# ============================================================
# R-Peak Detector
# ============================================================
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

        self.signal_history = deque(maxlen=fs)  # 1 second window
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

        # Update adaptive threshold
        if len(self.signal_history) > int(0.1 * self.fs):
            recent = np.array(list(self.signal_history))
            self.threshold = float(np.mean(recent) + 1.5 * np.std(recent))

        is_peak = False
        if (self.filtered_out > self.threshold
                and self.since_last_peak > self.refractory_samples
                and len(self.signal_history) >= 5):
            # Local maximum check
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


# ============================================================
# BMD101 Serial Worker (runs in QThread)
# ============================================================
class BMD101Worker(QtCore.QObject):
    """Reads and parses BMD101 ThinkGear protocol in a worker thread."""

    raw_data = QtCore.pyqtSignal(int)       # raw ECG value
    r_peak = QtCore.pyqtSignal()            # R-wave detected
    heart_rate = QtCore.pyqtSignal(float)   # computed HR (BPM)
    error = QtCore.pyqtSignal(str)
    connected = QtCore.pyqtSignal(bool)

    def __init__(self, port, baud=57600):
        super().__init__()
        self.port = port
        self.baud = baud
        self._running = False
        self._serial = None
        self.detector = RPeakDetector(FS)

        # RR-interval based HR calculation
        self._last_r_peak_time = 0.0
        self._rr_intervals = deque(maxlen=10)

    def run(self):
        """Main loop: read serial, parse packets, detect R-peaks."""
        if not HAS_SERIAL:
            self.error.emit("pyserial not installed. Run: pip install pyserial")
            return

        try:
            self._serial = serial.Serial(
                self.port, baudrate=self.baud, timeout=0.5
            )
            self.connected.emit(True)
        except Exception as e:
            self.error.emit(f"无法打开串口 {self.port}: {e}")
            return

        self._running = True
        while self._running:
            try:
                self._parse_packet()
            except serial.SerialException as e:
                self.error.emit(f"串口错误: {e}")
                break
            except Exception:
                # Packet parsing errors are common; skip and retry
                pass

        if self._serial and self._serial.is_open:
            self._serial.close()
        self.connected.emit(False)

    def stop(self):
        self._running = False

    def _parse_packet(self):
        if not self._serial or not self._serial.is_open:
            return

        # Look for 0xAA 0xAA sync
        if self._serial.read(1) != b'\xaa':
            return
        if self._serial.read(1) != b'\xaa':
            return

        pkt_len_byte = self._serial.read(1)
        if not pkt_len_byte:
            return
        pkt_len = pkt_len_byte[0]

        payload = self._serial.read(pkt_len + 1)  # +1 for checksum
        if len(payload) < pkt_len + 1:
            return  # incomplete packet

        received_checksum = payload[-1]
        data_bytes = payload[:-1]
        # ThinkGear checksum: one's complement of sum
        calc_checksum = (~sum(data_bytes)) & 0xFF

        if calc_checksum != received_checksum:
            return  # checksum mismatch

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
                self.raw_data.emit(raw_val)

                # R-peak detection
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
                        self.r_peak.emit()

            elif code == 0x12 and row_len == 4:
                # 0x12 packets (firmware-specific data) — skip
                pass

            elif code == 0x03:
                # HEART_RATE: BMD101 chip's internal DSP-computed HR (BPM)
                # CardioChip uses the same value via ThinkGear.dll → accurate!
                if row_len >= 1:
                    hr_val = data_bytes[i + 2]
                    if 20 <= hr_val <= 250:  # physiological range
                        self.heart_rate.emit(float(hr_val))

            i += 2 + row_len

    def reset_detector(self):
        self.detector.reset()
        self._last_r_peak_time = 0.0
        self._rr_intervals.clear()


# ============================================================
# Simulated Worker (for testing without hardware)
# ============================================================
class SimulatedWorker(QtCore.QObject):
    """Generates synthetic ECG-like R-peaks for testing."""

    raw_data = QtCore.pyqtSignal(int)
    r_peak = QtCore.pyqtSignal()
    heart_rate = QtCore.pyqtSignal(float)
    error = QtCore.pyqtSignal(str)
    connected = QtCore.pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self._running = False

    def run(self):
        self._running = True
        self.connected.emit(True)

        while self._running:
            # Simulate heart rate variability (60-80 BPM range)
            hr = 60 + random.uniform(0, 20)
            interval = 60.0 / hr
            # Add small variability
            interval *= (1 + random.uniform(-0.03, 0.03))

            t0 = time.perf_counter()
            while time.perf_counter() - t0 < interval:
                QtCore.QThread.msleep(50)
                if not self._running:
                    self.connected.emit(False)
                    return
                # Emit occasional raw data for recording
                if random.random() < 0.1:
                    self.raw_data.emit(random.randint(-300, 300))

            # Emit R-peak
            self.r_peak.emit()
            self.heart_rate.emit(round(60.0 / interval, 1))

        self.connected.emit(False)

    def stop(self):
        self._running = False

    def reset_detector(self):
        pass


# ============================================================
# Heart Widget - Animated Heart Icon
# ============================================================
class HeartWidget(QtWidgets.QWidget):
    """Draws an animated beating heart."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(160, 150)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self._scale = 1.0
        self._visible = False
        self._anim = QtCore.QVariantAnimation(self)
        self._anim.setDuration(400)
        self._anim.setStartValue(1.0)
        self._anim.setKeyValueAt(0.15, 1.25)
        self._anim.setKeyValueAt(0.35, 0.9)
        self._anim.setKeyValueAt(0.55, 1.1)
        self._anim.setEndValue(1.0)
        self._anim.valueChanged.connect(self._on_scale_changed)
        self._anim.finished.connect(lambda: self._on_anim_finished())
        self._heart_path = self._create_heart_path()

    def _create_heart_path(self):
        path = QtGui.QPainterPath()
        path.moveTo(80, 135)
        # Left curve
        path.cubicTo(-10, 80, 10, 5, 80, 25)
        # Right curve
        path.cubicTo(150, 5, 170, 80, 80, 135)
        return path

    def beat(self):
        if not self._visible:
            self._visible = True
            self.show()
        self._anim.stop()
        self._anim.start()

    def _on_scale_changed(self, val):
        self._scale = val
        self.update()

    def _on_anim_finished(self):
        self._scale = 1.0
        self.update()

    def hide_heart(self):
        self._anim.stop()
        self._visible = False
        self._scale = 1.0
        self.hide()
        self.update()

    def paintEvent(self, event):
        if not self._visible:
            return
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.translate(self.width() / 2, self.height() / 2)
        painter.scale(self._scale, self._scale)
        painter.translate(-self.width() / 2, -self.height() / 2)

        painter.setPen(QtGui.QPen(QtGui.QColor(220, 40, 40), 3))
        painter.setBrush(QtGui.QBrush(QtGui.QColor(240, 80, 80)))
        painter.drawPath(self._heart_path)


# ============================================================
# Experiment Controller (state machine)
# ============================================================
class ExperimentController(QtCore.QObject):
    """Manages the experiment state machine and trial logic."""

    # Signals for GUI updates
    sig_instruction = QtCore.pyqtSignal(str)
    sig_phase_info = QtCore.pyqtSignal(str)
    sig_trial_progress = QtCore.pyqtSignal(str)
    sig_feedback = QtCore.pyqtSignal(str, str)  # (type, text) type='correct'|'incorrect'|'timeout'
    sig_accuracy = QtCore.pyqtSignal(float)
    sig_heart_rate = QtCore.pyqtSignal(float)
    sig_heart_beat = QtCore.pyqtSignal()
    sig_show_rest = QtCore.pyqtSignal(int)  # rest seconds remaining
    sig_hide_rest = QtCore.pyqtSignal()
    sig_experiment_done = QtCore.pyqtSignal()
    sig_heartbeat_count = QtCore.pyqtSignal(int, int)  # (current, target)
    # Red signal for feedback phase response window
    sig_red_signal_on = QtCore.pyqtSignal()
    sig_red_signal_off = QtCore.pyqtSignal()
    # Wait-for-key signal: emitted when UI should show instruction and wait for key
    sig_wait_for_key = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._thread = None
        self._simulate = False

        # State
        self.state = 'idle'
        self._wait_callback = None  # called when key is pressed during wait_key state
        self.current_block = 0
        self.current_phase = ''  # 'feedback' or 'nofeedback'
        self.current_trial = 0
        self.current_target_n = 0
        self.trial_r_wave_count = 0
        self.trial_start_time = 0.0
        self.target_r_peak_time = 0.0
        self.pressed = False
        self.phase_correct = 0
        self.phase_total = 0
        self.block_phase_correct = 0
        self.block_phase_total = 0
        self.session_data_path = ''

        # Trial list
        self.trials = []  # list of dicts
        self.trial_idx = 0

        # Timers
        self._feedback_timer = QtCore.QTimer(self)
        self._feedback_timer.setSingleShot(True)
        self._feedback_timer.timeout.connect(self._on_feedback_done)

        self._timeout_timer = QtCore.QTimer(self)
        self._timeout_timer.setSingleShot(True)
        self._timeout_timer.timeout.connect(self._on_trial_timeout)

        # Max trial timer: prevents hanging if Nth R-wave never comes
        self._max_trial_timer = QtCore.QTimer(self)
        self._max_trial_timer.setSingleShot(True)
        self._max_trial_timer.timeout.connect(self._on_trial_timeout)

        self._rest_timer = QtCore.QTimer(self)
        self._rest_timer.timeout.connect(self._on_rest_tick)

        self._key_buffer = []  # (timestamp, key)
        self.participant_id = ""
        self.session = "1"

        # Recording
        self._r_peak_times = []
        self._trial_results = []
        self._ecg_file = None
        self._ecg_writer = None
        self._hr_log_file = None
        self._hr_log_writer = None
        self._hr_timestamps = deque()   # (timestamp, hr_bpm) for running avg
        self.heart_rate = 0.0
        self._hr_from_chip = False   # True when 0x03 chip HR has been received

    def set_participant_info(self, pid, session="1"):
        self.participant_id = pid
        self.session = session

    def _generate_trials(self):
        """Generate the full trial list for the session."""
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
                        'press_time': None,
                        'press_delay_ms': None,
                        'correct': None,
                        'response_time': None,
                    })
        self.trial_idx = 0

    def start_experiment(self, port=None, simulate=False):
        """Initialize and start the experiment."""
        self._simulate = simulate
        self._generate_trials()
        self._setup_data_recording()

        # Start worker thread
        self._start_worker(port, simulate)

        # Show waiting screen
        self.sig_instruction.emit("正在连接心电传感器...")
        self.state = 'starting'

        # Wait briefly for worker to connect, then start
        QtCore.QTimer.singleShot(2000, self._begin_experiment)

    def _begin_experiment(self):
        self.current_block = 1
        self.current_phase = 'feedback'
        self.current_trial = 0
        self.phase_correct = 0
        self.phase_total = 0
        self.trial_idx = 0
        self.state = 'ready'
        # 实验开始前显示说明，按任意键后开始
        self._show_instruction_wait(
            "实验说明\n\n"
            "这是一个内感受（interoception）视觉训练任务。\n"
            "您需要专注感受自己的心跳。\n\n"
            "• 屏幕上会提示「第 N 次心跳时按键」\n"
            "• 感受心跳，数到第 N 次时按下空格键\n"
            "• 反馈阶段会显示心脏图标和按键结果\n"
            "• 无反馈阶段则没有视觉提示\n\n"
            "准备好后，按任意键开始实验",
            self._start_block
        )

    def _show_instruction_wait(self, text, callback):
        """Show instruction and wait for any keypress before calling callback."""
        self._wait_callback = callback
        self.state = 'wait_key'
        self.sig_wait_for_key.emit(text)

    def _on_wait_key(self):
        """Called when user presses a key during wait_key state."""
        if self.state == 'wait_key' and self._wait_callback:
            cb = self._wait_callback
            self._wait_callback = None
            cb()

    def _setup_data_recording(self):
        # 保存在 ecg_intervention/recordings/ 目录下
        data_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'recordings'
        )
        os.makedirs(data_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        pid = self.participant_id or 'unknown'
        session = self.session or '1'
        session_dir = os.path.join(
            data_dir, f'{pid}_session{session}_{ts}'
        )
        os.makedirs(session_dir, exist_ok=True)
        self.session_data_path = session_dir

        # Open ECG raw data file
        ecg_path = os.path.join(session_dir, 'ecg_raw.csv')
        self._ecg_file = open(ecg_path, 'w', newline='')
        self._ecg_writer = csv.writer(self._ecg_file)
        self._ecg_writer.writerow(['timestamp', 'raw_value'])

        # ecg_log.csv — matches CardioChip ECGLog: timestamp, raw, HR, HR_4s_avg, HR_30s_avg
        ecg_log_path = os.path.join(session_dir, 'ecg_log.csv')
        self._hr_log_file = open(ecg_log_path, 'w', newline='')
        self._hr_log_writer = csv.writer(self._hr_log_file)
        self._hr_log_writer.writerow(['timestamp', 'raw_value', 'hr_bpm', 'hr_4s_avg', 'hr_30s_avg'])

    def _start_worker(self, port, simulate):
        self._thread = QtCore.QThread(self)
        if simulate or not HAS_SERIAL:
            self._worker = SimulatedWorker()
        else:
            self._worker = BMD101Worker(port)
        self._worker.moveToThread(self._thread)
        self._worker.r_peak.connect(self._on_r_peak)
        self._worker.raw_data.connect(self._on_raw_data)
        self._worker.heart_rate.connect(self._on_heart_rate)
        self._worker.error.connect(self._on_worker_error)
        self._thread.started.connect(self._worker.run)
        self._thread.start()

    def _on_heart_rate(self, hr):
        """Track HR data and forward to UI."""
        self.heart_rate = hr
        self._hr_from_chip = True  # mark that chip HR data is available
        self.sig_heart_rate.emit(hr)
        # Track for running average computation
        now = time.perf_counter()
        ts = self._exp_start_wall_time + (now - self._exp_start_time) if self._exp_start_wall_time > 0 else now
        self._hr_timestamps.append((ts, hr))
        while len(self._hr_timestamps) > 2 and ts - self._hr_timestamps[0][0] > 30:
            self._hr_timestamps.popleft()

    def _compute_hr_avg(self, window_sec):
        """Compute running average HR over the last N seconds."""
        if not self._hr_timestamps or self.heart_rate <= 0:
            return 0
        latest_ts = self._hr_timestamps[-1][0]
        recent = [hr for ts, hr in self._hr_timestamps if latest_ts - ts <= window_sec]
        return round(sum(recent) / len(recent), 1) if recent else 0

    def _on_raw_data(self, val):
        """Record raw ECG data and HR info to files."""
        now = time.perf_counter()
        ts = round(now, 4)
        if self._ecg_writer:
            self._ecg_writer.writerow([ts, val])
        if self._hr_log_writer:
            hr_4s = self._compute_hr_avg(4)
            hr_30s = self._compute_hr_avg(30)
            self._hr_log_writer.writerow([
                ts, val,
                round(self.heart_rate, 1) if self.heart_rate > 0 else '',
                hr_4s if hr_4s > 0 else '',
                hr_30s if hr_30s > 0 else '',
            ])

    def stop_experiment(self):
        self.state = 'idle'
        self._feedback_timer.stop()
        self._timeout_timer.stop()
        self._max_trial_timer.stop()
        self._rest_timer.stop()
        if self._worker:
            self._worker.stop()
        if self._thread:
            self._thread.quit()
            self._thread.wait(2000)
        if self._ecg_file:
            self._ecg_file.close()
            self._ecg_file = None
        if self._hr_log_file:
            self._hr_log_file.close()
            self._hr_log_file = None
        self._save_all_data()

    def _on_worker_error(self, msg):
        self.sig_instruction.emit(f"错误: {msg}")

    def _on_r_peak(self):
        """Called when worker detects an R-wave."""
        now = time.perf_counter()
        self._r_peak_times.append(now)
        # RR-interval HR fallback: only when chip 0x03 HR is NOT available
        if not self._hr_from_chip and len(self._r_peak_times) >= 2:
            rr = now - self._r_peak_times[-2]
            if 0.24 <= rr <= 3.0:  # 20~250 BPM physiological range
                self.heart_rate = round(60.0 / rr, 1)
                self.sig_heart_rate.emit(self.heart_rate)
        self._on_r_peak_experiment(now)

    def _on_r_peak_experiment(self, now):
        """Process R-wave when in an active trial or inter-trial hold."""
        if self.state == 'trial_active':
            self.trial_r_wave_count += 1
            self.sig_heartbeat_count.emit(
                self.trial_r_wave_count, self.current_target_n
            )

            # Store R-wave time for current trial
            idx = self.trial_idx - 1
            if 0 <= idx < len(self.trials):
                self.trials[idx]['r_peak_times'].append(now)

            # Check if this is the target (Nth) R-wave
            if self.trial_r_wave_count == self.current_target_n:
                self.target_r_peak_time = now
                # Start timeout timer (response window end + extra wait)
                timeout_ms = RESP_WIN_END + int(TIMEOUT_AFTER_TARGET_S * 1000)
                self._timeout_timer.start(timeout_ms)
                # Schedule red signal: appears immediately after target R-wave in feedback phase
                if self.current_phase == 'feedback':
                    QtCore.QTimer.singleShot(
                        HEART_APPEAR_DELAY_MS, self._activate_red_signal
                    )

            # In feedback phase: trigger heart icon 200ms after R-wave
            if self.current_phase == 'feedback':
                QtCore.QTimer.singleShot(
                    HEART_APPEAR_DELAY_MS, self.sig_heart_beat.emit
                )

    def _activate_red_signal(self):
        """Turn on the red signal indicator during the feedback response window."""
        if self.state == 'trial_active' and self.current_phase == 'feedback':
            self.sig_red_signal_on.emit()

    def _deactivate_red_signal(self):
        """Turn off the red signal indicator."""
        self.sig_red_signal_off.emit()

    def _start_block(self):
        if self.current_block > BLOCKS_PER_SESSION:
            self._finish_experiment()
            return

        self.current_phase = 'feedback'
        self.current_trial = 0
        self.phase_correct = 0
        self.phase_total = 0

        self.sig_phase_info.emit(
            f"组块 {self.current_block}/{BLOCKS_PER_SESSION} — 反馈阶段"
        )
        self.state = 'block_active'
        self._start_phase()

    def _start_phase(self):
        self.current_trial = 0
        self.phase_correct = 0
        self.phase_total = 0
        self.sig_accuracy.emit(0.0)

        if self.current_phase == 'feedback':
            self.sig_phase_info.emit(
                f"组块 {self.current_block}/{BLOCKS_PER_SESSION} — 反馈阶段"
            )
        else:
            self.sig_phase_info.emit(
                f"组块 {self.current_block}/{BLOCKS_PER_SESSION} — 无反馈阶段"
            )

        self.state = 'phase_active'
        self._start_trial()

    def _start_trial(self):
        if self.trial_idx >= len(self.trials):
            self._finish_phase()
            return

        trial = self.trials[self.trial_idx]
        if trial['phase'] != self.current_phase:
            self._finish_phase()
            return

        self.trial_idx += 1
        self.current_trial += 1
        self.current_target_n = trial['target_n']
        self.trial_r_wave_count = 0
        self.target_r_peak_time = 0.0
        self.pressed = False
        self._key_buffer.clear()

        self.sig_trial_progress.emit(
            f"试次 {self.current_trial}/{TRIALS_PER_PHASE}"
        )
        self.sig_instruction.emit(
            f"请在 <b>第 {self.current_target_n} 次</b> 心跳时按键"
        )
        self.sig_feedback.emit('', '')
        self.sig_heartbeat_count.emit(0, self.current_target_n)

        self.trial_start_time = time.perf_counter()
        self.state = 'trial_active'

        # Max trial duration safety: 15s regardless of R-waves
        self._max_trial_timer.start(15000)

    def handle_keypress(self):
        """Called when participant presses the space bar."""
        if self.state != 'trial_active' or self.pressed:
            return
        self.pressed = True
        press_time = time.perf_counter()

        self._timeout_timer.stop()
        self._max_trial_timer.stop()

        idx = self.trial_idx - 1
        if 0 <= idx < len(self.trials):
            self.trials[idx]['press_time'] = press_time

        # Determine correctness
        correct = False
        delay_ms = None
        if self.target_r_peak_time > 0:
            delay_ms = (press_time - self.target_r_peak_time) * 1000
            self.trials[idx]['press_delay_ms'] = delay_ms
            self.trials[idx]['response_time'] = press_time - self.trial_start_time
            if RESP_WIN_START <= delay_ms <= RESP_WIN_END:
                correct = True

        if correct:
            self.phase_correct += 1
            self.sig_feedback.emit('correct', '正确！')
        else:
            self.sig_feedback.emit('incorrect', '错误')

        self.phase_total += 1
        accuracy = (self.phase_correct / self.phase_total) * 100
        self.trials[idx]['correct'] = correct
        self.sig_accuracy.emit(round(accuracy, 1))
        self._deactivate_red_signal()

        # Show feedback for 1.5s then next trial
        self.state = 'trial_feedback'
        self._feedback_timer.start(1500)

    def _on_feedback_done(self):
        self._max_trial_timer.stop()
        self._start_trial()

    def _on_trial_timeout(self):
        """No keypress received in time."""
        if self.state != 'trial_active' or self.pressed:
            return
        self.pressed = True
        self._max_trial_timer.stop()
        self.phase_total += 1
        accuracy = (self.phase_correct / self.phase_total) * 100
        self.sig_accuracy.emit(round(accuracy, 1))

        idx = self.trial_idx - 1
        if 0 <= idx < len(self.trials):
            self.trials[idx]['correct'] = False
            self.trials[idx]['press_time'] = None
            self.trials[idx]['press_delay_ms'] = None

        self._deactivate_red_signal()
        self.sig_feedback.emit('timeout', '超时')
        self.state = 'trial_feedback'
        self._feedback_timer.start(1500)

    def _finish_phase(self):
        self._deactivate_red_signal()
        accuracy = 0
        if self.phase_total > 0:
            accuracy = (self.phase_correct / self.phase_total) * 100
        self.sig_accuracy.emit(round(accuracy, 1))
        self.state = 'phase_summary'

        if self.current_phase == 'feedback':
            # Switch to no-feedback phase — show wait screen first
            self.current_phase = 'nofeedback'
            self.sig_phase_info.emit(
                f"组块 {self.current_block}/{BLOCKS_PER_SESSION} — 无反馈阶段"
            )
            self._show_instruction_wait(
                f"反馈阶段结束！正确率: <b>{accuracy:.1f}%</b>\n\n"
                "即将进入<b>无反馈阶段</b>。\n"
                "在该阶段，您将不会看到心脏图标和正确/错误提示。\n"
                "请继续保持专注，感受心跳时按键。\n\n"
                "准备好后，按任意键继续",
                self._start_phase
            )
        else:
            # Block complete, start rest or next block
            if self.current_block < BLOCKS_PER_SESSION:
                self._show_instruction_wait(
                    f"组块 {self.current_block} 无反馈阶段结束！正确率: <b>{accuracy:.1f}%</b>\n\n"
                    "即将进入休息时间（5分钟）。\n\n"
                    "准备好后，按任意键开始休息",
                    self._start_rest
                )
            else:
                self._show_instruction_wait(
                    f"所有组块完成！最终正确率: <b>{accuracy:.1f}%</b>\n\n"
                    "实验即将结束。\n\n"
                    "按任意键查看结果",
                    self._finish_experiment
                )

    def _start_rest(self):
        self.state = 'rest'
        self.rest_remaining = REST_DURATION_S
        self.sig_show_rest.emit(self.rest_remaining)
        self._rest_timer.start(1000)

    def _on_rest_tick(self):
        self.rest_remaining -= 1
        self.sig_show_rest.emit(self.rest_remaining)
        if self.rest_remaining <= 0:
            self._rest_timer.stop()
            self.sig_hide_rest.emit()
            self.current_block += 1
            self._show_instruction_wait(
                f"休息结束！\n\n"
                f"即将进入<b>组块 {self.current_block}/{BLOCKS_PER_SESSION}</b>\n"
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
        lines.append(f"被试编号: {self.participant_id}    第 {self.session} 次训练")
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

    def _finish_experiment(self):
        self.state = 'complete'
        self._results_text = self._build_results_text()
        self.sig_feedback.emit('', '')
        self._save_all_data()
        self.sig_experiment_done.emit()

    def _save_all_data(self):
        if not self.session_data_path:
            return

        # Save trial results
        trial_file = os.path.join(self.session_data_path, 'trials.csv')
        with open(trial_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'trial_idx', 'block', 'phase', 'trial_num',
                'target_n', 'correct', 'press_delay_ms', 'response_time_s'
            ])
            for i, t in enumerate(self.trials, 1):
                writer.writerow([
                    i, t['block'], t['phase'], t['trial'],
                    t['target_n'], t['correct'],
                    round(t['press_delay_ms'], 2) if t['press_delay_ms'] is not None else '',
                    round(t['response_time'], 3) if t['response_time'] is not None else ''
                ])

        # Save R-peak timestamps
        rpeak_file = os.path.join(self.session_data_path, 'r_peaks.csv')
        with open(rpeak_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp'])
            for t in self._r_peak_times:
                writer.writerow([round(t, 4)])

        # Save summary
        total_correct = sum(1 for t in self.trials if t['correct'])
        total_trials = len(self.trials)
        summary = {
            'participant_id': self.participant_id,
            'session': self.session,
            'timestamp': datetime.now().isoformat(),
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

        print(f"数据已保存至: {self.session_data_path}")


# ============================================================
# Serial Port Selection Dialog
# ============================================================
class PortDialog(QtWidgets.QDialog):
    """Dialog for participant info and serial port selection."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ECG 生物反馈训练 — 实验设置")
        self.setFixedSize(420, 320)
        # Explicit styling to ensure placeholder text is visible (PsychoPy dark theme may hide it)
        self.setStyleSheet("""
            QLineEdit {
                color: #ffffff;
                background-color: #3a3a50;
                border: 1px solid #666;
                border-radius: 4px;
                padding: 6px 10px;
                font-size: 16px;
            }
            QLabel {
                color: #ffffff;
                font-size: 16px;
            }
        """)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(12)

        title = QtWidgets.QLabel("<h2>内感受视觉训练</h2>")
        title.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title)

        # Participant ID
        id_layout = QtWidgets.QHBoxLayout()
        id_layout.addWidget(QtWidgets.QLabel("被试编号:"))
        self.id_input = QtWidgets.QLineEdit()
        self.id_input.setPlaceholderText("例如: S001")
        # Ensure placeholder text is visible (PsychoPy dark theme may hide it)
        _p = self.id_input.palette()
        _p.setColor(QtGui.QPalette.PlaceholderText, QtGui.QColor(170, 170, 200))
        self.id_input.setPalette(_p)
        id_layout.addWidget(self.id_input)
        layout.addLayout(id_layout)

        # Session number
        sess_layout = QtWidgets.QHBoxLayout()
        sess_layout.addWidget(QtWidgets.QLabel("训练次数:"))
        self.session_input = QtWidgets.QLineEdit()
        self.session_input.setPlaceholderText("1, 2, 3...")
        _p2 = self.session_input.palette()
        _p2.setColor(QtGui.QPalette.PlaceholderText, QtGui.QColor(170, 170, 200))
        self.session_input.setPalette(_p2)
        sess_layout.addWidget(self.session_input)
        layout.addLayout(sess_layout)

        # Separator
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        layout.addWidget(line)

        # Simulate checkbox (create before _scan_ports to avoid AttributeError)
        self.sim_check = QtWidgets.QCheckBox("模拟模式（无硬件时测试用）")
        layout.addWidget(self.sim_check)

        # Port selection
        port_layout = QtWidgets.QHBoxLayout()
        port_layout.addWidget(QtWidgets.QLabel("串口:"))
        self.port_combo = QtWidgets.QComboBox()
        self._scan_ports()
        port_layout.addWidget(self.port_combo)
        btn_scan = QtWidgets.QPushButton("刷新")
        btn_scan.clicked.connect(self._scan_ports)
        port_layout.addWidget(btn_scan)
        layout.addLayout(port_layout)

        layout.addStretch()

        # Start button
        btn_start = QtWidgets.QPushButton("开始实验")
        btn_start.setMinimumHeight(40)
        btn_start.clicked.connect(self.accept)
        layout.addWidget(btn_start)

    def _scan_ports(self):
        self.port_combo.clear()
        if HAS_SERIAL:
            import serial.tools.list_ports
            ports = serial.tools.list_ports.comports()
            for p in ports:
                self.port_combo.addItem(f"{p.device} - {p.description}", p.device)
        if self.port_combo.count() == 0:
            self.port_combo.addItem("未检测到串口", "")
            self.sim_check.setChecked(True)

    def get_config(self):
        pid = self.id_input.text().strip()
        if not pid:
            pid = "unknown"
        session = self.session_input.text().strip()
        if not session:
            session = "1"
        simulate = self.sim_check.isChecked()
        port = self.port_combo.currentData() if not simulate else None
        return pid, session, port, simulate


# ============================================================
# Red Signal Overlay — glowing red border for feedback phase
# ============================================================
class RedSignalOverlay(QtWidgets.QWidget):
    """A transparent overlay that draws a pulsing red border glow."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self._active = False
        self._opacity = 0.0

        # Pulse animation
        self._pulse = QtCore.QVariantAnimation(self)
        self._pulse.setDuration(800)
        self._pulse.setStartValue(0.3)
        self._pulse.setKeyValueAt(0.5, 1.0)
        self._pulse.setEndValue(0.3)
        self._pulse.valueChanged.connect(self._on_pulse)
        self._pulse.setLoopCount(-1)  # infinite loop

    def activate(self):
        self._active = True
        self._opacity = 1.0
        self._pulse.start()
        self.show()
        self.raise_()
        self.update()

    def deactivate(self):
        self._active = False
        self._pulse.stop()
        self._opacity = 0.0
        self.hide()
        self.update()

    def _on_pulse(self, val):
        self._opacity = val
        self.update()

    def paintEvent(self, event):
        if not self._active:
            return
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        # Glowing red border: draw multiple layers with decreasing opacity
        pen_widths = [30, 20, 10]
        alphas = [int(self._opacity * 40), int(self._opacity * 80), int(self._opacity * 180)]
        for w, a in zip(pen_widths, alphas):
            pen = QtGui.QPen(QtGui.QColor(255, 0, 0, a), w)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.NoBrush)
            rect = self.rect().adjusted(w // 2, w // 2, -w // 2, -w // 2)
            painter.drawRoundedRect(rect, 8, 8)

    def resizeEvent(self, event):
        self.update()


# ============================================================
# Main Window
# ============================================================
class MainWindow(QtWidgets.QMainWindow):
    """Main experiment GUI window."""

    def __init__(self):
        super().__init__()
        self.controller = ExperimentController(self)
        self._red_overlay = None  # created in _setup_ui
        self._setup_ui()
        self._connect_signals()
        self._set_fullscreen()

    def _set_fullscreen(self):
        screen = QtWidgets.QApplication.primaryScreen()
        if screen:
            self.setGeometry(0, 0, screen.size().width(), screen.size().height())

    def _setup_ui(self):
        self.setWindowTitle("ECG 生物反馈训练")
        self.setStyleSheet("""
            QMainWindow { background-color: #1a1a2e; }
            QLabel { color: #ffffff; font-family: 'Microsoft YaHei', 'SimHei'; }
        """)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(50, 30, 50, 30)
        layout.setSpacing(20)

        # === Top bar: phase info + HR ===
        top_layout = QtWidgets.QHBoxLayout()

        self.phase_label = QtWidgets.QLabel("准备开始")
        self.phase_label.setStyleSheet("font-size: 22px; padding: 5px;")

        hr_layout = QtWidgets.QHBoxLayout()
        hr_layout.addStretch()
        hr_label = QtWidgets.QLabel("心率:")
        hr_label.setStyleSheet("font-size: 20px;")
        self.hr_value = QtWidgets.QLabel("-- BPM")
        self.hr_value.setStyleSheet("font-size: 24px; font-weight: bold; color: #ff6b6b;")
        hr_layout.addWidget(hr_label)
        hr_layout.addWidget(self.hr_value)

        top_layout.addWidget(self.phase_label)
        top_layout.addLayout(hr_layout)
        layout.addLayout(top_layout)

        # === Center area ===
        layout.addStretch(1)

        # Accuracy bar
        acc_layout = QtWidgets.QHBoxLayout()
        acc_layout.addStretch()
        self.acc_label = QtWidgets.QLabel("正确率: 0%")
        self.acc_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #4ecdc4;")
        acc_layout.addWidget(self.acc_label)
        acc_layout.addStretch()
        layout.addLayout(acc_layout)

        layout.addSpacing(10)

        # Instruction
        self.instruction_label = QtWidgets.QLabel("按下「开始」启动实验")
        self.instruction_label.setAlignment(QtCore.Qt.AlignCenter)
        self.instruction_label.setStyleSheet("""
            font-size: 44px; font-weight: bold;
            color: #ffffff; padding: 20px;
            background-color: rgba(255,255,255,0.08);
            border-radius: 15px;
        """)
        self.instruction_label.setMinimumHeight(100)
        self.instruction_label.setWordWrap(True)
        layout.addWidget(self.instruction_label)

        layout.addSpacing(10)

        # Heart icon (centered)
        heart_layout = QtWidgets.QHBoxLayout()
        heart_layout.addStretch()
        self.heart_widget = HeartWidget()
        self.heart_widget.hide_heart()
        heart_layout.addWidget(self.heart_widget)
        heart_layout.addStretch()
        layout.addLayout(heart_layout)

        layout.addSpacing(10)

        # Feedback
        self.feedback_label = QtWidgets.QLabel("")
        self.feedback_label.setAlignment(QtCore.Qt.AlignCenter)
        self.feedback_label.setStyleSheet("font-size: 80px; font-weight: bold;")
        layout.addWidget(self.feedback_label)

        # Heartbeat count
        self.count_label = QtWidgets.QLabel("")
        self.count_label.setAlignment(QtCore.Qt.AlignCenter)
        self.count_label.setStyleSheet("font-size: 34px; color: #a0a0c0;")
        layout.addWidget(self.count_label)

        layout.addStretch(2)

        # === Bottom status bar ===
        status_layout = QtWidgets.QHBoxLayout()

        self.trial_progress_label = QtWidgets.QLabel("")
        self.trial_progress_label.setStyleSheet("font-size: 20px; color: #a0a0c0;")

        self.status_label = QtWidgets.QLabel("就绪 — 按 Space 开始")
        self.status_label.setStyleSheet("font-size: 20px; color: #a0a0c0;")
        self.status_label.setAlignment(QtCore.Qt.AlignRight)

        status_layout.addWidget(self.trial_progress_label)
        status_layout.addStretch()
        status_layout.addWidget(self.status_label)
        layout.addLayout(status_layout)

        # === Red signal overlay (covers central area) ===
        self._red_overlay = RedSignalOverlay(central)
        self._red_overlay.setGeometry(central.rect())
        self._red_overlay.hide()

    def _connect_signals(self):
        c = self.controller
        c.sig_instruction.connect(self._set_instruction_text)

    def _set_instruction_text(self, text):
        """Set instruction text and reset style from wait-screen mode when a trial is active."""
        self.instruction_label.setText(text)
        if self.controller.state != 'wait_key':
            self.instruction_label.setStyleSheet("""
                font-size: 44px; font-weight: bold;
                color: #ffffff; padding: 20px;
                background-color: rgba(255,255,255,0.08);
                border-radius: 15px;
            """)
        c.sig_phase_info.connect(self.phase_label.setText)
        c.sig_trial_progress.connect(self.trial_progress_label.setText)
        c.sig_accuracy.connect(self._update_accuracy)
        c.sig_heart_rate.connect(lambda hr: self.hr_value.setText(f"{hr:.0f} BPM"))
        c.sig_heart_beat.connect(self.heart_widget.beat)
        c.sig_heartbeat_count.connect(self._update_heartbeat_count)
        c.sig_show_rest.connect(self._show_rest_screen)
        c.sig_hide_rest.connect(self._hide_rest)
        c.sig_experiment_done.connect(self._on_experiment_done)
        c.sig_feedback.connect(self._show_feedback)
        c.sig_red_signal_on.connect(self._on_red_signal_on)
        c.sig_red_signal_off.connect(self._on_red_signal_off)
        c.sig_wait_for_key.connect(self._show_wait_screen)

    def _update_accuracy(self, pct):
        self.acc_label.setText(f"正确率: {pct:.1f}%")

    def _update_heartbeat_count(self, current, target):
        self.count_label.setText(f"心跳计数: {current}/{target}")


    def _on_red_signal_on(self):
        """Show red glowing border during feedback response window."""
        if self._red_overlay:
            self._red_overlay.activate()

    def _on_red_signal_off(self):
        """Hide red glowing border."""
        if self._red_overlay:
            self._red_overlay.deactivate()

    def _show_wait_screen(self, text):
        """Display instruction text and wait for any keypress."""
        self.instruction_label.setText(text)
        self.instruction_label.setStyleSheet("""
            font-size: 34px; font-weight: bold;
            color: #ffffff; padding: 30px;
            background-color: rgba(255,255,255,0.08);
            border: 2px solid #4ecdc4;
            border-radius: 15px;
        """)
        self.feedback_label.setText("")
        self.heart_widget.hide_heart()
        self.status_label.setText("等待按键 — 按任意键继续")

    def _show_feedback(self, ftype, text):
        if ftype == 'correct':
            self.feedback_label.setText("✓")
            self.feedback_label.setStyleSheet(
                "font-size: 80px; font-weight: bold; color: #2ecc71;"
            )
        elif ftype == 'incorrect':
            self.feedback_label.setText("✗")
            self.feedback_label.setStyleSheet(
                "font-size: 80px; font-weight: bold; color: #e74c3c;"
            )
        elif ftype == 'timeout':
            self.feedback_label.setText("—")
            self.feedback_label.setStyleSheet(
                "font-size: 80px; font-weight: bold; color: #f39c12;"
            )
        else:
            self.feedback_label.setText("")

    def _show_rest_screen(self, remaining):
        mins = remaining // 60
        secs = remaining % 60
        self.instruction_label.setText(
            f"休息时间<br>"
            f"<span style='font-size:56px;'>{mins:02d}:{secs:02d}</span>"
        )
        self.feedback_label.setText("")
        self.heart_widget.hide_heart()

    def _hide_rest(self):
        self.instruction_label.setText("准备继续...")

    def _on_experiment_done(self):
        results = getattr(self.controller, '_results_text', '实验完成！')
        self.instruction_label.setText(results)
        self.instruction_label.setStyleSheet("""
            font-size: 22px; font-weight: normal;
            color: #ffffff; padding: 20px 30px;
            background-color: rgba(255,255,255,0.08);
            border-radius: 15px;
            font-family: 'Microsoft YaHei', 'SimHei', monospace;
        """)
        self.instruction_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.status_label.setText("实验完成 — 按任意键退出")
        self.feedback_label.setStyleSheet("font-size: 80px;")
        self.feedback_label.setText("")

    def keyPressEvent(self, event):
        # Complete state — any key closes the results window (no confirmation)
        if self.controller.state == 'complete':
            self.close()
            event.accept()
            return

        # Any key during wait_key state continues experiment (except Escape)
        if self.controller.state == 'wait_key':
            if event.key() == QtCore.Qt.Key_Escape:
                # Allow Escape to exit even during wait screen
                reply = QtWidgets.QMessageBox.question(
                    self, '确认退出',
                    '实验尚未开始，确定要退出吗？',
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
                )
                if reply == QtWidgets.QMessageBox.Yes:
                    self.controller.stop_experiment()
                    self.close()
                event.accept()
                return
            self.controller._on_wait_key()
            event.accept()
            return

        if event.key() == QtCore.Qt.Key_Space:
            if self.controller.state == 'trial_active' and not self.controller.pressed:
                self.controller.handle_keypress()
            event.accept()
        elif event.key() == QtCore.Qt.Key_Escape:
            if self.controller.state != 'complete':
                reply = QtWidgets.QMessageBox.question(
                    self, '确认退出',
                    '实验正在进行中，确定要退出吗？\n（已保存的数据不会丢失）',
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
                )
                if reply == QtWidgets.QMessageBox.Yes:
                    self.controller.stop_experiment()
                    self.close()
            event.accept()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        if self.controller.state not in ('complete', 'idle'):
            self.controller.stop_experiment()
        event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._red_overlay:
            # Overlay covers the entire central widget
            central = self.centralWidget()
            if central:
                self._red_overlay.setGeometry(central.rect())

    def start_experiment(self, pid, session, port, simulate):
        self.controller.set_participant_info(pid, session)
        self.controller.start_experiment(port, simulate)
        self.setFocus()
        self.status_label.setText("运行中 — 数到指定心跳时按 Space")


# ============================================================
# Entry Point
# ============================================================
def main():
    # Ensure high-DPI support
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("ECG Biofeedback Training")

    # Show port selection dialog
    dialog = PortDialog()
    if dialog.exec_() != QtWidgets.QDialog.Accepted:
        return

    pid, session, port, simulate = dialog.get_config()

    window = MainWindow()
    window.show()
    window.start_experiment(pid, session, port, simulate)

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
