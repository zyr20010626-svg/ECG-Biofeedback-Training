# ECG 生物反馈训练程序

内感受视觉训练干预方案 — 基于 BMD101 心电传感器

## 项目结构

```
ecg_intervention/
├── ecg_biofeedback_psychopy.py   # PsychoPy 原生版（正式实验用）
├── ecg_biofeedback_practice.py   # PsychoPy 练习版（演示用）
├── ecg_biofeedback_training.py   # PyQt5 版（旧版）
└── recordings/                   # 被试数据保存目录
```

## 实验流程

- 3 个训练组块，每个组块含反馈/无反馈两个阶段
- 每阶段 24 试次，共 144 试次
- 反馈阶段：R 波后呈现动态心脏图标 + 红色脉冲边框
- 按键窗口：指定 R 波后 0ms–500ms

## 运行方式

### PsychoPy 版（推荐）
在 PsychoPy Coder 中打开 `ecg_intervention/ecg_biofeedback_psychopy.py`，按 F5 运行。

### PyQt5 版
```bash
python ecg_intervention/ecg_biofeedback_training.py
```

## 最近更新

### 2026-06-17：心率显示平滑修复

**问题**：程序右上角的心率数值在真实硬件模式下跳脱严重（每次心跳都在变动）。

**原因**：
- 心率来源有两个：BMD101 芯片 `0x03` 包（稳定）和 RR 间期推算（单次心跳计算，受 HRV 影响大）
- 如果 BMD101 未发送 `0x03` 包，回退到 RR 间期计算 → 每次心跳显示值都变
- 显示直接用瞬时值，没有任何平滑

**修复**：
1. RR 间期推算的心率也加入 `_hr_timestamps` 滑动窗口队列
2. 显示改为用 **4 秒滑动平均**（`_compute_hr_avg(4)`）代替瞬时值
3. 模拟模式心率改为缓慢漂移（±2 BPM/次）而非完全随机（60-80 BPM 乱跳）
4. 三版程序（PsychoPy / 练习版 / PyQt5）同步修复

**修改文件**：
- `ecg_intervention/ecg_biofeedback_psychopy.py`
- `ecg_intervention/ecg_biofeedback_practice.py`
- `ecg_intervention/ecg_biofeedback_training.py`
