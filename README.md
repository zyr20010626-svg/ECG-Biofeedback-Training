# ECG 生物反馈训练程序

内感受视觉训练干预方案 — 基于 BMD101 心电传感器

## 项目结构

```
ecg_intervention/
├── ecg_biofeedback_psychopy.py   # PsychoPy 原生版（正式实验用）
├── ecg_biofeedback_practice.py   # PsychoPy 练习版（演示用）
├── ecg_biofeedback_training.py   # PyQt5 版
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
