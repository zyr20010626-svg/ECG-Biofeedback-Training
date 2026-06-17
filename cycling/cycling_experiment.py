#!/usr/bin/env python
# -*- coding: utf-8 -*-

from psychopy import visual, core, event, gui
import os
import time
import csv
from PIL import Image

# ========== 全局设置 ==========
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

exp_info = {
    'participant': '',
    'gender': ['男', '女'],
    'session': '001'
}
dlg = gui.DlgFromDict(exp_info, title='骑行负荷主观评定实验', order=['participant', 'gender', 'session'])
if not dlg.OK:
    core.quit()

# ========== 全屏模式 ==========
win = visual.Window(fullscr=True, color=[0.3, 0.3, 0.3], colorSpace='rgb', units='pix')
win.mouseVisible = True

results = []
font_name = 'Microsoft YaHei'

# ========== 图片路径 ==========
SAM_PATH = os.path.join(_thisDir, 'sam_images')
RPE_PATH = os.path.join(_thisDir, 'RPE_images')

VALENCE_IMG = os.path.join(SAM_PATH, 'sam_happy.png')
AROUSAL_IMG = os.path.join(SAM_PATH, 'sam_arousal.png')
RPE_6_20_IMG = os.path.join(RPE_PATH, 'RPE6_20.png')
CR_10_IMG = os.path.join(RPE_PATH, 'cr10.png')

# ========== 通用组件 ==========
welcome_text = visual.TextStim(win, text='', font=font_name, height=30, color='white', wrapWidth=1000, alignText='center')
press_key_text = visual.TextStim(win, text='按空格键继续', font=font_name, height=25, pos=(0, -380), color='lightgray')

timer_text = visual.TextStim(win, text='00:00', font=font_name, height=50, pos=(0, 320), color='green', bold=True)
pause_text = visual.TextStim(win, text='【已暂停】按空格键继续', font=font_name, height=35, pos=(0, 0), color='yellow', bold=True)

# 红色提醒文字
reminder_text = visual.TextStim(
    win, text='请不要停止蹬车', font=font_name, height=28,
    pos=(0, -340), color='red', bold=True
)


# ========== 安全退出 ==========

def safe_quit(save=True):
    if save and len(results) > 0:
        try:
            save_data()
        except Exception as e:
            try:
                if not os.path.exists('data'):
                    os.makedirs('data')
                backup_file = f"data/{exp_info['participant']}-{exp_info['session']}_backup_{int(time.time())}.csv"
                with open(backup_file, 'w', newline='', encoding='utf-8-sig') as f:
                    fieldnames = ['participant', 'gender', 'session', 'block', 'scale_type', 'StartTime', 'EndTime', 'timestamp']
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(results)
            except:
                pass
    win.close()
    core.quit()


def save_data():
    if not os.path.exists('data'):
        os.makedirs('data')
    
    filename = f"data/{exp_info['participant']}-{exp_info['session']}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    
    fieldnames = ['participant', 'gender', 'session', 'block', 'scale_type', 'StartTime', 'EndTime', 'timestamp']
    with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"数据已保存: {filename}，共{len(results)}条")
    return filename


# ========== 图片加载函数（PIL缩放后保存临时文件） ==========

def load_image_resized(image_path, target_height=600, max_width=1000):
    """
    用PIL加载图片，等比例缩放，保存为临时文件，返回临时文件路径和尺寸
    """
    if not os.path.exists(image_path):
        print(f"  图片不存在: {image_path}")
        return None, 0, 0
    
    try:
        img = Image.open(image_path)
        orig_w, orig_h = img.size
        print(f"  PIL原始尺寸: {orig_w}x{orig_h}")
        
        # 等比例缩放
        scale = target_height / orig_h
        new_w = int(orig_w * scale)
        new_h = target_height
        
        if new_w > max_width:
            new_w = max_width
            scale = new_w / orig_w
            new_h = int(orig_h * scale)
        
        # PIL缩放
        img_resized = img.resize((new_w, new_h), Image.LANCZOS)
        
        # 保存临时文件
        temp_filename = f"temp_{os.path.basename(image_path)}"
        temp_path = os.path.join(_thisDir, temp_filename)
        img_resized.save(temp_path)
        
        print(f"  缩放后: {new_w}x{new_h}, 临时文件: {temp_path}")
        return temp_path, new_w, new_h
        
    except Exception as e:
        print(f"  图片处理失败: {e}")
        return None, 0, 0


# ========== 通用量表展示类（RPE/CR-10用，等比例放大） ==========

class ScaleDisplay:
    def __init__(self, win, image_path, title_text, block_name, scale_type, target_height=600, img_pos=(0, -20)):
        self.win = win
        self.block_name = block_name
        self.scale_type = scale_type
        
        self.title = visual.TextStim(
            win, text=title_text, font=font_name, height=32,
            pos=(0, 360), color='white', bold=True
        )
        
        # 用PIL缩放并保存临时文件
        temp_path, new_w, new_h = load_image_resized(image_path, target_height=target_height)
        
        if temp_path is not None and os.path.exists(temp_path):
            self.scale_img = visual.ImageStim(
                win, image=temp_path, pos=img_pos
            )
            print(f"  ImageStim创建成功: {new_w}x{new_h}")
        else:
            print(f"图片加载失败: {image_path}")
            self.scale_img = visual.TextStim(
                win, text=f'[图片加载失败]\n{os.path.basename(image_path)}',
                font=font_name, height=25, pos=img_pos, color='red'
            )
        
        self.hint = visual.TextStim(
            win, text='主试记录完成后，按空格键继续',
            font=font_name, height=22, pos=(0, -380), color='lightgray'
        )
    
    def show(self):
        start_time = time.time()
        print(f"[{self.block_name}] {self.scale_type} 开始: {start_time}")
        
        while True:
            self.title.draw()
            self.scale_img.draw()
            reminder_text.draw()
            self.hint.draw()
            win.flip()
            
            keys = event.getKeys(keyList=['space', 'escape'])
            if 'escape' in keys:
                end_time = time.time()
                results.append({
                    'participant': exp_info['participant'],
                    'gender': exp_info['gender'],
                    'session': exp_info['session'],
                    'block': self.block_name,
                    'scale_type': self.scale_type,
                    'StartTime': round(start_time, 3),
                    'EndTime': round(end_time, 3),
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                })
                safe_quit(save=True)
            if 'space' in keys:
                break
        
        end_time = time.time()
        print(f"[{self.block_name}] {self.scale_type} 结束: {end_time}")
        
        results.append({
            'participant': exp_info['participant'],
            'gender': exp_info['gender'],
            'session': exp_info['session'],
            'block': self.block_name,
            'scale_type': self.scale_type,
            'StartTime': round(start_time, 3),
            'EndTime': round(end_time, 3),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })


# ========== 合并SAM展示（PIL统一缩放到相同像素尺寸） ==========

class CombinedSAMDisplay:
    def __init__(self, win, valence_img, arousal_img, block_name):
        self.win = win
        self.block_name = block_name
        
        self.title = visual.TextStim(
            win, text='情绪状态评定', font=font_name, height=32,
            pos=(0, 370), color='white', bold=True
        )
        
        # SAM统一尺寸：850×150 像素（强制两个图片完全相同）
        unified_w, unified_h = 850, 150
        
        # 效价：PIL缩放到统一尺寸
        if os.path.exists(valence_img):
            try:
                img = Image.open(valence_img)
                img_resized = img.resize((unified_w, unified_h), Image.LANCZOS)
                temp_v = os.path.join(_thisDir, 'temp_sam_valence.png')
                img_resized.save(temp_v)
                self.valence_img = visual.ImageStim(win, image=temp_v, pos=(0, 140))
                print(f"  效价统一尺寸: {unified_w}x{unified_h}")
            except Exception as e:
                print(f"  效价图片处理失败: {e}")
                self.valence_img = visual.TextStim(win, text='[效价图片错误]', font=font_name, height=25, pos=(0, 140), color='red')
        else:
            self.valence_img = visual.TextStim(win, text='[效价图片缺失]', font=font_name, height=25, pos=(0, 140), color='red')
        
        # 唤醒度：PIL缩放到统一尺寸
        if os.path.exists(arousal_img):
            try:
                img = Image.open(arousal_img)
                img_resized = img.resize((unified_w, unified_h), Image.LANCZOS)
                temp_a = os.path.join(_thisDir, 'temp_sam_arousal.png')
                img_resized.save(temp_a)
                self.arousal_img = visual.ImageStim(win, image=temp_a, pos=(0, -90))
                print(f"  唤醒度统一尺寸: {unified_w}x{unified_h}")
            except Exception as e:
                print(f"  唤醒度图片处理失败: {e}")
                self.arousal_img = visual.TextStim(win, text='[唤醒度图片错误]', font=font_name, height=25, pos=(0, -90), color='red')
        else:
            self.arousal_img = visual.TextStim(win, text='[唤醒度图片缺失]', font=font_name, height=25, pos=(0, -90), color='red')
        
        self.valence_label = visual.TextStim(
            win, text='效价（不开心 → 开心）', font=font_name, height=22,
            pos=(0, 250), color='white'
        )
        self.arousal_label = visual.TextStim(
            win, text='唤醒度（平静 → 兴奋）', font=font_name, height=22,
            pos=(0, 20), color='white'
        )
        
        self.hint = visual.TextStim(
            win, text='主试记录完成后，按空格键继续',
            font=font_name, height=22, pos=(0, -380), color='lightgray'
        )
    
    def show(self):
        start_time = time.time()
        print(f"[{self.block_name}] SAM合并展示 开始: {start_time}")
        
        while True:
            self.title.draw()
            self.valence_label.draw()
            self.valence_img.draw()
            self.arousal_label.draw()
            self.arousal_img.draw()
            reminder_text.draw()
            self.hint.draw()
            win.flip()
            
            keys = event.getKeys(keyList=['space', 'escape'])
            if 'escape' in keys:
                end_time = time.time()
                results.append({
                    'participant': exp_info['participant'],
                    'gender': exp_info['gender'],
                    'session': exp_info['session'],
                    'block': self.block_name,
                    'scale_type': 'SAM_combined',
                    'StartTime': round(start_time, 3),
                    'EndTime': round(end_time, 3),
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                })
                safe_quit(save=True)
            if 'space' in keys:
                break
        
        end_time = time.time()
        print(f"[{self.block_name}] SAM合并展示 结束: {end_time}")
        
        results.append({
            'participant': exp_info['participant'],
            'gender': exp_info['gender'],
            'session': exp_info['session'],
            'block': self.block_name,
            'scale_type': 'SAM_combined',
            'StartTime': round(start_time, 3),
            'EndTime': round(end_time, 3),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })


# ========== 辅助函数 ==========

def format_time(seconds):
    mins = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{mins:02d}:{secs:02d}"

def show_message(text, wait_keys=True):
    welcome_text.text = text
    welcome_text.draw()
    if wait_keys:
        press_key_text.draw()
    win.flip()
    if wait_keys:
        event.waitKeys(keyList=['space', 'escape'])

def run_questionnaire(block_name):
    """运行完整量表展示：SAM合并 → RPE6-20 → CR-10"""
    print(f"\n=== 开始量表: {block_name} ===")
    
    # SAM合并展示（统一尺寸）
    sam = CombinedSAMDisplay(win, VALENCE_IMG, AROUSAL_IMG, block_name)
    sam.show()
    
    # RPE 6-20（等比例放大，位置上移）
    rpe_6_20 = ScaleDisplay(win, RPE_6_20_IMG,
        '主观疲劳感觉评估（RPE 6-20）',
        block_name, 'fatigue_RPE_6_20',
        target_height=600, img_pos=(0, 10))
    rpe_6_20.show()
    
    # CR-10（等比例放大，位置上移）
    cr_10 = ScaleDisplay(win, CR_10_IMG,
        '主观疲劳感觉评估（CR-10）',
        block_name, 'fatigue_CR_10',
        target_height=600, img_pos=(0, 10))
    cr_10.show()
    
    print(f"=== 量表完成: {block_name} ===")

def run_cycling(block_name, duration=180, allow_skip=False):
    # D环节：女性直接结束，男性直接进入计时
    if allow_skip:
        if exp_info['gender'] == '女':
            print("女性被试，跳过D环节，直接结束实验")
            return False, 0
        else:
            print("男性被试，直接进入D骑行")
    
    cycling_label = visual.TextStim(win, text=f'{block_name} 负荷蹬车', font=font_name,
        height=35, pos=(0, 250), color='white', bold=True)
    hint_text = visual.TextStim(win, text='按空格键暂停/恢复计时', font=font_name,
        height=22, pos=(0, -350), color='lightgray')
    
    timer = core.Clock()
    timer.reset()
    paused = False
    pause_duration = 0
    pause_start = 0
    actual_time = 0
    
    print(f"开始骑行: {block_name} ({duration}秒)")
    
    while actual_time < duration:
        keys = event.getKeys(keyList=['space', 'escape'])
        if 'escape' in keys:
            safe_quit(save=True)
        if 'space' in keys:
            if not paused:
                paused = True
                pause_start = timer.getTime()
                print("计时暂停")
            else:
                paused = False
                pause_duration += timer.getTime() - pause_start
                print("计时恢复")
        
        if paused:
            actual_time = pause_start - pause_duration
        else:
            actual_time = timer.getTime() - pause_duration
        
        if actual_time >= duration:
            break
        
        cycling_label.draw()
        timer_text.text = format_time(actual_time)
        timer_text.draw()
        hint_text.draw()
        if paused:
            pause_text.draw()
        win.flip()
    
    print(f"骑行结束: {block_name}, 实际时间: {actual_time:.1f}秒")
    
    results.append({
        'participant': exp_info['participant'],
        'gender': exp_info['gender'],
        'session': exp_info['session'],
        'block': block_name,
        'scale_type': 'cycling_time',
        'StartTime': None,
        'EndTime': None,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    })
    return True, actual_time


# ========== 主实验流程 ==========

try:
    print("\n=== 实验开始 ===")
    
    welcome_msg = ('欢迎您参与实验！\n\n'
        '接下来进入约10分钟的负荷蹬车环节。\n'
        '每3分钟会弹出主观评定量表，请根据真实感受认真回答。\n\n'
        '请调整姿势，骑行过程中上半身尽量保持稳定。\n\n'
        '按空格键开始第一次评定')
    show_message(welcome_msg, wait_keys=True)
    
    run_questionnaire('Baseline')
    
    for block in ['B+', 'C', 'C+', 'C++']:
        run_cycling(block, duration=180)  # 3分钟
        run_questionnaire(block)
    
    # D环节
    executed, _ = run_cycling('D', duration=180, allow_skip=True)  # D，3分钟
    if executed:
        run_questionnaire('D')
    
    # 结束语
    end_msg = ('实验结束！\n\n'
        '请不要停止蹬车！\n'
        '联系主试进入下一阶段。')
    show_message(end_msg, wait_keys=True)
    
    filename = save_data()
    show_message(f'数据已保存至:\n{filename}\n\n按空格键退出', wait_keys=True)

except Exception as e:
    print(f"\n!!! 实验出错: {e}")
    import traceback
    traceback.print_exc()
    if len(results) > 0:
        try:
            save_data()
        except:
            pass

finally:
    print("\n=== 实验结束 ===")
    safe_quit(save=False)