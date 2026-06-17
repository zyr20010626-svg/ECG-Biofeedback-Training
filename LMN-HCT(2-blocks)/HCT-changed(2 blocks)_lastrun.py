#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.2),
    on 五月 30, 2026, at 10:36
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'sounddevice'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.2'
expName = 'HCT-3'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1536, 864]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_session_%s_%s' % (expInfo['participant'], expInfo['session'], expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='D:\\毕业论文\\26年5月实验二\\LMN-HCT(2-blocks)\\HCT-changed(2 blocks)_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowStencil=True,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height', 
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.mouseVisible = True
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('key_resp') is None:
        # initialise key_resp
        key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp',
        )
    if deviceManager.getDevice('key_resp_2') is None:
        # initialise key_resp_2
        key_resp_2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_2',
        )
    if deviceManager.getDevice('key_resp_4') is None:
        # initialise key_resp_4
        key_resp_4 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_4',
        )
    # create speaker 'beep_1'
    deviceManager.addDevice(
        deviceName='beep_1',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=7.0
    )
    # create speaker 'beep_2'
    deviceManager.addDevice(
        deviceName='beep_2',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=7.0
    )
    # create speaker 'beep_3'
    deviceManager.addDevice(
        deviceName='beep_3',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=7.0
    )
    if deviceManager.getDevice('key_resp_3') is None:
        # initialise key_resp_3
        key_resp_3 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_3',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "welcome" ---
    key_resp = keyboard.Keyboard(deviceName='key_resp')
    welcome_textbox = visual.TextBox2(
         win, text='欢迎您参与我的实验！\n\n在开始前请确认以下内容：\n\n请调整姿势，在骑行过程中上半身尽量不要动。\n这项任务很简单，但请认真遵守。\n\n下面我们将进行的是心跳计数任务。\n\n请点击任意键进入第一部分：', placeholder='Type here...', font='Microsoft YaHei',
         ori=0.0, pos=(0, 0), draggable=False,      letterHeight=0.05,
         size=(1.2, 1.2), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='welcome_textbox',
         depth=-1, autoLog=True,
    )
    
    # --- Initialize components for Routine "Instructions_Heartbeat_2" ---
    key_resp_2 = keyboard.Keyboard(deviceName='key_resp_2')
    introducion_textbox = visual.TextBox2(
         win, text='心跳计数任务\n\n实验中，会让你试着去感受你的心跳，不是主动地感觉(触摸)你的手腕或脖子的脉动。你可能有感觉，也可能根本感觉不到，或者偶尔会有一些感觉。\n\n当听到第一个信号时，请将注意放到你的心跳上；5s后将会出现第二个信号，请你开始数你感觉到的心跳。当听到第三个信号时，请停止数心跳并输入你数的数量以及对你的自信心进行打分。（认真感受和思考自信心评分，若数次的信心相同、明显敷衍，被试费将会被扣除）\n\n对这项任务，不要试图猜测你的心率。实验中请忽略外界噪音、自行车噪音等一切声音，认真聆听自己的内在身体节律。\n\n如果对实验有何问题，及时询问主试。点击键盘任意按键进入实验。', placeholder='Type here...', font='Microsoft YaHei',
         ori=0.0, pos=(0, 0), draggable=False,      letterHeight=0.04,
         size=(1.2, 1.2), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='introducion_textbox',
         depth=-1, autoLog=True,
    )
    
    # --- Initialize components for Routine "Ready_" ---
    ready_textbox = visual.TextBox2(
         win, text='准备好了吗？\n\n准备好点击任意键进入实验', placeholder='Type here...', font='Microsoft YaHei',
         ori=0.0, pos=(0, 0), draggable=False,      letterHeight=0.05,
         size=(0.5, 0.5), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='ready_textbox',
         depth=0, autoLog=True,
    )
    key_resp_4 = keyboard.Keyboard(deviceName='key_resp_4')
    
    # --- Initialize components for Routine "Preparation_Heartbeat" ---
    beep_1 = sound.Sound(
        'A', 
        secs=0.5, 
        stereo=True, 
        hamming=True, 
        speaker='beep_1',    name='beep_1'
    )
    beep_1.setVolume(1.0)
    preparation_text = visual.TextStim(win=win, name='preparation_text',
        text='将您的注意集中在您的心跳上。',
        font='Microsoft YaHei',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    beep_2 = sound.Sound(
        'A', 
        secs=0.5, 
        stereo=True, 
        hamming=True, 
        speaker='beep_2',    name='beep_2'
    )
    beep_2.setVolume(1.0)
    
    # --- Initialize components for Routine "Heartbeat_Count" ---
    HeartbeatCount_text = visual.TextStim(win=win, name='HeartbeatCount_text',
        text='',
        font='Microsoft YaHei',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    # Run 'Begin Experiment' code from code
    import time
    from datetime import datetime
    
    # --- Initialize components for Routine "EndBlock_Heartbeat" ---
    beep_3 = sound.Sound(
        'A', 
        secs=0.5, 
        stereo=True, 
        hamming=True, 
        speaker='beep_3',    name='beep_3'
    )
    beep_3.setVolume(1.0)
    EndBlock_Heartbeat_text = visual.TextStim(win=win, name='EndBlock_Heartbeat_text',
        text='停止计数',
        font='Microsoft YaHei',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "Report_Heartbeat" ---
    question_heartbeat = visual.TextStim(win=win, name='question_heartbeat',
        text='您刚刚数了多少次心跳？',
        font='Microsoft YaHei',
        pos=(0,0.2), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    heartbeatCount = visual.TextBox2(
         win, text=None, placeholder='--', font='Microsoft YaHei',
         ori=0.0, pos=(0, -0.2), draggable=False,      letterHeight=0.05,
         size=(0.5, 0.5), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=True,
         name='heartbeatCount',
         depth=-1, autoLog=True,
    )
    
    # --- Initialize components for Routine "report_confident" ---
    question_confidence_2 = visual.TextStim(win=win, name='question_confidence_2',
        text='对您刚刚回答的结果自信程度进行打分',
        font='Microsoft YaHei',
        pos=(0, 0.2), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    confidenceRating_2 = visual.Slider(win=win, name='confidenceRating_2',
        startValue=None, size=(1.0, 0.1), pos=(0, -0.2), units=win.units,
        labels=["完全不自信", "", "", "", "", "", "", "", "", "非常自信"], ticks=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10), granularity=1.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Microsoft YaHei', labelHeight=0.03,
        flip=False, ori=0.0, depth=-1, readOnly=False)
    
    # --- Initialize components for Routine "Rest" ---
    Rest_text = visual.TextStim(win=win, name='Rest_text',
        text='间歇但不要停止蹬车哦',
        font='Microsoft YaHei',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "END" ---
    END_text = visual.TextStim(win=win, name='END_text',
        text='第一部分结束，请联系主试进入第二部分',
        font='Microsoft YaHei',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_3 = keyboard.Keyboard(deviceName='key_resp_3')
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "welcome" ---
    # create an object to store info about Routine welcome
    welcome = data.Routine(
        name='welcome',
        components=[key_resp, welcome_textbox],
    )
    welcome.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp
    key_resp.keys = []
    key_resp.rt = []
    _key_resp_allKeys = []
    welcome_textbox.reset()
    # store start times for welcome
    welcome.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    welcome.tStart = globalClock.getTime(format='float')
    welcome.status = STARTED
    thisExp.addData('welcome.started', welcome.tStart)
    welcome.maxDuration = None
    # keep track of which components have finished
    welcomeComponents = welcome.components
    for thisComponent in welcome.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "welcome" ---
    welcome.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *key_resp* updates
        waitOnFlip = False
        
        # if key_resp is starting this frame...
        if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp.frameNStart = frameN  # exact frame index
            key_resp.tStart = t  # local t and not account for scr refresh
            key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp.started')
            # update status
            key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp.status == STARTED and not waitOnFlip:
            theseKeys = key_resp.getKeys(keyList=None, ignoreKeys=["escape"], waitRelease=False)
            _key_resp_allKeys.extend(theseKeys)
            if len(_key_resp_allKeys):
                key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                key_resp.rt = _key_resp_allKeys[-1].rt
                key_resp.duration = _key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *welcome_textbox* updates
        
        # if welcome_textbox is starting this frame...
        if welcome_textbox.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            welcome_textbox.frameNStart = frameN  # exact frame index
            welcome_textbox.tStart = t  # local t and not account for scr refresh
            welcome_textbox.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(welcome_textbox, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'welcome_textbox.started')
            # update status
            welcome_textbox.status = STARTED
            welcome_textbox.setAutoDraw(True)
        
        # if welcome_textbox is active this frame...
        if welcome_textbox.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            welcome.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in welcome.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "welcome" ---
    for thisComponent in welcome.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for welcome
    welcome.tStop = globalClock.getTime(format='float')
    welcome.tStopRefresh = tThisFlipGlobal
    thisExp.addData('welcome.stopped', welcome.tStop)
    # check responses
    if key_resp.keys in ['', [], None]:  # No response was made
        key_resp.keys = None
    thisExp.addData('key_resp.keys',key_resp.keys)
    if key_resp.keys != None:  # we had a response
        thisExp.addData('key_resp.rt', key_resp.rt)
        thisExp.addData('key_resp.duration', key_resp.duration)
    thisExp.nextEntry()
    # the Routine "welcome" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Instructions_Heartbeat_2" ---
    # create an object to store info about Routine Instructions_Heartbeat_2
    Instructions_Heartbeat_2 = data.Routine(
        name='Instructions_Heartbeat_2',
        components=[key_resp_2, introducion_textbox],
    )
    Instructions_Heartbeat_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_2
    key_resp_2.keys = []
    key_resp_2.rt = []
    _key_resp_2_allKeys = []
    introducion_textbox.reset()
    # store start times for Instructions_Heartbeat_2
    Instructions_Heartbeat_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Instructions_Heartbeat_2.tStart = globalClock.getTime(format='float')
    Instructions_Heartbeat_2.status = STARTED
    thisExp.addData('Instructions_Heartbeat_2.started', Instructions_Heartbeat_2.tStart)
    Instructions_Heartbeat_2.maxDuration = None
    # keep track of which components have finished
    Instructions_Heartbeat_2Components = Instructions_Heartbeat_2.components
    for thisComponent in Instructions_Heartbeat_2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Instructions_Heartbeat_2" ---
    Instructions_Heartbeat_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *key_resp_2* updates
        waitOnFlip = False
        
        # if key_resp_2 is starting this frame...
        if key_resp_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_2.frameNStart = frameN  # exact frame index
            key_resp_2.tStart = t  # local t and not account for scr refresh
            key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_2.started')
            # update status
            key_resp_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_2.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_2.getKeys(keyList=None, ignoreKeys=["escape"], waitRelease=False)
            _key_resp_2_allKeys.extend(theseKeys)
            if len(_key_resp_2_allKeys):
                key_resp_2.keys = _key_resp_2_allKeys[-1].name  # just the last key pressed
                key_resp_2.rt = _key_resp_2_allKeys[-1].rt
                key_resp_2.duration = _key_resp_2_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *introducion_textbox* updates
        
        # if introducion_textbox is starting this frame...
        if introducion_textbox.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            introducion_textbox.frameNStart = frameN  # exact frame index
            introducion_textbox.tStart = t  # local t and not account for scr refresh
            introducion_textbox.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(introducion_textbox, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'introducion_textbox.started')
            # update status
            introducion_textbox.status = STARTED
            introducion_textbox.setAutoDraw(True)
        
        # if introducion_textbox is active this frame...
        if introducion_textbox.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Instructions_Heartbeat_2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Instructions_Heartbeat_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Instructions_Heartbeat_2" ---
    for thisComponent in Instructions_Heartbeat_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Instructions_Heartbeat_2
    Instructions_Heartbeat_2.tStop = globalClock.getTime(format='float')
    Instructions_Heartbeat_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Instructions_Heartbeat_2.stopped', Instructions_Heartbeat_2.tStop)
    # check responses
    if key_resp_2.keys in ['', [], None]:  # No response was made
        key_resp_2.keys = None
    thisExp.addData('key_resp_2.keys',key_resp_2.keys)
    if key_resp_2.keys != None:  # we had a response
        thisExp.addData('key_resp_2.rt', key_resp_2.rt)
        thisExp.addData('key_resp_2.duration', key_resp_2.duration)
    thisExp.nextEntry()
    # the Routine "Instructions_Heartbeat_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Ready_" ---
    # create an object to store info about Routine Ready_
    Ready_ = data.Routine(
        name='Ready_',
        components=[ready_textbox, key_resp_4],
    )
    Ready_.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    ready_textbox.reset()
    # create starting attributes for key_resp_4
    key_resp_4.keys = []
    key_resp_4.rt = []
    _key_resp_4_allKeys = []
    # store start times for Ready_
    Ready_.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Ready_.tStart = globalClock.getTime(format='float')
    Ready_.status = STARTED
    thisExp.addData('Ready_.started', Ready_.tStart)
    Ready_.maxDuration = None
    # keep track of which components have finished
    Ready_Components = Ready_.components
    for thisComponent in Ready_.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Ready_" ---
    Ready_.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *ready_textbox* updates
        
        # if ready_textbox is starting this frame...
        if ready_textbox.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            ready_textbox.frameNStart = frameN  # exact frame index
            ready_textbox.tStart = t  # local t and not account for scr refresh
            ready_textbox.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(ready_textbox, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'ready_textbox.started')
            # update status
            ready_textbox.status = STARTED
            ready_textbox.setAutoDraw(True)
        
        # if ready_textbox is active this frame...
        if ready_textbox.status == STARTED:
            # update params
            pass
        
        # *key_resp_4* updates
        waitOnFlip = False
        
        # if key_resp_4 is starting this frame...
        if key_resp_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_4.frameNStart = frameN  # exact frame index
            key_resp_4.tStart = t  # local t and not account for scr refresh
            key_resp_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_4.started')
            # update status
            key_resp_4.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_4.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_4.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_4.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_4.getKeys(keyList=None, ignoreKeys=["escape"], waitRelease=False)
            _key_resp_4_allKeys.extend(theseKeys)
            if len(_key_resp_4_allKeys):
                key_resp_4.keys = _key_resp_4_allKeys[-1].name  # just the last key pressed
                key_resp_4.rt = _key_resp_4_allKeys[-1].rt
                key_resp_4.duration = _key_resp_4_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Ready_.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Ready_.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Ready_" ---
    for thisComponent in Ready_.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Ready_
    Ready_.tStop = globalClock.getTime(format='float')
    Ready_.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Ready_.stopped', Ready_.tStop)
    # check responses
    if key_resp_4.keys in ['', [], None]:  # No response was made
        key_resp_4.keys = None
    thisExp.addData('key_resp_4.keys',key_resp_4.keys)
    if key_resp_4.keys != None:  # we had a response
        thisExp.addData('key_resp_4.rt', key_resp_4.rt)
        thisExp.addData('key_resp_4.duration', key_resp_4.duration)
    thisExp.nextEntry()
    # the Routine "Ready_" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    heartbeat_Loop = data.TrialHandler2(
        name='heartbeat_Loop',
        nReps=2.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('Conditions_HCT.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(heartbeat_Loop)  # add the loop to the experiment
    thisHeartbeat_Loop = heartbeat_Loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisHeartbeat_Loop.rgb)
    if thisHeartbeat_Loop != None:
        for paramName in thisHeartbeat_Loop:
            globals()[paramName] = thisHeartbeat_Loop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisHeartbeat_Loop in heartbeat_Loop:
        currentLoop = heartbeat_Loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisHeartbeat_Loop.rgb)
        if thisHeartbeat_Loop != None:
            for paramName in thisHeartbeat_Loop:
                globals()[paramName] = thisHeartbeat_Loop[paramName]
        
        # --- Prepare to start Routine "Preparation_Heartbeat" ---
        # create an object to store info about Routine Preparation_Heartbeat
        Preparation_Heartbeat = data.Routine(
            name='Preparation_Heartbeat',
            components=[beep_1, preparation_text, beep_2],
        )
        Preparation_Heartbeat.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        beep_1.setSound('A', secs=0.5, hamming=True)
        beep_1.setVolume(1.0, log=False)
        beep_1.seek(0)
        beep_2.setSound('A', secs=0.5, hamming=True)
        beep_2.setVolume(1.0, log=False)
        beep_2.seek(0)
        # store start times for Preparation_Heartbeat
        Preparation_Heartbeat.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Preparation_Heartbeat.tStart = globalClock.getTime(format='float')
        Preparation_Heartbeat.status = STARTED
        thisExp.addData('Preparation_Heartbeat.started', Preparation_Heartbeat.tStart)
        Preparation_Heartbeat.maxDuration = None
        # keep track of which components have finished
        Preparation_HeartbeatComponents = Preparation_Heartbeat.components
        for thisComponent in Preparation_Heartbeat.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Preparation_Heartbeat" ---
        # if trial has changed, end Routine now
        if isinstance(heartbeat_Loop, data.TrialHandler2) and thisHeartbeat_Loop.thisN != heartbeat_Loop.thisTrial.thisN:
            continueRoutine = False
        Preparation_Heartbeat.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 7.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *beep_1* updates
            
            # if beep_1 is starting this frame...
            if beep_1.status == NOT_STARTED and tThisFlip >= 1.5-frameTolerance:
                # keep track of start time/frame for later
                beep_1.frameNStart = frameN  # exact frame index
                beep_1.tStart = t  # local t and not account for scr refresh
                beep_1.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('beep_1.started', tThisFlipGlobal)
                # update status
                beep_1.status = STARTED
                beep_1.play(when=win)  # sync with win flip
            
            # if beep_1 is stopping this frame...
            if beep_1.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > beep_1.tStartRefresh + 0.5-frameTolerance or beep_1.isFinished:
                    # keep track of stop time/frame for later
                    beep_1.tStop = t  # not accounting for scr refresh
                    beep_1.tStopRefresh = tThisFlipGlobal  # on global time
                    beep_1.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'beep_1.stopped')
                    # update status
                    beep_1.status = FINISHED
                    beep_1.stop()
            
            # *preparation_text* updates
            
            # if preparation_text is starting this frame...
            if preparation_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                preparation_text.frameNStart = frameN  # exact frame index
                preparation_text.tStart = t  # local t and not account for scr refresh
                preparation_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(preparation_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'preparation_text.started')
                # update status
                preparation_text.status = STARTED
                preparation_text.setAutoDraw(True)
            
            # if preparation_text is active this frame...
            if preparation_text.status == STARTED:
                # update params
                pass
            
            # if preparation_text is stopping this frame...
            if preparation_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > preparation_text.tStartRefresh + 7.0-frameTolerance:
                    # keep track of stop time/frame for later
                    preparation_text.tStop = t  # not accounting for scr refresh
                    preparation_text.tStopRefresh = tThisFlipGlobal  # on global time
                    preparation_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'preparation_text.stopped')
                    # update status
                    preparation_text.status = FINISHED
                    preparation_text.setAutoDraw(False)
            
            # *beep_2* updates
            
            # if beep_2 is starting this frame...
            if beep_2.status == NOT_STARTED and tThisFlip >= 7-frameTolerance:
                # keep track of start time/frame for later
                beep_2.frameNStart = frameN  # exact frame index
                beep_2.tStart = t  # local t and not account for scr refresh
                beep_2.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('beep_2.started', tThisFlipGlobal)
                # update status
                beep_2.status = STARTED
                beep_2.play(when=win)  # sync with win flip
            
            # if beep_2 is stopping this frame...
            if beep_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > beep_2.tStartRefresh + 0.5-frameTolerance or beep_2.isFinished:
                    # keep track of stop time/frame for later
                    beep_2.tStop = t  # not accounting for scr refresh
                    beep_2.tStopRefresh = tThisFlipGlobal  # on global time
                    beep_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'beep_2.stopped')
                    # update status
                    beep_2.status = FINISHED
                    beep_2.stop()
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[beep_1, beep_2]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                Preparation_Heartbeat.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Preparation_Heartbeat.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Preparation_Heartbeat" ---
        for thisComponent in Preparation_Heartbeat.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Preparation_Heartbeat
        Preparation_Heartbeat.tStop = globalClock.getTime(format='float')
        Preparation_Heartbeat.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Preparation_Heartbeat.stopped', Preparation_Heartbeat.tStop)
        beep_1.pause()  # ensure sound has stopped at end of Routine
        beep_2.pause()  # ensure sound has stopped at end of Routine
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if Preparation_Heartbeat.maxDurationReached:
            routineTimer.addTime(-Preparation_Heartbeat.maxDuration)
        elif Preparation_Heartbeat.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-7.500000)
        
        # --- Prepare to start Routine "Heartbeat_Count" ---
        # create an object to store info about Routine Heartbeat_Count
        Heartbeat_Count = data.Routine(
            name='Heartbeat_Count',
            components=[HeartbeatCount_text],
        )
        Heartbeat_Count.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        HeartbeatCount_text.setText('注意--心跳')
        # Run 'Begin Routine' code from code
        thisExp.addData("StartTime",time.time())
        # store start times for Heartbeat_Count
        Heartbeat_Count.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Heartbeat_Count.tStart = globalClock.getTime(format='float')
        Heartbeat_Count.status = STARTED
        thisExp.addData('Heartbeat_Count.started', Heartbeat_Count.tStart)
        Heartbeat_Count.maxDuration = None
        # keep track of which components have finished
        Heartbeat_CountComponents = Heartbeat_Count.components
        for thisComponent in Heartbeat_Count.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Heartbeat_Count" ---
        # if trial has changed, end Routine now
        if isinstance(heartbeat_Loop, data.TrialHandler2) and thisHeartbeat_Loop.thisN != heartbeat_Loop.thisTrial.thisN:
            continueRoutine = False
        Heartbeat_Count.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *HeartbeatCount_text* updates
            
            # if HeartbeatCount_text is starting this frame...
            if HeartbeatCount_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                HeartbeatCount_text.frameNStart = frameN  # exact frame index
                HeartbeatCount_text.tStart = t  # local t and not account for scr refresh
                HeartbeatCount_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(HeartbeatCount_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'HeartbeatCount_text.started')
                # update status
                HeartbeatCount_text.status = STARTED
                HeartbeatCount_text.setAutoDraw(True)
            
            # if HeartbeatCount_text is active this frame...
            if HeartbeatCount_text.status == STARTED:
                # update params
                pass
            
            # if HeartbeatCount_text is stopping this frame...
            if HeartbeatCount_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > HeartbeatCount_text.tStartRefresh + HCT_time-frameTolerance:
                    # keep track of stop time/frame for later
                    HeartbeatCount_text.tStop = t  # not accounting for scr refresh
                    HeartbeatCount_text.tStopRefresh = tThisFlipGlobal  # on global time
                    HeartbeatCount_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'HeartbeatCount_text.stopped')
                    # update status
                    HeartbeatCount_text.status = FINISHED
                    HeartbeatCount_text.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                Heartbeat_Count.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Heartbeat_Count.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Heartbeat_Count" ---
        for thisComponent in Heartbeat_Count.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Heartbeat_Count
        Heartbeat_Count.tStop = globalClock.getTime(format='float')
        Heartbeat_Count.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Heartbeat_Count.stopped', Heartbeat_Count.tStop)
        # Run 'End Routine' code from code
        thisExp.addData("EndTime",time.time())
        # the Routine "Heartbeat_Count" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "EndBlock_Heartbeat" ---
        # create an object to store info about Routine EndBlock_Heartbeat
        EndBlock_Heartbeat = data.Routine(
            name='EndBlock_Heartbeat',
            components=[beep_3, EndBlock_Heartbeat_text],
        )
        EndBlock_Heartbeat.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        beep_3.setSound('A', secs=0.5, hamming=True)
        beep_3.setVolume(1.0, log=False)
        beep_3.seek(0)
        # store start times for EndBlock_Heartbeat
        EndBlock_Heartbeat.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        EndBlock_Heartbeat.tStart = globalClock.getTime(format='float')
        EndBlock_Heartbeat.status = STARTED
        thisExp.addData('EndBlock_Heartbeat.started', EndBlock_Heartbeat.tStart)
        EndBlock_Heartbeat.maxDuration = None
        # keep track of which components have finished
        EndBlock_HeartbeatComponents = EndBlock_Heartbeat.components
        for thisComponent in EndBlock_Heartbeat.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "EndBlock_Heartbeat" ---
        # if trial has changed, end Routine now
        if isinstance(heartbeat_Loop, data.TrialHandler2) and thisHeartbeat_Loop.thisN != heartbeat_Loop.thisTrial.thisN:
            continueRoutine = False
        EndBlock_Heartbeat.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *beep_3* updates
            
            # if beep_3 is starting this frame...
            if beep_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                beep_3.frameNStart = frameN  # exact frame index
                beep_3.tStart = t  # local t and not account for scr refresh
                beep_3.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('beep_3.started', tThisFlipGlobal)
                # update status
                beep_3.status = STARTED
                beep_3.play(when=win)  # sync with win flip
            
            # if beep_3 is stopping this frame...
            if beep_3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > beep_3.tStartRefresh + 0.5-frameTolerance or beep_3.isFinished:
                    # keep track of stop time/frame for later
                    beep_3.tStop = t  # not accounting for scr refresh
                    beep_3.tStopRefresh = tThisFlipGlobal  # on global time
                    beep_3.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'beep_3.stopped')
                    # update status
                    beep_3.status = FINISHED
                    beep_3.stop()
            
            # *EndBlock_Heartbeat_text* updates
            
            # if EndBlock_Heartbeat_text is starting this frame...
            if EndBlock_Heartbeat_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                EndBlock_Heartbeat_text.frameNStart = frameN  # exact frame index
                EndBlock_Heartbeat_text.tStart = t  # local t and not account for scr refresh
                EndBlock_Heartbeat_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(EndBlock_Heartbeat_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'EndBlock_Heartbeat_text.started')
                # update status
                EndBlock_Heartbeat_text.status = STARTED
                EndBlock_Heartbeat_text.setAutoDraw(True)
            
            # if EndBlock_Heartbeat_text is active this frame...
            if EndBlock_Heartbeat_text.status == STARTED:
                # update params
                pass
            
            # if EndBlock_Heartbeat_text is stopping this frame...
            if EndBlock_Heartbeat_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > EndBlock_Heartbeat_text.tStartRefresh + 2.0-frameTolerance:
                    # keep track of stop time/frame for later
                    EndBlock_Heartbeat_text.tStop = t  # not accounting for scr refresh
                    EndBlock_Heartbeat_text.tStopRefresh = tThisFlipGlobal  # on global time
                    EndBlock_Heartbeat_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'EndBlock_Heartbeat_text.stopped')
                    # update status
                    EndBlock_Heartbeat_text.status = FINISHED
                    EndBlock_Heartbeat_text.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[beep_3]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                EndBlock_Heartbeat.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in EndBlock_Heartbeat.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "EndBlock_Heartbeat" ---
        for thisComponent in EndBlock_Heartbeat.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for EndBlock_Heartbeat
        EndBlock_Heartbeat.tStop = globalClock.getTime(format='float')
        EndBlock_Heartbeat.tStopRefresh = tThisFlipGlobal
        thisExp.addData('EndBlock_Heartbeat.stopped', EndBlock_Heartbeat.tStop)
        beep_3.pause()  # ensure sound has stopped at end of Routine
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if EndBlock_Heartbeat.maxDurationReached:
            routineTimer.addTime(-EndBlock_Heartbeat.maxDuration)
        elif EndBlock_Heartbeat.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "Report_Heartbeat" ---
        # create an object to store info about Routine Report_Heartbeat
        Report_Heartbeat = data.Routine(
            name='Report_Heartbeat',
            components=[question_heartbeat, heartbeatCount],
        )
        Report_Heartbeat.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        heartbeatCount.reset()
        # store start times for Report_Heartbeat
        Report_Heartbeat.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Report_Heartbeat.tStart = globalClock.getTime(format='float')
        Report_Heartbeat.status = STARTED
        thisExp.addData('Report_Heartbeat.started', Report_Heartbeat.tStart)
        Report_Heartbeat.maxDuration = None
        # keep track of which components have finished
        Report_HeartbeatComponents = Report_Heartbeat.components
        for thisComponent in Report_Heartbeat.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Report_Heartbeat" ---
        # if trial has changed, end Routine now
        if isinstance(heartbeat_Loop, data.TrialHandler2) and thisHeartbeat_Loop.thisN != heartbeat_Loop.thisTrial.thisN:
            continueRoutine = False
        Report_Heartbeat.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *question_heartbeat* updates
            
            # if question_heartbeat is starting this frame...
            if question_heartbeat.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                question_heartbeat.frameNStart = frameN  # exact frame index
                question_heartbeat.tStart = t  # local t and not account for scr refresh
                question_heartbeat.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(question_heartbeat, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'question_heartbeat.started')
                # update status
                question_heartbeat.status = STARTED
                question_heartbeat.setAutoDraw(True)
            
            # if question_heartbeat is active this frame...
            if question_heartbeat.status == STARTED:
                # update params
                pass
            
            # *heartbeatCount* updates
            
            # if heartbeatCount is starting this frame...
            if heartbeatCount.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                heartbeatCount.frameNStart = frameN  # exact frame index
                heartbeatCount.tStart = t  # local t and not account for scr refresh
                heartbeatCount.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(heartbeatCount, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'heartbeatCount.started')
                # update status
                heartbeatCount.status = STARTED
                heartbeatCount.setAutoDraw(True)
            
            # if heartbeatCount is active this frame...
            if heartbeatCount.status == STARTED:
                # update params
                pass
            # Run 'Each Frame' code from checkResponseKey
            # 如果检测到回车键或空格键
            if 'return' in event.getKeys() or 'space' in event.getKeys():
                # 确保输入框不是空的（至少输入了一个数字）
                if len(heartbeatCount.text) > 0:
                    continueRoutine = False
            
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                Report_Heartbeat.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Report_Heartbeat.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Report_Heartbeat" ---
        for thisComponent in Report_Heartbeat.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Report_Heartbeat
        Report_Heartbeat.tStop = globalClock.getTime(format='float')
        Report_Heartbeat.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Report_Heartbeat.stopped', Report_Heartbeat.tStop)
        heartbeat_Loop.addData('heartbeatCount.text',heartbeatCount.text)
        # the Routine "Report_Heartbeat" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "report_confident" ---
        # create an object to store info about Routine report_confident
        report_confident = data.Routine(
            name='report_confident',
            components=[question_confidence_2, confidenceRating_2],
        )
        report_confident.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        confidenceRating_2.reset()
        # store start times for report_confident
        report_confident.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        report_confident.tStart = globalClock.getTime(format='float')
        report_confident.status = STARTED
        thisExp.addData('report_confident.started', report_confident.tStart)
        report_confident.maxDuration = None
        # keep track of which components have finished
        report_confidentComponents = report_confident.components
        for thisComponent in report_confident.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "report_confident" ---
        # if trial has changed, end Routine now
        if isinstance(heartbeat_Loop, data.TrialHandler2) and thisHeartbeat_Loop.thisN != heartbeat_Loop.thisTrial.thisN:
            continueRoutine = False
        report_confident.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *question_confidence_2* updates
            
            # if question_confidence_2 is starting this frame...
            if question_confidence_2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                question_confidence_2.frameNStart = frameN  # exact frame index
                question_confidence_2.tStart = t  # local t and not account for scr refresh
                question_confidence_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(question_confidence_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'question_confidence_2.started')
                # update status
                question_confidence_2.status = STARTED
                question_confidence_2.setAutoDraw(True)
            
            # if question_confidence_2 is active this frame...
            if question_confidence_2.status == STARTED:
                # update params
                pass
            
            # *confidenceRating_2* updates
            
            # if confidenceRating_2 is starting this frame...
            if confidenceRating_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                confidenceRating_2.frameNStart = frameN  # exact frame index
                confidenceRating_2.tStart = t  # local t and not account for scr refresh
                confidenceRating_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(confidenceRating_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'confidenceRating_2.started')
                # update status
                confidenceRating_2.status = STARTED
                confidenceRating_2.setAutoDraw(True)
            
            # if confidenceRating_2 is active this frame...
            if confidenceRating_2.status == STARTED:
                # update params
                pass
            
            # Check confidenceRating_2 for response to end Routine
            if confidenceRating_2.getRating() is not None and confidenceRating_2.status == STARTED:
                continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                report_confident.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in report_confident.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "report_confident" ---
        for thisComponent in report_confident.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for report_confident
        report_confident.tStop = globalClock.getTime(format='float')
        report_confident.tStopRefresh = tThisFlipGlobal
        thisExp.addData('report_confident.stopped', report_confident.tStop)
        heartbeat_Loop.addData('confidenceRating_2.response', confidenceRating_2.getRating())
        heartbeat_Loop.addData('confidenceRating_2.rt', confidenceRating_2.getRT())
        # the Routine "report_confident" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "Rest" ---
        # create an object to store info about Routine Rest
        Rest = data.Routine(
            name='Rest',
            components=[Rest_text],
        )
        Rest.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for Rest
        Rest.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Rest.tStart = globalClock.getTime(format='float')
        Rest.status = STARTED
        thisExp.addData('Rest.started', Rest.tStart)
        Rest.maxDuration = None
        # keep track of which components have finished
        RestComponents = Rest.components
        for thisComponent in Rest.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Rest" ---
        # if trial has changed, end Routine now
        if isinstance(heartbeat_Loop, data.TrialHandler2) and thisHeartbeat_Loop.thisN != heartbeat_Loop.thisTrial.thisN:
            continueRoutine = False
        Rest.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 5.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *Rest_text* updates
            
            # if Rest_text is starting this frame...
            if Rest_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Rest_text.frameNStart = frameN  # exact frame index
                Rest_text.tStart = t  # local t and not account for scr refresh
                Rest_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Rest_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Rest_text.started')
                # update status
                Rest_text.status = STARTED
                Rest_text.setAutoDraw(True)
            
            # if Rest_text is active this frame...
            if Rest_text.status == STARTED:
                # update params
                pass
            
            # if Rest_text is stopping this frame...
            if Rest_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Rest_text.tStartRefresh + 5-frameTolerance:
                    # keep track of stop time/frame for later
                    Rest_text.tStop = t  # not accounting for scr refresh
                    Rest_text.tStopRefresh = tThisFlipGlobal  # on global time
                    Rest_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Rest_text.stopped')
                    # update status
                    Rest_text.status = FINISHED
                    Rest_text.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                Rest.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Rest.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Rest" ---
        for thisComponent in Rest.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Rest
        Rest.tStop = globalClock.getTime(format='float')
        Rest.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Rest.stopped', Rest.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if Rest.maxDurationReached:
            routineTimer.addTime(-Rest.maxDuration)
        elif Rest.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-5.000000)
        thisExp.nextEntry()
        
    # completed 2.0 repeats of 'heartbeat_Loop'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "END" ---
    # create an object to store info about Routine END
    END = data.Routine(
        name='END',
        components=[END_text, key_resp_3],
    )
    END.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_3
    key_resp_3.keys = []
    key_resp_3.rt = []
    _key_resp_3_allKeys = []
    # store start times for END
    END.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    END.tStart = globalClock.getTime(format='float')
    END.status = STARTED
    thisExp.addData('END.started', END.tStart)
    END.maxDuration = None
    # keep track of which components have finished
    ENDComponents = END.components
    for thisComponent in END.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "END" ---
    END.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *END_text* updates
        
        # if END_text is starting this frame...
        if END_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            END_text.frameNStart = frameN  # exact frame index
            END_text.tStart = t  # local t and not account for scr refresh
            END_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(END_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'END_text.started')
            # update status
            END_text.status = STARTED
            END_text.setAutoDraw(True)
        
        # if END_text is active this frame...
        if END_text.status == STARTED:
            # update params
            pass
        
        # *key_resp_3* updates
        waitOnFlip = False
        
        # if key_resp_3 is starting this frame...
        if key_resp_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_3.frameNStart = frameN  # exact frame index
            key_resp_3.tStart = t  # local t and not account for scr refresh
            key_resp_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_3.started')
            # update status
            key_resp_3.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_3.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_3.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_3.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_3.getKeys(keyList=None, ignoreKeys=["escape"], waitRelease=False)
            _key_resp_3_allKeys.extend(theseKeys)
            if len(_key_resp_3_allKeys):
                key_resp_3.keys = _key_resp_3_allKeys[-1].name  # just the last key pressed
                key_resp_3.rt = _key_resp_3_allKeys[-1].rt
                key_resp_3.duration = _key_resp_3_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            END.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in END.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "END" ---
    for thisComponent in END.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for END
    END.tStop = globalClock.getTime(format='float')
    END.tStopRefresh = tThisFlipGlobal
    thisExp.addData('END.stopped', END.tStop)
    # check responses
    if key_resp_3.keys in ['', [], None]:  # No response was made
        key_resp_3.keys = None
    thisExp.addData('key_resp_3.keys',key_resp_3.keys)
    if key_resp_3.keys != None:  # we had a response
        thisExp.addData('key_resp_3.rt', key_resp_3.rt)
        thisExp.addData('key_resp_3.duration', key_resp_3.duration)
    thisExp.nextEntry()
    # the Routine "END" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
