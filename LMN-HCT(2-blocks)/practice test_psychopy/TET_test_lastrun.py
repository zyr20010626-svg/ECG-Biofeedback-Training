#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.2),
    on 十月 30, 2025, at 09:45
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
prefs.hardware['audioLib'] = 'ptb'
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
_winSize = [1280, 800]
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
        originPath='C:\\Users\\李大黄\\Desktop\\test_psychopy\\TET_test_lastrun.py',
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
    
    # --- Initialize components for Routine "Instructions_TET_2" ---
    key_resp_2 = keyboard.Keyboard(deviceName='key_resp_2')
    introducion_textbox = visual.TextBox2(
         win, text='接下来请进行时间计数任务\n\n在这个任务中，你将会听到三个提示音，第一个提示提示你做好准备，第二个提示声指示你开始数秒，第三个提示声声告诉你停止数秒，此时你应该告诉实验者你数了多少秒。\n\n如果对实验有何问题，及时询问主试。', placeholder='Type here...', font='Microsoft YaHei',
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
    
    # --- Initialize components for Routine "Preparation_TET" ---
    beep_1 = sound.Sound(
        'A', 
        secs=0.5, 
        stereo=True, 
        hamming=True, 
        speaker='beep_1',    name='beep_1'
    )
    beep_1.setVolume(1.0)
    preparation_text = visual.TextStim(win=win, name='preparation_text',
        text='准备开始计时',
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
    
    # --- Initialize components for Routine "TET_Count" ---
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
    
    # --- Initialize components for Routine "EndBlock_TET" ---
    beep_3 = sound.Sound(
        'A', 
        secs=0.5, 
        stereo=True, 
        hamming=True, 
        speaker='beep_3',    name='beep_3'
    )
    beep_3.setVolume(1.0)
    EndBlock_TET_text = visual.TextStim(win=win, name='EndBlock_TET_text',
        text='停止计数',
        font='Microsoft YaHei',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "Report_TET" ---
    question_TET = visual.TextStim(win=win, name='question_TET',
        text='您刚刚数了多少秒？',
        font='Microsoft YaHei',
        pos=(0,0.2), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    tetCount = visual.TextBox2(
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
         name='tetCount',
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
        text='休息',
        font='Microsoft YaHei',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "END" ---
    END_text = visual.TextStim(win=win, name='END_text',
        text='实验结束，感谢您的参与！\n\n请联系主试。',
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
    
    # --- Prepare to start Routine "Instructions_TET_2" ---
    # create an object to store info about Routine Instructions_TET_2
    Instructions_TET_2 = data.Routine(
        name='Instructions_TET_2',
        components=[key_resp_2, introducion_textbox],
    )
    Instructions_TET_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_2
    key_resp_2.keys = []
    key_resp_2.rt = []
    _key_resp_2_allKeys = []
    introducion_textbox.reset()
    # store start times for Instructions_TET_2
    Instructions_TET_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Instructions_TET_2.tStart = globalClock.getTime(format='float')
    Instructions_TET_2.status = STARTED
    thisExp.addData('Instructions_TET_2.started', Instructions_TET_2.tStart)
    Instructions_TET_2.maxDuration = None
    # keep track of which components have finished
    Instructions_TET_2Components = Instructions_TET_2.components
    for thisComponent in Instructions_TET_2.components:
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
    
    # --- Run Routine "Instructions_TET_2" ---
    Instructions_TET_2.forceEnded = routineForceEnded = not continueRoutine
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
            Instructions_TET_2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Instructions_TET_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Instructions_TET_2" ---
    for thisComponent in Instructions_TET_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Instructions_TET_2
    Instructions_TET_2.tStop = globalClock.getTime(format='float')
    Instructions_TET_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Instructions_TET_2.stopped', Instructions_TET_2.tStop)
    # check responses
    if key_resp_2.keys in ['', [], None]:  # No response was made
        key_resp_2.keys = None
    thisExp.addData('key_resp_2.keys',key_resp_2.keys)
    if key_resp_2.keys != None:  # we had a response
        thisExp.addData('key_resp_2.rt', key_resp_2.rt)
        thisExp.addData('key_resp_2.duration', key_resp_2.duration)
    thisExp.nextEntry()
    # the Routine "Instructions_TET_2" was not non-slip safe, so reset the non-slip timer
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
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('Conditions_TET.xlsx'), 
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
        
        # --- Prepare to start Routine "Preparation_TET" ---
        # create an object to store info about Routine Preparation_TET
        Preparation_TET = data.Routine(
            name='Preparation_TET',
            components=[beep_1, preparation_text, beep_2],
        )
        Preparation_TET.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        beep_1.setSound('A', secs=0.5, hamming=True)
        beep_1.setVolume(1.0, log=False)
        beep_1.seek(0)
        beep_2.setSound('A', secs=0.5, hamming=True)
        beep_2.setVolume(1.0, log=False)
        beep_2.seek(0)
        # store start times for Preparation_TET
        Preparation_TET.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Preparation_TET.tStart = globalClock.getTime(format='float')
        Preparation_TET.status = STARTED
        thisExp.addData('Preparation_TET.started', Preparation_TET.tStart)
        Preparation_TET.maxDuration = None
        # keep track of which components have finished
        Preparation_TETComponents = Preparation_TET.components
        for thisComponent in Preparation_TET.components:
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
        
        # --- Run Routine "Preparation_TET" ---
        # if trial has changed, end Routine now
        if isinstance(heartbeat_Loop, data.TrialHandler2) and thisHeartbeat_Loop.thisN != heartbeat_Loop.thisTrial.thisN:
            continueRoutine = False
        Preparation_TET.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 5.5:
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
                if tThisFlipGlobal > preparation_text.tStartRefresh + 5.0-frameTolerance:
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
            if beep_2.status == NOT_STARTED and tThisFlip >= 5-frameTolerance:
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
                Preparation_TET.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Preparation_TET.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Preparation_TET" ---
        for thisComponent in Preparation_TET.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Preparation_TET
        Preparation_TET.tStop = globalClock.getTime(format='float')
        Preparation_TET.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Preparation_TET.stopped', Preparation_TET.tStop)
        beep_1.pause()  # ensure sound has stopped at end of Routine
        beep_2.pause()  # ensure sound has stopped at end of Routine
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if Preparation_TET.maxDurationReached:
            routineTimer.addTime(-Preparation_TET.maxDuration)
        elif Preparation_TET.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-5.500000)
        
        # --- Prepare to start Routine "TET_Count" ---
        # create an object to store info about Routine TET_Count
        TET_Count = data.Routine(
            name='TET_Count',
            components=[HeartbeatCount_text],
        )
        TET_Count.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        HeartbeatCount_text.setText('注意--计时\n')
        # Run 'Begin Routine' code from code
        thisExp.addData("StartTime",time.time())
        # store start times for TET_Count
        TET_Count.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        TET_Count.tStart = globalClock.getTime(format='float')
        TET_Count.status = STARTED
        thisExp.addData('TET_Count.started', TET_Count.tStart)
        TET_Count.maxDuration = None
        # keep track of which components have finished
        TET_CountComponents = TET_Count.components
        for thisComponent in TET_Count.components:
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
        
        # --- Run Routine "TET_Count" ---
        # if trial has changed, end Routine now
        if isinstance(heartbeat_Loop, data.TrialHandler2) and thisHeartbeat_Loop.thisN != heartbeat_Loop.thisTrial.thisN:
            continueRoutine = False
        TET_Count.forceEnded = routineForceEnded = not continueRoutine
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
                if tThisFlipGlobal > HeartbeatCount_text.tStartRefresh + TET_time-frameTolerance:
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
                TET_Count.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in TET_Count.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "TET_Count" ---
        for thisComponent in TET_Count.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for TET_Count
        TET_Count.tStop = globalClock.getTime(format='float')
        TET_Count.tStopRefresh = tThisFlipGlobal
        thisExp.addData('TET_Count.stopped', TET_Count.tStop)
        # Run 'End Routine' code from code
        thisExp.addData("EndTime",time.time())
        # the Routine "TET_Count" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "EndBlock_TET" ---
        # create an object to store info about Routine EndBlock_TET
        EndBlock_TET = data.Routine(
            name='EndBlock_TET',
            components=[beep_3, EndBlock_TET_text],
        )
        EndBlock_TET.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        beep_3.setSound('A', secs=0.5, hamming=True)
        beep_3.setVolume(1.0, log=False)
        beep_3.seek(0)
        # store start times for EndBlock_TET
        EndBlock_TET.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        EndBlock_TET.tStart = globalClock.getTime(format='float')
        EndBlock_TET.status = STARTED
        thisExp.addData('EndBlock_TET.started', EndBlock_TET.tStart)
        EndBlock_TET.maxDuration = None
        # keep track of which components have finished
        EndBlock_TETComponents = EndBlock_TET.components
        for thisComponent in EndBlock_TET.components:
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
        
        # --- Run Routine "EndBlock_TET" ---
        # if trial has changed, end Routine now
        if isinstance(heartbeat_Loop, data.TrialHandler2) and thisHeartbeat_Loop.thisN != heartbeat_Loop.thisTrial.thisN:
            continueRoutine = False
        EndBlock_TET.forceEnded = routineForceEnded = not continueRoutine
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
            
            # *EndBlock_TET_text* updates
            
            # if EndBlock_TET_text is starting this frame...
            if EndBlock_TET_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                EndBlock_TET_text.frameNStart = frameN  # exact frame index
                EndBlock_TET_text.tStart = t  # local t and not account for scr refresh
                EndBlock_TET_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(EndBlock_TET_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'EndBlock_TET_text.started')
                # update status
                EndBlock_TET_text.status = STARTED
                EndBlock_TET_text.setAutoDraw(True)
            
            # if EndBlock_TET_text is active this frame...
            if EndBlock_TET_text.status == STARTED:
                # update params
                pass
            
            # if EndBlock_TET_text is stopping this frame...
            if EndBlock_TET_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > EndBlock_TET_text.tStartRefresh + 2.0-frameTolerance:
                    # keep track of stop time/frame for later
                    EndBlock_TET_text.tStop = t  # not accounting for scr refresh
                    EndBlock_TET_text.tStopRefresh = tThisFlipGlobal  # on global time
                    EndBlock_TET_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'EndBlock_TET_text.stopped')
                    # update status
                    EndBlock_TET_text.status = FINISHED
                    EndBlock_TET_text.setAutoDraw(False)
            
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
                EndBlock_TET.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in EndBlock_TET.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "EndBlock_TET" ---
        for thisComponent in EndBlock_TET.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for EndBlock_TET
        EndBlock_TET.tStop = globalClock.getTime(format='float')
        EndBlock_TET.tStopRefresh = tThisFlipGlobal
        thisExp.addData('EndBlock_TET.stopped', EndBlock_TET.tStop)
        beep_3.pause()  # ensure sound has stopped at end of Routine
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if EndBlock_TET.maxDurationReached:
            routineTimer.addTime(-EndBlock_TET.maxDuration)
        elif EndBlock_TET.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "Report_TET" ---
        # create an object to store info about Routine Report_TET
        Report_TET = data.Routine(
            name='Report_TET',
            components=[question_TET, tetCount],
        )
        Report_TET.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        tetCount.reset()
        # store start times for Report_TET
        Report_TET.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Report_TET.tStart = globalClock.getTime(format='float')
        Report_TET.status = STARTED
        thisExp.addData('Report_TET.started', Report_TET.tStart)
        Report_TET.maxDuration = None
        # keep track of which components have finished
        Report_TETComponents = Report_TET.components
        for thisComponent in Report_TET.components:
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
        
        # --- Run Routine "Report_TET" ---
        # if trial has changed, end Routine now
        if isinstance(heartbeat_Loop, data.TrialHandler2) and thisHeartbeat_Loop.thisN != heartbeat_Loop.thisTrial.thisN:
            continueRoutine = False
        Report_TET.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *question_TET* updates
            
            # if question_TET is starting this frame...
            if question_TET.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                question_TET.frameNStart = frameN  # exact frame index
                question_TET.tStart = t  # local t and not account for scr refresh
                question_TET.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(question_TET, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'question_TET.started')
                # update status
                question_TET.status = STARTED
                question_TET.setAutoDraw(True)
            
            # if question_TET is active this frame...
            if question_TET.status == STARTED:
                # update params
                pass
            
            # *tetCount* updates
            
            # if tetCount is starting this frame...
            if tetCount.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                tetCount.frameNStart = frameN  # exact frame index
                tetCount.tStart = t  # local t and not account for scr refresh
                tetCount.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(tetCount, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'tetCount.started')
                # update status
                tetCount.status = STARTED
                tetCount.setAutoDraw(True)
            
            # if tetCount is active this frame...
            if tetCount.status == STARTED:
                # update params
                pass
            # Run 'Each Frame' code from checkResponseKey
            # 如果检测到回车键或空格键
            if 'return' in event.getKeys() or 'space' in event.getKeys():
                # 确保输入框不是空的（至少输入了一个数字）
                if len(tetCount.text) > 0:
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
                Report_TET.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Report_TET.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Report_TET" ---
        for thisComponent in Report_TET.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Report_TET
        Report_TET.tStop = globalClock.getTime(format='float')
        Report_TET.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Report_TET.stopped', Report_TET.tStop)
        heartbeat_Loop.addData('tetCount.text',tetCount.text)
        # the Routine "Report_TET" was not non-slip safe, so reset the non-slip timer
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
        
    # completed 1.0 repeats of 'heartbeat_Loop'
    
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
