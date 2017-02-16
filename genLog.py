import pythoncom, pyHook
from pyHook import HookConstants
import win32con,win32api,win32ui,win32gui
import logging
import os 
import ctypes

import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--p', type=str, help='path to save folder', default='')
parser.add_argument('--i', type=bool, help='whether to save screenshot or not?', default=True)
args = parser.parse_args()
save_path = args.p
saveSS = args.i

##solve HiDPI display. This 2 lines are needed if you run the script on High DPI screen, without them screen resolution and cursor coordinate will be incorrect.
user32 = ctypes.windll.user32
user32.SetProcessDPIAware()

Minimize = win32gui.GetForegroundWindow()
win32gui.ShowWindow(Minimize,win32con.SW_MINIMIZE)

if not os.path.exists(save_path):
	os.mkdir(save_path)

if os.path.exists(save_path + 'log.txt'):
	os.remove(save_path + 'log.txt')
	
if not os.path.exists(save_path + 'imgs'):
	os.mkdir(save_path + 'imgs')

logging.basicConfig(filename=save_path + 'log.txt',filemode='a',level=logging.INFO)
logger = logging.getLogger()

main_thread_id = win32api.GetCurrentThreadId()

pressL = False
pressR = False

WIDTH = win32api.GetSystemMetrics(0)
HEIGHT = win32api.GetSystemMetrics(1)

hwin = win32gui.GetDesktopWindow()
hwindc = win32gui.GetWindowDC(hwin)
srcdc = win32ui.CreateDCFromHandle(hwindc)
memdc = srcdc.CreateCompatibleDC()

bmp = win32ui.CreateBitmap()
bmp.CreateCompatibleBitmap(srcdc, WIDTH , HEIGHT)
memdc.SelectObject(bmp)

counter = 0
last_time = 0

def OnMouseEventLD(event): #when the left button of the mouse is pressed.
	CTRL = ''
	SHIFT = ''
	#check if Shift key is down.
	if pyHook.GetKeyState(HookConstants.VKeyToID('VK_LSHIFT')) or pyHook.GetKeyState(HookConstants.VKeyToID('VK_RSHIFT')):
		SHIFT = 'Shift-'
	#check if Control key is down.
	if pyHook.GetKeyState(HookConstants.VKeyToID('VK_LCONTROL')) or pyHook.GetKeyState(HookConstants.VKeyToID('VK_RCONTROL')):
		CTRL = 'Ctr-'
	#write 2 screenshots to disk.
	if saveSS:
		global counter
		counter -= 1
		imgname = '%.6d.bmp'%(counter) #the log for this image was logged while mouse was moving.
		bmp.SaveBitmapFile(memdc, save_path + 'imgs/' + imgname) #save an image before the mouse button is pressed.
		counter += 1

		memdc.BitBlt((0, 0), (WIDTH , HEIGHT), srcdc, (0, 0), win32con.SRCCOPY)
		imgname = '%.6d.bmp'%(counter)
		bmp.SaveBitmapFile(memdc, save_path + 'imgs/' + imgname) #save an image when the mouse button is being pressed.
		counter += 1
	else:
		imgname = ''
	#log this activity
	logger.info(str(event.Time) + ',' + imgname + ',' + str(event.Position[0])+ ',' + str(event.Position[1]) + ',' + 'press left' + ','+CTRL+SHIFT)
	global pressL
	pressL = True
	return True
	
def OnMouseEventLU(event): #when the left button of the mouse is released.
	CTRL = ''
	SHIFT = ''
	#check if Shift key is down.
	if pyHook.GetKeyState(HookConstants.VKeyToID('VK_LSHIFT')) or pyHook.GetKeyState(HookConstants.VKeyToID('VK_RSHIFT')):
		SHIFT = 'Shift-'
	#check if Control key is down.
	if pyHook.GetKeyState(HookConstants.VKeyToID('VK_LCONTROL')) or pyHook.GetKeyState(HookConstants.VKeyToID('VK_RCONTROL')):
		CTRL = 'Ctr-'
	#write 2 screenshots to disk.
	if saveSS:
		global counter
		counter -= 1
		imgname = '%.6d.bmp'%(counter) #the log for this image was logged while mouse was moving.
		bmp.SaveBitmapFile(memdc, save_path + 'imgs/' + imgname) #save an image before the mouse button is released.
		counter += 1
		memdc.BitBlt((0, 0), (WIDTH , HEIGHT), srcdc, (0, 0), win32con.SRCCOPY)
		imgname = '%.6d.bmp'%(counter)
		bmp.SaveBitmapFile(memdc, save_path + 'imgs/' + imgname) #save an image when the mouse button is being released.
		counter += 1
	else:
		imgname = ''
	#log this activity
	logger.info(str(event.Time) + ',' + imgname + ',' + str(event.Position[0])+ ',' + str(event.Position[1]) + ',' + 'no press' + ','+CTRL+SHIFT)
	global pressL
	pressL = False
	return True
	
def OnMouseEventRD(event): #when the right button of the mouse is pressed.
	CTRL = ''
	SHIFT = ''
	#check if Shift key is down.
	if pyHook.GetKeyState(HookConstants.VKeyToID('VK_LSHIFT')) or pyHook.GetKeyState(HookConstants.VKeyToID('VK_RSHIFT')):
		SHIFT = 'Shift-'
	#check if Control key is down.
	if pyHook.GetKeyState(HookConstants.VKeyToID('VK_LCONTROL')) or pyHook.GetKeyState(HookConstants.VKeyToID('VK_RCONTROL')):
		CTRL = 'Ctr-'
	#write 2 screenshots to disk.
	if saveSS:
		global counter
		counter -= 1
		imgname = '%.6d.bmp'%(counter) #the log for this image was logged while mouse was moving.
		bmp.SaveBitmapFile(memdc, save_path + 'imgs/' + imgname) #save an image before the mouse button is pressed.
		counter += 1
		memdc.BitBlt((0, 0), (WIDTH , HEIGHT), srcdc, (0, 0), win32con.SRCCOPY)
		imgname = '%.6d.bmp'%(counter)
		bmp.SaveBitmapFile(memdc, save_path + 'imgs/' + imgname) #save an image when the mouse button is being pressed.
		counter += 1
	else:
		imgname = ''
	#log this activity
	logger.info(str(event.Time) + ',' + imgname + ',' + str(event.Position[0])+ ',' + str(event.Position[1]) + ',' + 'press right' + ','+CTRL+SHIFT)
	global pressR
	pressR = True
	return True

def OnMouseEventRU(event): #when the right button of the mouse is released.
	CTRL = ''
	SHIFT = ''
	#check if Shift key is down.
	if pyHook.GetKeyState(HookConstants.VKeyToID('VK_LSHIFT')) or pyHook.GetKeyState(HookConstants.VKeyToID('VK_RSHIFT')):
		SHIFT = 'Shift-'
	#check if Control key is down.
	if pyHook.GetKeyState(HookConstants.VKeyToID('VK_LCONTROL')) or pyHook.GetKeyState(HookConstants.VKeyToID('VK_RCONTROL')):
		CTRL = 'Ctr-'
	#write 2 screenshots to disk.
	if saveSS:
		global counter
		counter -= 1
		imgname = '%.6d.bmp'%(counter) #the log for this image was logged while mouse was moving.
		bmp.SaveBitmapFile(memdc, save_path + 'imgs/' + imgname) #save an image before the mouse button is released.
		counter += 1
		memdc.BitBlt((0, 0), (WIDTH , HEIGHT), srcdc, (0, 0), win32con.SRCCOPY)
		imgname = '%.6d.bmp'%(counter)
		bmp.SaveBitmapFile(memdc, save_path + 'imgs/' + imgname) #save an image when the mouse button is being released.
		counter += 1
	else:
		imgname = ''
	#log this activity
	logger.info(str(event.Time) + ',' + imgname + ',' + str(event.Position[0])+ ',' + str(event.Position[1]) + ',' + 'no press' + ','+CTRL+SHIFT)
	global pressR
	pressR = False
	return True

def OnMouseEventMV(event): #while the mouse is moving, keep logging. It Does not write any image to the disk, but load a screenshot to the memory. 

	global memdc
	memdc.BitBlt((0, 0), (WIDTH , HEIGHT), srcdc, (0, 0), win32con.SRCCOPY) #load the screenshot into memory
	global counter
	global last_time
	last_time = event.Time
	if pressL: #while the Left mouse button is being pressed. 
		if saveSS:
			imgname = '%.6d.bmp'%(counter)
			counter += 1
		else:
			imgname = ''
		logger.info(str(event.Time) + ',' + imgname + ',' + str(event.Position[0])+ ',' + str(event.Position[1]) + ',' + 'press left' + ',')

	elif pressR: #while the Right mouse button is being pressed. 
		if saveSS:
			imgname = '%.6d.bmp'%(counter)
			counter += 1
		else:
			imgname = ''
		logger.info(str(event.Time) + ',' + imgname + ',' + str(event.Position[0])+ ',' + str(event.Position[1]) + ',' + 'press right' + ',')
	
	else:
		if saveSS:
			imgname = '%.6d.bmp'%(counter)
			counter += 1
		else:
			imgname = ''
		logger.info(str(event.Time) + ',' + imgname + ',' + str(event.Position[0])+ ',' + str(event.Position[1]) + ',' + 'no press' + ',')


	return True

def OnKeyboardEvent(event): #Log a keyboard typing action

	f_,h_,mospos = win32gui.GetCursorInfo()
	global counter
	CTRL = ''
	SHIFT = ''
	#check if Shift key is down.
	if pyHook.GetKeyState(HookConstants.VKeyToID('VK_LSHIFT')) or pyHook.GetKeyState(HookConstants.VKeyToID('VK_RSHIFT')):
		SHIFT = 'Shift-'
	#check if Control key is down.
	if pyHook.GetKeyState(HookConstants.VKeyToID('VK_LCONTROL')) or pyHook.GetKeyState(HookConstants.VKeyToID('VK_RCONTROL')):
		CTRL = 'Ctr-'
	#if Shift+ESC are pressed together, end the logging session
	if (pyHook.GetKeyState(HookConstants.VKeyToID('VK_LSHIFT')) or pyHook.GetKeyState(HookConstants.VKeyToID('VK_RSHIFT'))) and event.KeyID == 27: #esc(key#27)
		counter += 1	
		imgname = '%.6d.bmp'%(counter)
		bmp.SaveBitmapFile(memdc, save_path + 'imgs/' + imgname) #save the last image before exit (screenshot when after everything is done)
		win32gui.ShowWindow(Minimize,win32con.SW_MAXIMIZE)
		win32api.PostThreadMessage(main_thread_id, win32con.WM_QUIT, 0, 0)
	#looping signal (Shift+Control+L)
	elif pyHook.GetKeyState(HookConstants.VKeyToID('VK_LSHIFT')) and pyHook.GetKeyState(HookConstants.VKeyToID('VK_LCONTROL')) and event.Key == 'L':
		if saveSS:
			imgname = '%.6d.bmp'%(counter)
			counter += 1
		else:
			imgname = ''
		logger.info(str(event.Time) +  ',' + imgname + ',' + str(mospos[0]) + ',' + str(mospos[1]) + ',' + 'no press' + ',' + 'Ctr-Shift-l')
	#looping signal (Shift+Control+pause/break)
	elif pyHook.GetKeyState(HookConstants.VKeyToID('VK_LSHIFT')) and pyHook.GetKeyState(HookConstants.VKeyToID('VK_LCONTROL')) and event.Key == 'Cancel':
		if saveSS:
			imgname = '%.6d.bmp'%(counter)
			counter += 1
		else:
			imgname = ''
		logger.info(str(event.Time) +  ',' + imgname + ',' + str(mospos[0]) + ',' + str(mospos[1]) + ',' + 'no press' + ',' + 'Ctr-Shift-Pause')
	#standby signal (Shift+Control+w or Shift+Control+PrtScr)
	elif pyHook.GetKeyState(HookConstants.VKeyToID('VK_LSHIFT')) and pyHook.GetKeyState(HookConstants.VKeyToID('VK_LCONTROL')) and (event.Key == 'W' or event.Key == 'Snapshot'):
		if saveSS:
			imgname = '%.6d.bmp'%(counter)
			bmp.SaveBitmapFile(memdc, save_path + 'imgs/' + imgname)
			counter += 1
		else:
			imgname = ''
		logger.info(str(event.Time) +  ',' + imgname + ',' + str(mospos[0]) + ',' + str(mospos[1]) + ',' + 'no press' + ',' + 'Ctr-Shift-w')
	
	else: #while pressing other keys (a, b, ... etc )
		if pressL:
			if saveSS:
				imgname = '%.6d.bmp'%(counter)
				counter += 1
			else:
				imgname = ''
			logger.info(str(event.Time) +  ',' + imgname + ',' + str(mospos[0]) + ',' + str(mospos[1]) + ',' + 'press left' + ',' + CTRL+SHIFT+event.Key)

		elif pressR:
			if saveSS:
				imgname = '%.6d.bmp'%(counter)
				counter += 1
			else:
				imgname = ''
			logger.info(str(event.Time) +  ',' + imgname + ',' + str(mospos[0]) + ',' + str(mospos[1]) + ',' + 'press right' + ',' + CTRL+SHIFT+event.Key)
			
		else:
			if saveSS:
				imgname = '%.6d.bmp'%(counter)
				counter += 1
			else:
				imgname = ''
			logger.info(str(event.Time) +  ',' + imgname + ',' + str(mospos[0]) + ',' + str(mospos[1]) + ',' + 'no press' + ',' + CTRL+SHIFT+event.Key)

	return True
	
def main():
	print 'START!!'
	
	# create a hook manager
	hm = pyHook.HookManager()

	# watch for all mouse events
	hm.KeyDown = OnKeyboardEvent
	hm.MouseLeftDown = OnMouseEventLD
	hm.MouseLeftUp = OnMouseEventLU
	hm.MouseRightDown = OnMouseEventRD
	hm.MouseRightUp = OnMouseEventRU
	hm.MouseMove = OnMouseEventMV
	# set the hook
	hm.HookMouse()
	hm.HookKeyboard()
	# wait forever
	pythoncom.PumpMessages()
	
if __name__ == "__main__":
	main()
