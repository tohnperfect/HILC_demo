""" for each basic action in the transcribed sequence, 
scan the current screenshot to find the target pattern and apply basic action to that pattern."""

import pyautogui as GUIbot # using as a robot to control mice and keyboards and also screenshot function
from time import sleep # for setting time delay
import numpy
import pickle
import argparse
import os
import utils
import matplotlib.pyplot as plt
from skimage.feature import match_template
import win32gui,win32con
import pygame
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--p', type=str, help='path to action list', default='')
parser.add_argument('--l', type=bool, help='if True, users define looping instances by themself', default=False)
parser.add_argument('--d', type=bool, help='if True, enter debug mode', default=False)
parser.add_argument('--q', type=bool, help='if True, enter query mode', default=False)
args = parser.parse_args()

action_path = args.p
DEBUG = args.d
DEFINE = args.l
QUERY = args.q
Minimize = win32gui.GetForegroundWindow()
win32gui.ShowWindow(Minimize,win32con.SW_MINIMIZE)

with open(action_path+'prepared_actionlist.pickle', 'rb') as handle:
	loaded = pickle.load(handle)

if len(loaded) == 2 and 'class' not in loaded[1]:
	[actionlist,LOOP_ACTIONS] = loaded
	collect_action = False
else:
	actionlist = loaded
	LOOP_ACTIONS = list()
	collect_action = True

#### pygame console is used here in case the end-user does not want to use the pre-trained detector for looping. 
### The cosole is called to let the user to mark the instances she want to loop over.
##########################################################################----pygame----###########################################################
def displayImage(screen,px):
	pygame.draw.rect(screen, (0,255,0), pygame.Rect(pygame.mouse.get_pos()[0]-15, pygame.mouse.get_pos()[1]-15,31,31),2)
	pygame.display.flip()
	
def setup(path):
	px = pygame.image.load(path)
	screen = pygame.display.set_mode(px.get_rect()[2:])
	screen.blit(px,px.get_rect())
	pygame.display.flip()

	return screen,px
	
def mainLoop(screen, px, instances, txt):
	runProgram = True
	pygame.display.set_caption(txt)
	while runProgram:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				runProgram = False
				print 'quit!!'
			elif event.type == pygame.MOUSEBUTTONUP:
				instances.append((2*pygame.mouse.get_pos()[0],2*pygame.mouse.get_pos()[1]))
				displayImage(screen, px)

	return instances

def run_pygame(img,text):
	pygame.init()
	pilbmp = Image.fromarray(numpy.uint8(img))
	pilbmp.thumbnail((640,360),Image.ANTIALIAS)
	pilbmp.save('temp_input_img.bmp')
	positive = list()
	screen, px =setup('temp_input_img.bmp')
	salience = mainLoop(screen, px, positive, text)
	pygame.display.quit()
	
	return salience
##########################################################################----pygame----###########################################################

thresh = 0.7
duration = 0.25
prev_time = 0
IN_LOOP = False
TRAINED = False
MONITOR = False
OLD_TYPE = ''

for each_action in actionlist:
	if each_action['class'] == 'standby': ## if found standby signal(monitoring signal), create a list of monitoring script, which will be triggered after the visual cue is found.
		MONITOR = True
		monitor_actions = list()
		looking_for = each_action['looking_for']
		standby_clf = each_action['standby_clf']
		standby_img = each_action['wait_img']
	elif MONITOR:
		monitor_actions.append(each_action) ## create a list of monitoring script
	else:
		#get ss image
		if DEBUG:
			print each_action['class']
			print 'DB-2'
			print 'about to take SS in 5 seconds'
		
		if each_action['class'] != 'Typing':
			sleep(0.5)
			SS_img = numpy.asarray(GUIbot.screenshot())
			
		if DEBUG:
			print 'DB-1'
			plt.imshow(SS_img)
			plt.show()

		#create list of object to loop
		if each_action['class'] == 'instance_list':
			loop_img = each_action['removed_pt_img']
			if 'detector' in each_action:
				loop_clf = each_action['detector']
				loop_sup = each_action['supporters']
				sup_clf = each_action['supporter_clf']
				TRAINED=True
			else:
				saliences_list = each_action['saliences_list']

			instances = each_action['instances']
		elif each_action['class'] == 'start_loop':
			IN_LOOP = True
			if DEFINE:
				highXs,highYs = zip(*run_pygame(SS_img, 'Please click on instances you want to loop through.'))
			else:
				if TRAINED:
					found_sup = list()
					if len(sup_clf)>0:
						for each_sup_clf in sup_clf:
							PREDICT = utils.locateIMG(SS_img,each_sup_clf)
							flat_index=numpy.argmax(PREDICT)
							Y_max,X_max=numpy.unravel_index(flat_index,PREDICT.shape)
							found_sup.append((X_max,Y_max))
					else:
						for each_sup in loop_sup:
							size=30
							padded_img_SP = numpy.zeros(shape=(loop_img.shape[0]+(2*size),loop_img.shape[1]+(2*size),loop_img.shape[2]))
							padded_img_SP[size:-size,size:-size,:] = loop_img
							template = padded_img_SP[max(each_sup[1],0):min(each_sup[1]+(2*size)+1,padded_img_SP.shape[0]), \
													 max(each_sup[0],0):min(each_sup[0]+(2*size)+1,padded_img_SP.shape[1]),:]
							padded_img = numpy.zeros(shape=(SS_img.shape[0]+(2*size),SS_img.shape[1]+(2*size),SS_img.shape[2]))
							padded_img[size:-size,size:-size,:] = SS_img
							PREDICT = match_template(padded_img,template)[:,:,0] 
							flat_index=numpy.argmax(PREDICT)
							Y_max,X_max=numpy.unravel_index(flat_index,PREDICT.shape)
							found_sup.append((X_max,Y_max))

					PREDICT = utils.locateIMG_with_supporter(SS_img, found_sup, loop_clf)
					predict_nms=utils.nonMaxSuppress(PREDICT)
					highYs,highXs = numpy.nonzero(predict_nms>thresh)

				else:
					highXs = list()
					highYs = list()
					for indx_loop in range(len(instances)):
						PREDICT=numpy.zeros(SS_img.shape[:2])
						size=30
						padded_img_TP = numpy.zeros(shape=(loop_img.shape[0]+(2*size),loop_img.shape[1]+(2*size),loop_img.shape[2]))
						padded_img_TP[size:-size,size:-size,:] = loop_img
						template = padded_img_TP[max(instances[indx_loop][1],0):min(instances[indx_loop][1]+(2*size)+1,padded_img_TP.shape[0]), \
														 max(instances[indx_loop][0],0):min(instances[indx_loop][0]+(2*size)+1,padded_img_TP.shape[1]),:]
						padded_img = numpy.zeros(shape=(SS_img.shape[0]+(2*size),SS_img.shape[1]+(2*size),SS_img.shape[2]))
						padded_img[size:-size,size:-size,:] = SS_img
						PREDICT += (1.2 * match_template(padded_img,template)[:,:,0]) #weight the main template a little bit higher

						non = lambda s: s if s<0 else None
						mom = lambda s: max(0,s)

						for x,y in saliences_list[indx_loop]:
							template = padded_img_TP[max(y,0):min(y+(2*size)+1,loop_img.shape[0]), \
													 max(x,0):min(x+(2*size)+1,loop_img.shape[1]),:]
							pred = match_template(padded_img,template)[:,:,0]	 
							ox,oy = (instances[indx_loop][0]-x,instances[indx_loop][1]-y)
							shift_pred = numpy.zeros_like(pred)
							shift_pred[mom(oy):non(oy), mom(ox):non(ox)] = pred[mom(-oy):non(-oy), mom(-ox):non(-ox)]
							PREDICT += shift_pred

						flat_index=numpy.argmax(PREDICT)
						Y_max,X_max=numpy.unravel_index(flat_index,PREDICT.shape)
						highXs.append(X_max)
						highYs.append(Y_max)
						
				#### Sorting ####
				if numpy.std(highYs) < numpy.std(highXs):
					xy = zip(highXs,highYs)
					xy.sort()
					highXs = [x for x,y in xy]
					highYs = [y for x,y in xy]

				if DEBUG:
					plt.imshow(PREDICT)
					plt.show()
					print 'DB0'
					code.interact(local=dict(globals(),**locals()))
			
		elif each_action['class'] == 'end_loop':
			IN_LOOP = False
			print 'LOOP start ..found {0} instances'.format(len(highXs))

			for e_indx in range(len(highXs)):
				GUIbot.moveTo(highXs[e_indx],highYs[e_indx],duration=duration)

				for sub_a_indx,e_action in enumerate(LOOP_ACTIONS):
					if e_action['class'] == 'Typing':
							sleep(0.033)
							TYPE = e_action['typing']
							if TYPE != OLD_TYPE:
								OLD_TYPE = TYPE	
								print '\tLoop action : ' + 'type-> ' + TYPE
								
								if 'CTRL-' in TYPE:
									GUIbot.hotkey('ctrl',TYPE.replace('CTRL-','').lower())
								elif 'SHIFT-' in TYPE:
									GUIbot.hotkey('shift',TYPE.replace('SHIFT-','').lower())
								else:
									GUIbot.press(TYPE)

					else:
						print '\tLoop action : ' + e_action['class']
						if sub_a_indx > 0 :
							sleep(0.5)
							SS_img = numpy.asarray(GUIbot.screenshot())
							if 'action_clf' in e_action:
								PREDICT = utils.locateIMG(SS_img,each_sup_clf)
								flat_index=numpy.argmax(PREDICT)
								Y_max,X_max=numpy.unravel_index(flat_index,PREDICT.shape)
								
							else:
								PREDICT=numpy.zeros(SS_img.shape[:2])
								size=30
								padded_img_TP = numpy.zeros(shape=(e_action['removed_pt_img_str'].shape[0]+(2*size),e_action['removed_pt_img_str'].shape[1]+(2*size),e_action['removed_pt_img_str'].shape[2]))
								padded_img_TP[size:-size,size:-size,:] = e_action['removed_pt_img_str']
								template = padded_img_TP[max(e_action['pos_start'][1],0):min(e_action['pos_start'][1]+(2*size)+1,padded_img_TP.shape[0]), \
														 max(e_action['pos_start'][0],0):min(e_action['pos_start'][0]+(2*size)+1,padded_img_TP.shape[1]),:]
								padded_img = numpy.zeros(shape=(SS_img.shape[0]+(2*size),SS_img.shape[1]+(2*size),SS_img.shape[2]))
								padded_img[size:-size,size:-size,:] = SS_img

								non = lambda s: s if s<0 else None
								mom = lambda s: max(0,s)

								PREDICT += (1.2 * match_template(padded_img,template)[:,:,0]) #weight the main template a little bit higher
								for x,y in e_action['saliences']:
									template = padded_img_TP[max(y,0):min(y+(2*size)+1,padded_img_TP.shape[0]), \
															 max(x,0):min(x+(2*size)+1,padded_img_TP.shape[1]),:]
									pred = match_template(padded_img,template)[:,:,0]	 
									ox,oy = (e_action['pos_start'][0]-x,e_action['pos_start'][1]-y)
									shift_pred = numpy.zeros_like(pred)
									shift_pred[mom(oy):non(oy), mom(ox):non(ox)] = pred[mom(-oy):non(-oy), mom(-ox):non(-ox)]
									PREDICT += shift_pred
									
								flat_index=numpy.argmax(PREDICT)
								Y_max,X_max=numpy.unravel_index(flat_index,PREDICT.shape)
							
							if QUERY:
								V_max = numpy.max(PREDICT)
								PREDICT_nms = numpy.empty_like(PREDICT)
								PREDICT_nms[:] = PREDICT
								PREDICT_nms = utils.nonMaxSuppress(PREDICT_nms)
								PREDICT_nms[Y_max-10:Y_max+10,X_max-10:X_max+10] = 0
								rows,cols = numpy.nonzero(PREDICT_nms>(0.85*V_max))
								if len(rows) > 0:
									bbimg = numpy.empty_like(SS_img)
									bbimg[:] = SS_img

									for d_indx in range(len(rows)):
										bbimg = utils.draw_bounding_boxR(bbimg,cols[d_indx],rows[d_indx])

									bbimg = utils.draw_bounding_boxB(bbimg,X_max,Y_max)	
									addition = run_pygame(bbimg*255,'Please select the target you want to {0}. BLUE is the most possible target.'.format(e_action['class']))

									if len(addition) == 1:
										X_max,Y_max = addition[0]
										e_action['saliences'].append((X_max,Y_max))
									elif len(addition) > 1:
										print 'please select only one instance'

							if DEBUG:
								plt.imshow(PREDICT)
								plt.show()
								print 'DB1'
								code.interact(local=dict(globals(),**locals()))

							GUIbot.moveTo(X_max,Y_max,duration=duration)
							
						try:
							if e_action['class'] == 'Click':
								GUIbot.click()
								prev_time = e_action['end']
								sleep(0.033)
								OLD_TYPE = ''
							elif e_action['class'] == 'RClick':
								GUIbot.rightClick()
								prev_time = e_action['end']
								sleep(0.033)
								OLD_TYPE = ''
							elif e_action['class'] == 'DoubleClick':
								GUIbot.doubleClick()
								prev_time = e_action['end']
								sleep(5)
								OLD_TYPE = ''
							elif e_action['class'] == 'ClickDrag':
								if 'same_as_start' not in e_action:
									PREDICT=numpy.zeros(SS_img.shape[:2])
									size=30
									padded_img_TP = numpy.zeros(shape=(e_action['removed_pt_img_end'].shape[0]+(2*size),e_action['removed_pt_img_end'].shape[1]+(2*size),e_action['removed_pt_img_end'].shape[2]))
									padded_img_TP[size:-size,size:-size,:] = e_action['removed_pt_img_end']
									template = padded_img_TP[max(e_action['pos_end'][1],0):min(e_action['pos_end'][1]+(2*size)+1,padded_img_TP.shape[0]), \
															 max(e_action['pos_end'][0],0):min(e_action['pos_end'][0]+(2*size)+1,padded_img_TP.shape[1]),:]
									padded_img = numpy.zeros(shape=(SS_img.shape[0]+(2*size),SS_img.shape[1]+(2*size),SS_img.shape[2]))
									padded_img[size:-size,size:-size,:] = SS_img

									non = lambda s: s if s<0 else None
									mom = lambda s: max(0,s)

									PREDICT += (1.2 * match_template(padded_img,template)[:,:,0]) #weight the main template a little bit higher

									for x,y in e_action['saliences_end']:
										template = padded_img_TP[max(y,0):min(y+(2*size)+1,padded_img_TP.shape[0]), \
																 max(x,0):min(x+(2*size)+1,padded_img_TP.shape[1]),:]
										pred = match_template(padded_img,template)[:,:,0]	 
										ox,oy = (e_action['pos_end'][0]-x,e_action['pos_end'][1]-y)
										shift_pred = numpy.zeros_like(pred)
										shift_pred[mom(oy):non(oy), mom(ox):non(ox)] = pred[mom(-oy):non(-oy), mom(-ox):non(-ox)]
										PREDICT += shift_pred
									
									flat_index=numpy.argmax(PREDICT)
									Y_max,X_max=numpy.unravel_index(flat_index,PREDICT.shape)
									
									if DEBUG:
										plt.imshow(PREDICT)
										plt.show()
										print 'DB2'

								GUIbot.dragTo(X_max,Y_max,duration=0.033*(e_action['end']-e_action['start']))
								prev_time = e_action['end']
								sleep(0.033)
								OLD_TYPE = ''

						except RuntimeError:
							raise SystemExit('error at line:341')
			print 'LOOP end'

		else:
			if not IN_LOOP:
				if each_action['class'] == 'Typing':
					sleep(0.033)
					TYPE = each_action['typing']
					if TYPE != OLD_TYPE:
						OLD_TYPE = TYPE	
						print 'type-> ' + TYPE
						if 'CTRL-' in TYPE:
							GUIbot.hotkey('ctrl',TYPE.replace('CTRL-','').lower())
						elif 'SHIFT-' in TYPE:
							GUIbot.hotkey('shift',TYPE.replace('SHIFT-','').lower())
						else:
							GUIbot.press(TYPE)
				else:
					if 'action_clf' in each_action:
						PREDICT = utils.locateIMG(SS_img,each_action['action_clf'])
						flat_index=numpy.argmax(PREDICT)
						Y_max,X_max=numpy.unravel_index(flat_index,PREDICT.shape)
					else:
						PREDICT=numpy.zeros(SS_img.shape[:2])
						size=30
						padded_img_TP = numpy.zeros(shape=(each_action['removed_pt_img_str'].shape[0]+(2*size),each_action['removed_pt_img_str'].shape[1]+(2*size),each_action['removed_pt_img_str'].shape[2]))
						padded_img_TP[size:-size,size:-size,:] = each_action['removed_pt_img_str']
						template = padded_img_TP[max(each_action['pos_start'][1],0):min(each_action['pos_start'][1]+(2*size)+1,padded_img_TP.shape[0]), \
												 max(each_action['pos_start'][0],0):min(each_action['pos_start'][0]+(2*size)+1,padded_img_TP.shape[1]),:]
						padded_img = numpy.zeros(shape=(SS_img.shape[0]+(2*size),SS_img.shape[1]+(2*size),SS_img.shape[2]))
						padded_img[size:-size,size:-size,:] = SS_img

						non = lambda s: s if s<0 else None
						mom = lambda s: max(0,s)

						PREDICT += (1.2 * match_template(padded_img,template)[:,:,0]) #weight the main template a little bit higher
						for x,y in each_action['saliences']:
							template = padded_img_TP[max(y,0):min(y+(2*size)+1,padded_img_TP.shape[0]), \
													 max(x,0):min(x+(2*size)+1,padded_img_TP.shape[1]),:]
							pred = match_template(padded_img,template)[:,:,0]	 
							ox,oy = (each_action['pos_start'][0]-x,each_action['pos_start'][1]-y)
							shift_pred = numpy.zeros_like(pred)
							shift_pred[mom(oy):non(oy), mom(ox):non(ox)] = pred[mom(-oy):non(-oy), mom(-ox):non(-ox)]
							PREDICT += shift_pred
							
						flat_index=numpy.argmax(PREDICT)
						Y_max,X_max=numpy.unravel_index(flat_index,PREDICT.shape)

					if QUERY:
								V_max = numpy.max(PREDICT)
								PREDICT_nms = numpy.empty_like(PREDICT)
								PREDICT_nms[:] = PREDICT
								PREDICT_nms = utils.nonMaxSuppress(PREDICT_nms)
								PREDICT_nms[Y_max-10:Y_max+10,X_max-10:X_max+10] = 0
								rows,cols = numpy.nonzero(PREDICT_nms>(0.85*V_max))

								if len(rows) > 0:
									bbimg = numpy.empty_like(SS_img)
									bbimg[:] = SS_img
									
									for d_indx in range(len(rows)):
										bbimg = utils.draw_bounding_boxR(bbimg,cols[d_indx],rows[d_indx])
									bbimg = utils.draw_bounding_boxB(bbimg,X_max,Y_max)

									addition = run_pygame(bbimg,'Please select the target you want to {0}. BLUE is the most possible target.'.format(each_action['class']))

									if len(addition) == 1:
										X_max,Y_max = addition[0]
										each_action['saliences'].append((X_max,Y_max))
									else:
										print 'please select only one instance'

					if DEBUG:
						plt.imshow(PREDICT)
						plt.show()
						print 'DB3'

					#move mouse cursor to the start position
					GUIbot.moveTo(X_max,Y_max,duration=duration)
					print each_action['class']
					try:
						if each_action['class'] == 'Click':
							GUIbot.click()
							sleep(0.033*(each_action['start']-prev_time))
							prev_time = each_action['end']
							OLD_TYPE = ''
						elif each_action['class'] == 'DoubleClick':
							GUIbot.doubleClick()
							sleep(0.033*(each_action['start']-prev_time))
							prev_time = each_action['end']
							sleep(5)
							OLD_TYPE = ''
						elif each_action['class'] == 'RClick':
							GUIbot.rightClick()
							sleep(0.033*(each_action['start']-prev_time))
							prev_time = each_action['end']
							OLD_TYPE = ''
						elif each_action['class'] == 'ClicksRename':
							GUIbot.click()
							sleep(0.5)
							GUIbot.click()
							sleep(0.033*(each_action['start']-prev_time))
							prev_time = each_action['end']
							OLD_TYPE = ''
						elif each_action['class'] == 'ClickDrag':
							if 'same_as_start' not in each_action:
								PREDICT=numpy.zeros(SS_img.shape[:2])
								size=30
								padded_img_TP = numpy.zeros(shape=(each_action['removed_pt_img_end'].shape[0]+(2*size),each_action['removed_pt_img_end'].shape[1]+(2*size),each_action['removed_pt_img_end'].shape[2]))
								padded_img_TP[size:-size,size:-size,:] = each_action['removed_pt_img_end']
								template = padded_img_TP[max(each_action['pos_end'][1],0):min(each_action['pos_end'][1]+(2*size)+1,padded_img_TP.shape[0]), \
														 max(each_action['pos_end'][0],0):min(each_action['pos_end'][0]+(2*size)+1,padded_img_TP.shape[1]),:]
								padded_img = numpy.zeros(shape=(SS_img.shape[0]+(2*size),SS_img.shape[1]+(2*size),SS_img.shape[2]))
								padded_img[size:-size,size:-size,:] = SS_img

								non = lambda s: s if s<0 else None
								mom = lambda s: max(0,s)

								PREDICT += (1.2 * match_template(padded_img,template)[:,:,0]) #weight the main template a little bit higher

								for x,y in each_action['saliences_end']:
									template = padded_img_TP[max(y,0):min(y+(2*size)+1,padded_img_TP.shape[0]), \
															 max(x,0):min(x+(2*size)+1,padded_img_TP.shape[1]),:]
									pred = match_template(padded_img,template)[:,:,0]	 
									ox,oy = (each_action['pos_end'][0]-x,each_action['pos_end'][1]-y)
									shift_pred = numpy.zeros_like(pred)
									shift_pred[mom(oy):non(oy), mom(ox):non(ox)] = pred[mom(-oy):non(-oy), mom(-ox):non(-ox)]
									PREDICT += shift_pred
									
								flat_index=numpy.argmax(PREDICT)
								Y_max,X_max=numpy.unravel_index(flat_index,PREDICT.shape)

								if DEBUG:
									plt.imshow(pred)
									plt.show()
									print 'DB4'
							
							GUIbot.dragTo(X_max,Y_max,duration=0.033*(each_action['end']-each_action['start']))
							sleep(0.033*(each_action['start']-prev_time))
							prev_time = each_action['end']
							OLD_TYPE = ''

					except RuntimeError:
						raise SystemExit('error at line:480')

			else:
				if collect_action:
					LOOP_ACTIONS.append(each_action)

print 'DONE!!'

#### for monitoring task
if MONITOR:
	print 'start monitoring'

	while True:
		sleep(3)
		size = 30
		SS_img = numpy.asarray(GUIbot.screenshot())
		template = standby_img[max(looking_for[1]-size,0):min(looking_for[1]+(size)+1,standby_img.shape[0]), \
							   max(looking_for[0]-size,0):min(looking_for[0]+(size)+1,standby_img.shape[1]),:]
		to_search = SS_img[max(looking_for[1]-size,0):min(looking_for[1]+(size)+1,SS_img.shape[0]), \
						   max(looking_for[0]-size,0):min(looking_for[0]+(size)+1,SS_img.shape[1]),:]
		matched = standby_clf.predict_proba(to_search.flatten().reshape(1,-1))[0,1]
		print matched,

		if matched > 0.5:
			found = True
		else:
			found = False

		if found:
			prev_time = 0
			IN_LOOP = False
			TRAINED = False
			OLD_TYPE = ''

			for indx,each_action in enumerate(monitor_actions):
				if DEBUG:
					print each_action['class']
					print 'DB-2'
					print 'about to take SS in 5 seconds'

				if each_action['class'] != 'Typing' and indx > 0:
					sleep(0.1)
					SS_img = numpy.asarray(GUIbot.screenshot())

				if DEBUG:
					print 'DB-1'
					plt.imshow(SS_img)
					plt.show()

				#create list of object to loop
				if each_action['class'] == 'instance_list':
					loop_img = each_action['removed_pt_img']

					if 'detector' in each_action:
						loop_clf = each_action['detector']
						TRAINED=True
					else:
						saliences_list = each_action['saliences_list']

					instances = each_action['instances']

				elif each_action['class'] == 'start_loop':
					IN_LOOP = True

					if DEFINE:
						highXs,highYs = zip(*run_pygame(SS_img, 'Please click on instances you want to loop over.'))
					else:

						if TRAINED:
							PREDICT = utils.locateIMG(SS_img,loop_clf)
							predict_nms=utils.nonMaxSuppress(PREDICT)
							highYs,highXs = numpy.nonzero(predict_nms>thresh)
						else:
							highXs = list()
							highYs = list()
							for indx_loop in range(len(instances)):
								PREDICT=numpy.zeros(SS_img.shape[:2])
								size=30
								padded_img_TP = numpy.zeros(shape=(loop_img.shape[0]+(2*size),loop_img.shape[1]+(2*size),loop_img.shape[2]))
								padded_img_TP[size:-size,size:-size,:] = loop_img
								template = padded_img_TP[max(instances[indx_loop][1],0):min(instances[indx_loop][1]+(2*size)+1,padded_img_TP.shape[0]), \
																 max(instances[indx_loop][0],0):min(instances[indx_loop][0]+(2*size)+1,padded_img_TP.shape[1]),:]
								padded_img = numpy.zeros(shape=(SS_img.shape[0]+(2*size),SS_img.shape[1]+(2*size),SS_img.shape[2]))
								padded_img[size:-size,size:-size,:] = SS_img
								PREDICT += (1.2 * match_template(padded_img,template)[:,:,0]) #weight the main template a little bit higher

								non = lambda s: s if s<0 else None
								mom = lambda s: max(0,s)

								for x,y in saliences_list[indx_loop]:
									template = padded_img_TP[max(y,0):min(y+(2*size)+1,loop_img.shape[0]), \
															 max(x,0):min(x+(2*size)+1,loop_img.shape[1]),:]
									pred = match_template(padded_img,template)[:,:,0]	 
									ox,oy = (instances[indx_loop][0]-x,instances[indx_loop][1]-y)
									shift_pred = numpy.zeros_like(pred)
									shift_pred[mom(oy):non(oy), mom(ox):non(ox)] = pred[mom(-oy):non(-oy), mom(-ox):non(-ox)]
									PREDICT += shift_pred

								flat_index=numpy.argmax(PREDICT)
								Y_max,X_max=numpy.unravel_index(flat_index,PREDICT.shape)
								highXs.append(X_max)
								highYs.append(Y_max)

						if DEBUG:
							plt.imshow(PREDICT)
							plt.show()
							print 'DB0'

				elif each_action['class'] == 'end_loop':
					IN_LOOP = False
					print 'LOOP start'
					for e_indx in range(len(highXs)):
						GUIbot.moveTo(highXs[e_indx],highYs[e_indx],duration=duration)
						for sub_a_indx,e_action in enumerate(LOOP_ACTIONS):
							if e_action['class'] == 'Typing':
									TYPE = e_action['typing']
									if TYPE != OLD_TYPE:
										OLD_TYPE = TYPE	
										print '\tLoop action : ' + 'type-> ' + TYPE
										if 'CTRL-' in TYPE:
											GUIbot.hotkey('ctrl',TYPE.replace('CTRL-','').lower())
										elif 'SHIFT-' in TYPE:
											GUIbot.hotkey('shift',TYPE.replace('SHIFT-','').lower())
										else:
											GUIbot.press(TYPE)

							else:
								print '\tLoop action : ' + e_action['class']
								if sub_a_indx > 0 :
									sleep(0.5)
									SS_img = numpy.asarray(GUIbot.screenshot())
									PREDICT=numpy.zeros(SS_img.shape[:2])
									size=30
									padded_img_TP = numpy.zeros(shape=(e_action['removed_pt_img_str'].shape[0]+(2*size),e_action['removed_pt_img_str'].shape[1]+(2*size),e_action['removed_pt_img_str'].shape[2]))
									padded_img_TP[size:-size,size:-size,:] = e_action['removed_pt_img_str']
									template = padded_img_TP[max(e_action['pos_start'][1],0):min(e_action['pos_start'][1]+(2*size)+1,padded_img_TP.shape[0]), \
															 max(e_action['pos_start'][0],0):min(e_action['pos_start'][0]+(2*size)+1,padded_img_TP.shape[1]),:]
									padded_img = numpy.zeros(shape=(SS_img.shape[0]+(2*size),SS_img.shape[1]+(2*size),SS_img.shape[2]))
									padded_img[size:-size,size:-size,:] = SS_img

									non = lambda s: s if s<0 else None
									mom = lambda s: max(0,s)

									PREDICT += (1.2 * match_template(padded_img,template)[:,:,0]) #weight the main template a little bit higher
									for x,y in e_action['saliences']:
										template = padded_img_TP[max(y,0):min(y+(2*size)+1,padded_img_TP.shape[0]), \
																 max(x,0):min(x+(2*size)+1,padded_img_TP.shape[1]),:]
										pred = match_template(padded_img,template)[:,:,0]	 
										ox,oy = (e_action['pos_start'][0]-x,e_action['pos_start'][1]-y)
										shift_pred = numpy.zeros_like(pred)
										shift_pred[mom(oy):non(oy), mom(ox):non(ox)] = pred[mom(-oy):non(-oy), mom(-ox):non(-ox)]
										PREDICT += shift_pred
										
									flat_index=numpy.argmax(PREDICT)
									Y_max,X_max=numpy.unravel_index(flat_index,PREDICT.shape)

									if DEBUG:
										plt.imshow(PREDICT)
										plt.show()
										print 'DB1'
										code.interact(local=dict(globals(),**locals()))
									GUIbot.moveTo(X_max,Y_max,duration=duration)

								try:
									if e_action['class'] == 'Click':
										sleep(0.033)
										GUIbot.click()
										prev_time = e_action['end']
										OLD_TYPE = ''
									elif e_action['class'] == 'RClick':
										sleep(0.033)
										GUIbot.rightClick()
										prev_time = e_action['end']
										OLD_TYPE = ''
									elif e_action['class'] == 'DoubleClick':
										sleep(0.033)
										GUIbot.doubleClick()
										prev_time = e_action['end']
										sleep(5)
										OLD_TYPE = ''
									elif e_action['class'] == 'ClicksRename':
										sleep(0.033)
										GUIbot.click()
										sleep(0.5)
										GUIbot.click()
										prev_time = e_action['end']
										OLD_TYPE = ''
									elif e_action['class'] == 'ClickDrag':
										PREDICT=numpy.zeros(SS_img.shape[:2])
										size=30
										padded_img_TP = numpy.zeros(shape=(e_action['removed_pt_img_end'].shape[0]+(2*size),e_action['removed_pt_img_end'].shape[1]+(2*size),e_action['removed_pt_img_end'].shape[2]))
										padded_img_TP[size:-size,size:-size,:] = e_action['removed_pt_img_end']
										template = padded_img_TP[max(e_action['pos_end'][1],0):min(e_action['pos_end'][1]+(2*size)+1,padded_img_TP.shape[0]), \
																 max(e_action['pos_end'][0],0):min(e_action['pos_end'][0]+(2*size)+1,padded_img_TP.shape[1]),:]
										padded_img = numpy.zeros(shape=(SS_img.shape[0]+(2*size),SS_img.shape[1]+(2*size),SS_img.shape[2]))
										padded_img[size:-size,size:-size,:] = SS_img
										non = lambda s: s if s<0 else None
										mom = lambda s: max(0,s)
										PREDICT += (1.2 * match_template(padded_img,template)[:,:,0]) #weight the main template a little bit higher

										for x,y in e_action['saliences_end']:
											template = padded_img_TP[max(y,0):min(y+(2*size)+1,padded_img_TP.shape[0]), \
																	 max(x,0):min(x+(2*size)+1,padded_img_TP.shape[1]),:]
											pred = match_template(padded_img,template)[:,:,0]	 
											ox,oy = (e_action['pos_end'][0]-x,e_action['pos_end'][1]-y)
											shift_pred = numpy.zeros_like(pred)
											shift_pred[mom(oy):non(oy), mom(ox):non(ox)] = pred[mom(-oy):non(-oy), mom(-ox):non(-ox)]
											PREDICT += shift_pred
										
										flat_index=numpy.argmax(PREDICT)
										Y_max,X_max=numpy.unravel_index(flat_index,PREDICT.shape)
										if DEBUG:
											plt.imshow(PREDICT)
											plt.show()
											print 'DB2'

										sleep(0.033)
										GUIbot.dragTo(X_max,Y_max,duration=0.033*(e_action['end']-e_action['start']))
										prev_time = e_action['end']
										OLD_TYPE = ''
								except RuntimeError:
									raise SystemExit('error at line:700')
					print 'LOOP end'

				else:
					if not IN_LOOP:
						if each_action['class'] == 'Typing':
							TYPE = each_action['typing']
							if TYPE != OLD_TYPE:
								OLD_TYPE = TYPE	
								print 'type-> ' + TYPE

								if 'CTRL-' in TYPE:
									GUIbot.hotkey('ctrl',TYPE.replace('CTRL-','').lower())
								elif 'SHIFT-' in TYPE:
									GUIbot.hotkey('shift',TYPE.replace('SHIFT-','').lower())
								else:
									GUIbot.press(TYPE)

						else:
							if 'same_obj_as_standby' in each_action:
									X_max = each_action['pos_start'][0]
									Y_max = each_action['pos_start'][1]
							else:
								PREDICT=numpy.zeros(SS_img.shape[:2])
								size=30
								padded_img_TP = numpy.zeros(shape=(each_action['removed_pt_img_str'].shape[0]+(2*size),each_action['removed_pt_img_str'].shape[1]+(2*size),each_action['removed_pt_img_str'].shape[2]))
								padded_img_TP[size:-size,size:-size,:] = each_action['removed_pt_img_str']
								template = padded_img_TP[max(each_action['pos_start'][1],0):min(each_action['pos_start'][1]+(2*size)+1,padded_img_TP.shape[0]), \
													 max(each_action['pos_start'][0],0):min(each_action['pos_start'][0]+(2*size)+1,padded_img_TP.shape[1]),:]
								padded_img = numpy.zeros(shape=(SS_img.shape[0]+(2*size),SS_img.shape[1]+(2*size),SS_img.shape[2]))
								padded_img[size:-size,size:-size,:] = SS_img

								non = lambda s: s if s<0 else None
								mom = lambda s: max(0,s)

								PREDICT += (1.2 * match_template(padded_img,template)[:,:,0]) #weight the main template a little bit higher

								for x,y in each_action['saliences']:
									template = padded_img_TP[max(y,0):min(y+(2*size)+1,padded_img_TP.shape[0]), \
															 max(x,0):min(x+(2*size)+1,padded_img_TP.shape[1]),:]
									pred = match_template(padded_img,template)[:,:,0]	 
									ox,oy = (each_action['pos_start'][0]-x,each_action['pos_start'][1]-y)
									shift_pred = numpy.zeros_like(pred)
									shift_pred[mom(oy):non(oy), mom(ox):non(ox)] = pred[mom(-oy):non(-oy), mom(-ox):non(-ox)]
									PREDICT += shift_pred
									
								flat_index=numpy.argmax(PREDICT)
								Y_max,X_max=numpy.unravel_index(flat_index,PREDICT.shape)

								if DEBUG:
									plt.imshow(PREDICT)
									plt.show()
									print 'DB3'

							#move mouse cursor to the start position
							GUIbot.moveTo(X_max,Y_max,duration=duration)
							print each_action['class']

							try:
								if each_action['class'] == 'Click':
									GUIbot.click()
									prev_time = each_action['end']
									OLD_TYPE = ''
								elif each_action['class'] == 'DoubleClick':
									GUIbot.doubleClick()
									prev_time = each_action['end']
									sleep(5)
									OLD_TYPE = ''
								elif each_action['class'] == 'RClick':
									GUIbot.rightClick()
									prev_time = each_action['end']
									OLD_TYPE = ''
								elif each_action['class'] == 'ClicksRename':
									GUIbot.click()
									sleep(0.5)
									GUIbot.click()
									prev_time = each_action['end']
									OLD_TYPE = ''
								elif each_action['class'] == 'ClickDrag':
									PREDICT=numpy.zeros(SS_img.shape[:2])
									size=30
									padded_img_TP = numpy.zeros(shape=(each_action['removed_pt_img_end'].shape[0]+(2*size),each_action['removed_pt_img_end'].shape[1]+(2*size),each_action['removed_pt_img_end'].shape[2]))
									padded_img_TP[size:-size,size:-size,:] = each_action['removed_pt_img_end']
									template = padded_img_TP[max(each_action['pos_end'][1],0):min(each_action['pos_end'][1]+(2*size)+1,padded_img_TP.shape[0]), \
															 max(each_action['pos_end'][0],0):min(each_action['pos_end'][0]+(2*size)+1,padded_img_TP.shape[1]),:]
									padded_img = numpy.zeros(shape=(SS_img.shape[0]+(2*size),SS_img.shape[1]+(2*size),SS_img.shape[2]))
									padded_img[size:-size,size:-size,:] = SS_img

									non = lambda s: s if s<0 else None
									mom = lambda s: max(0,s)

									PREDICT += (1.2 * match_template(padded_img,template)[:,:,0]) #weight the main template a little bit higher

									for x,y in each_action['saliences_end']:
										template = padded_img_TP[max(y,0):min(y+(2*size)+1,padded_img_TP.shape[0]), \
																 max(x,0):min(x+(2*size)+1,padded_img_TP.shape[1]),:]
										pred = match_template(padded_img,template)[:,:,0]	 
										ox,oy = (each_action['pos_end'][0]-x,each_action['pos_end'][1]-y)
										shift_pred = numpy.zeros_like(pred)
										shift_pred[mom(oy):non(oy), mom(ox):non(ox)] = pred[mom(-oy):non(-oy), mom(-ox):non(-ox)]
										PREDICT += shift_pred
										
									flat_index=numpy.argmax(PREDICT)
									Y_max,X_max=numpy.unravel_index(flat_index,PREDICT.shape)

									if DEBUG:
										plt.imshow(pred)
										plt.show()
										print 'DB4'

									GUIbot.dragTo(X_max,Y_max,duration=0.033*(each_action['end']-each_action['start']))
									prev_time = each_action['end']
									OLD_TYPE = ''

							except RuntimeError:
								raise SystemExit('error at line:811')

					else:
						if collect_action:
							LOOP_ACTIONS.append(each_action)

win32gui.ShowWindow(Minimize,win32con.SW_MAXIMIZE)

print 'press CTRL+D to save updated action or type exit() to exit without saving'
with open(action_path+'updated_prepared_actionlist.pickle', 'wb') as handle:
	pickle.dump([actionlist,LOOP_ACTIONS], handle)