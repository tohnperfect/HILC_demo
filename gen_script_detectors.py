"""generate script and interactively train a detector for """

import pickle
import argparse
import os
import random
import string
import matplotlib.pyplot as plt
import numpy
from skimage.feature import match_template

import utils
from genHTML import *

import pygame
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument('--p', type=str, help='path to action list', default='')
parser.add_argument('--s', type=str, help='path to save directory  default is empty string', default='')
parser.add_argument('--c', type=bool, help='use classifier (if --c=False use pattern matching)', default=True)
parser.add_argument('--t', type=bool, help='enter tedious mode , ask every question', default=False)
parser.add_argument('--d', type=bool, help='enter debug mode', default=False)
args = parser.parse_args()

action_path = args.p
save_path = args.s
TRAIN_CLASSIFIER = args.c
DEBUG = args.d

TEDIOUS = args.t
questions = list()
with open('pygame_question_genscript.txt','r') as input_file:
	for line in input_file:
		questions.append(line)

path_to_imgs = action_path + 'imgs\\'
path_to_save = action_path+save_path+'script_visualize\\'

if not os.path.exists(path_to_save):	
	os.mkdir(path_to_save)

result_path = action_path + 'test_userinput\\RESULT'
if not os.path.exists(action_path + 'test_userinput\\'):
	os.mkdir(action_path + 'test_userinput\\') 
if not os.path.exists(result_path):	
	os.mkdir(result_path)

with open(action_path+'transcript_actionlist.pickle', 'rb') as handle:
	[transcript,actionlist] = pickle.load(handle)

##########################################################################----pygame----###########################################################
# def displayImage( screen, px):
# 	pygame.draw.rect( screen, (0,255,0), pygame.Rect(pygame.mouse.get_pos()[0]-15, pygame.mouse.get_pos()[1]-15, 31,31), 2)
# 	pygame.display.flip()

# def displayImage2( screen, px):
# 	pygame.draw.rect( screen, (0,0,0), pygame.Rect(pygame.mouse.get_pos()[0]-15, pygame.mouse.get_pos()[1]-15, 31,31))
# 	pygame.display.flip()

# def displayImage3( screen, px):
# 	pygame.draw.rect( screen, (255,255,0), pygame.Rect(pygame.mouse.get_pos()[0]-15, pygame.mouse.get_pos()[1]-15, 31,31), 3)
# 	pygame.display.flip()

def displayBoxes( screen, px, positives, negatives, unsures, supporters):
	screen.blit(px, px.get_rect())
	for coor in positives:
		pygame.draw.rect( screen, (0,0,255), pygame.Rect((0.5*coor[0])-15, (0.5*coor[1])-15, 31,31), 2) #blue real positive
	for coor in unsures:
		pygame.draw.rect( screen, (250,200,20), pygame.Rect((0.5*coor[0])-15, (0.5*coor[1])-15, 31,31), 1) #yellow unsuer positive
	for coor in supporters:
		pygame.draw.rect( screen, (0,255,0), pygame.Rect((0.5*coor[0])-15, (0.5*coor[1])-15, 31,31), 3) #green supporter
	for coor in negatives:
		pygame.draw.rect( screen, (255,0,0), pygame.Rect((0.5*coor[0])-15, (0.5*coor[1])-15, 31,31), 2) #red negatives
	pygame.display.flip()

def setup(path):
	px = pygame.image.load(path)
	screen = pygame.display.set_mode( px.get_rect()[2:] )
	screen.blit(px, px.get_rect())
	pygame.display.flip()
	return screen, px

def mainLoop(screen, px, instances, txt, mode = 'pos'):

	bottomright = None
	runProgram = True
	pressL = False
	pressR = False
	pygame.display.set_caption(txt) 
	while runProgram:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				runProgram = False
				print 'quit!!'
			elif event.type == pygame.MOUSEBUTTONDOWN:
				if pygame.mouse.get_pressed()[0] == 1:
					pressL = True
				elif pygame.mouse.get_pressed()[2] == 1:
					pressR = True
			elif event.type == pygame.MOUSEBUTTONUP:
				if pressL:
					pressL = False
					instances.append((2*pygame.mouse.get_pos()[0],2*pygame.mouse.get_pos()[1]))

					displayBoxes(screen, px, positives=[], negatives=[], unsures=instances, supporters=[])
				elif pressR:
					pressR= False

					ininsta = check_if_in(pygame.mouse.get_pos(),instances)
					if ininsta is not False:
						instances.remove(ininsta)

					displayBoxes(screen, px, positives=[], negatives=[], unsures=instances, supporters=[])
				else:
					'do noting'

	return instances

def check_if_in(point,mainlist,size=30):
	for each in mainlist:
		if 2*point[0]>each[0]-size and 2*point[0]<each[0]+size and 2*point[1]>each[1]-size and 2*point[1]<each[1]+size:
			return each

	return False


def mainLoop_v2(screen, px, positives,negatives,unsures, supporters, txt):

	runProgram = True
	pygame.display.set_caption(txt)
	pressL = False
	pressR = False
	REFRESH = False
	loop_instances_clf = False
	while runProgram:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				runProgram = False
				print 'quit!!'
				if not loop_instances_clf :
					loop_instances_clf = utils.learn_img_classifier_with_supporter(training_img,positives,hardnegatives=negatives,supporters=supporters)
			elif event.type == pygame.MOUSEBUTTONDOWN:
				if pygame.mouse.get_pressed()[0] == 1:
					pressL = True
				elif pygame.mouse.get_pressed()[2] == 1:
					pressR = True
			elif event.type == pygame.MOUSEBUTTONUP:
				if pressL:
					pressL = False
					positives.append((2*pygame.mouse.get_pos()[0],2*pygame.mouse.get_pos()[1]))
					displayBoxes(screen, px, positives, negatives, unsures, supporters)
				elif pressR:
					pressR= False
					inpos = check_if_in(pygame.mouse.get_pos(),positives)
					if inpos is not False:
						positives.remove(inpos)
						negatives.append(inpos)
					else:
						inneg = check_if_in(pygame.mouse.get_pos(),negatives)
						if inneg is not False:
							negatives.remove(inneg)
							supporters.append(inneg)
						else:
							insup = check_if_in(pygame.mouse.get_pos(),supporters)
							if insup is not False:
								supporters.remove(insup)
								unsures.append(insup)
							else:
								inuns = check_if_in(pygame.mouse.get_pos(),unsures)
								if inuns is not False:
									unsures.remove(inuns)
									positives.append(inuns)
					displayBoxes(screen, px, positives, negatives, unsures, supporters)
				else:
					pressL = False
					pressR = False
			elif event.type == pygame.KEYDOWN:
				if event.key == pygame.K_F4:
					REFRESH = True
			elif event.type == pygame.KEYUP:
				if REFRESH:
					REFRESH = False
					print 'Start Training'
					loop_instances_clf = utils.learn_img_classifier_with_supporter(training_img,positives,hardnegatives=negatives,supporters=supporters)
					predictI = utils.locateIMG_with_supporter(training_img, supporters, loop_instances_clf)
					predict = numpy.empty_like(predictI)
					fig = plt.figure()
					ax = fig.add_subplot(111)
					ax.imshow(predictI)
					numrows, numcols = predictI.shape

					def format_coord(x, y):
						col = int(x+0.5)
						row = int(y+0.5)
						if col>=0 and col<numcols and row>=0 and row<numrows:
							z = predictI[row,col]
							return 'x=%1.4f, y=%1.4f, z=%1.4f'%(x, y, z)
						else:
							return 'x=%1.4f, y=%1.4f'%(x, y)

					ax.format_coord = format_coord
					plt.show()
					thresh = 0.7
					predict[:] = predictI
					rowsI,colsI = utils.nonMaxSuppress_nonZero(predict,thresh=thresh)
					unsures = zip(colsI,rowsI)
					displayBoxes(screen, px, positives, negatives, unsures, supporters)
					print '...done'

	return positives, negatives, unsures, supporters,loop_instances_clf

def run_pygame_v2(img,positives,unsures,negatives = list()):
	pygame.init()
	pilbmp = Image.fromarray(numpy.uint8(img))
	pilbmp.thumbnail((img.shape[1]/2,img.shape[0]/2),Image.ANTIALIAS)
	pilbmp.save('temp_input_img.bmp')
	supporters = list()
	screen, px = setup('temp_input_img.bmp')
	displayBoxes(screen, px, positives, negatives, unsures, supporters)
	print 'Question? : '
	print questions[1]
	positives,negatives,unsures,supporters, clf = mainLoop_v2(screen, px, positives, negatives, unsures, supporters, questions[1])
	pygame.display.quit()

	return positives,negatives,unsures,supporters,clf

def run_pygame(img):
	pygame.init()
	pilbmp = Image.fromarray(numpy.uint8(img*255))
	pilbmp.save('temp_input_img.bmp')
	positive=list()
	screen, px = setup('temp_input_img.bmp')
	positivesX = mainLoop(screen, px,positive, 'Please Select Positive Examples That Fall Off The Radar. (BLUE:input,RED:detected)',mode='pos')
	pygame.display.quit()
	bbimgX = numpy.empty_like(img)
	bbimgX[:] = img

	for col,row in positivesX:
	        bbimgX=utils.draw_bounding_boxG(bbimgX,col,row)

	pilbmp = Image.fromarray(numpy.uint8(bbimgX*255))
	pilbmp.save('temp_selected_img.bmp')
	negative = list()
	screen, px = setup('temp_selected_img.bmp')
	negativesX = mainLoop(screen, px, negative, 'Please Mark False Positive Boxes', mode='neg')
	pygame.display.quit()

	return positivesX,negativesX

def run_pygame_asking(img,text):
	pygame.init()
	pilbmp = Image.fromarray(numpy.uint8(img*255))
	pilbmp.thumbnail((img.shape[1]/2,img.shape[0]/2),Image.ANTIALIAS)
	pilbmp.save('temp_input_img.bmp')
	positive=list()
	screen, px = setup('temp_input_img.bmp')
	salience = mainLoop(screen, px,positive, text, mode='salience')
	pygame.display.quit()

	return salience
##########################################################################----pygame----###########################################################

rearranged_actionlist = list()
HTML = genHTML_header()
TEXT = ''
prev_time = 0
size = 30
start = False
HAS_LOOP=False
#### Detect loop signal and create list of instances to loop
## create list of instances to loop
for each_action in actionlist:
	if each_action['class'] == 'end_loop':
		start = True
		HTML += '<span class=\"skw\">Instances List</span>&#61;find_patterns_like('
		TEXT += 'instances = ['
		instances = list()
		imgs = list()
	elif start:
		if each_action['class'] == 'Click' and each_action['CTRL_ON']:
			imstart = plt.imread(path_to_imgs + each_action['fname_start'])
			im_crop_start = imstart[max(each_action['pos_start'][1]-size,0):min(each_action['pos_start'][1]+size+1,imstart.shape[0]),max(each_action['pos_start'][0]-size,0):min(each_action['pos_start'][0]+size+1,imstart.shape[1]),:]
			imname = ''.join([random.choice(string.letters+string.digits) for i in xrange(16)])
			HTML += '<img src=\"{0}.png\" />&#44;'.format(imname)
			TEXT += '"{0}.png",'.format(imname)
			plt.imsave(path_to_save+imname+'.png',im_crop_start)
			#for training classifier
			instances.append(each_action['pos_start'])
			imgs.append(each_action['fname_start']) 

		elif each_action['class'] == 'end_selection':
			first_im_int = int(imgs[0][:-4])
			if os.path.exists(path_to_imgs + '%.6d.bmp'%(first_im_int-1)):
				medianIMG=plt.imread(path_to_imgs + '%.6d.bmp'%(first_im_int-1))
			else:
				medianIMG=plt.imread(path_to_imgs + imgs[0])

			loop_action={'class':'instance_list','instances':instances,'removed_pt_img':medianIMG}
			HTML = HTML[:-5] + ')\n'
			TEXT = TEXT[:-1] + ']\n'
			example_imname = imname
			HAS_LOOP = True

#find locations of each instance in the list (For sikuliX)
if HAS_LOOP:
	print 'find the location of each instance in the list'
	if TRAIN_CLASSIFIER:
		num_insts=len(instances)
		thresh=0.7
		##train and locate location
		training_img = medianIMG
		random.shuffle(instances)
		training_instances = instances[:num_insts]
		loop_instances_clf = utils.learn_img_classifier(training_img,training_instances)
		predictI = utils.locateIMG(training_img,loop_instances_clf)
		predict = numpy.empty_like(predictI)
		predict[:] = predictI
		rowsI,colsI = utils.nonMaxSuppress_nonZero(predict,thresh=thresh)
		bbimg = numpy.empty_like(medianIMG)
		bbimg[:] = medianIMG

		#draw bb on dectected object 
		for indx in range(len(rowsI)):
			bbimg=utils.draw_bounding_boxR(bbimg,colsI[indx],rowsI[indx])

		#draw bb on training object
		for col,row in training_instances:
			bbimg=utils.draw_bounding_boxB(bbimg,col,row)

		bbimgI = numpy.empty_like(bbimg)
		positives,negatives,unsures,supporters,loop_instances_clf = run_pygame_v2(medianIMG,training_instances,zip(colsI,rowsI))
		thresh = 0.5
		supporter_clf =list()

		for each_sup in supporters:
			padded_img = numpy.zeros(shape=(medianIMG.shape[0]+(2*size),medianIMG.shape[1]+(2*size),medianIMG.shape[2]))
			padded_img[size:-size,size:-size,:] = medianIMG
			template = padded_img[max(each_sup[1],0):min(each_sup[1]+(2*size)+1,padded_img.shape[0]),max(each_sup[0],0):min(each_sup[0]+(2*size)+1,padded_img.shape[1]),:]

			####### 1st with NCC ###########
			matched = match_template(padded_img,template)[:,:,0]
			predict = numpy.empty_like(matched)
			predict[:] = matched
			rowsI,colsI = utils.nonMaxSuppress_nonZero(predict,thresh=thresh)
			neglist=list()

			for X_max,Y_max in zip(colsI,rowsI):
				if numpy.dot([Y_max-each_sup[1],X_max-each_sup[0]],[Y_max-each_sup[1],X_max-each_sup[0]]) > 10:
					neglist.append((X_max,Y_max))

			if len(neglist)>0:
				print 'in-1 number of false positives = {0}'.format(len(neglist))
				####### 2nd with rf hard negative ###########
				supporter_instances_clf = utils.learn_img_classifier(training_img,each_sup,hardnegatives = neglist, sample_percentage = 0, neighbour = 0)
				predictII = utils.locateIMG(training_img,supporter_instances_clf)
				rowsI,colsI = utils.nonMaxSuppress_nonZero(predictII,thresh=thresh)

				for X_max,Y_max in zip(colsI,rowsI):
					if numpy.dot([Y_max-each_sup[1],X_max-each_sup[0]],[Y_max-each_sup[1],X_max-each_sup[0]]) > 10:
						neglist.append((X_max,Y_max))
				
				if len(colsI)>0:
					print 'in-2 number of false positives = {0}'.format(len(neglist))
					####### 3nd with rf hard negative ###########  we keep adding hard negative until there is no more to add
					supporter_instances_clf = utils.learn_img_classifier(training_img,each_sup,hardnegatives = neglist, sample_percentage = 0, neighbour = 0)
					predictII = utils.locateIMG(training_img,supporter_instances_clf)
					rowsI,colsI = utils.nonMaxSuppress_nonZero(predictII,thresh=thresh)
					for X_max,Y_max in zip(colsI,rowsI):
						if numpy.dot([Y_max-each_sup[1],X_max-each_sup[0]],[Y_max-each_sup[1],X_max-each_sup[0]]) > 10:
							neglist.append((X_max,Y_max))

				supporter_clf.append(supporter_instances_clf)

		loop_action['supporter_clf'] = supporter_clf
		loop_action['positives'] = positives
		loop_action['negatives'] = negatives
		loop_action['supporters'] = supporters
		loop_action['detector'] = loop_instances_clf

	else:
		TEXT += 'instance_locations = list()\n'
		TEXT += 'for each_imgname in instances:\n'
		TEXT += '\tinstance_locations.append(find(each_imgname))\n'
		saliences_list = list()

		for x,y in instances:
			padded_img = numpy.zeros(shape=(medianIMG.shape[0]+(2*size),medianIMG.shape[1]+(2*size),medianIMG.shape[2]))
			padded_img[size:-size,size:-size,:] = medianIMG
			template = padded_img[max(y,0):min(y+(2*size)+1,padded_img.shape[0]),max(x,0):min(x+(2*size)+1,padded_img.shape[1]),:]
			matched = match_template(padded_img,template)[:,:,0]
			flat_index = numpy.argmax(matched)
			Y_max,X_max = numpy.unravel_index(flat_index,matched.shape)

			if numpy.dot([Y_max-y,X_max-x],[Y_max-y,X_max-x]) < 10 and not TEDIOUS: 
				saliences_list.append([])
			else:
				bbimg = numpy.empty_like(medianIMG)
				bbimg[:] = medianIMG
				bbimg=utils.draw_bounding_boxR(bbimg,X_max,Y_max)
				bbimg=utils.draw_bounding_boxG(bbimg,x,y)
				print 'Question? : '
				print questions[4]
				saliences_list.append(run_pygame_asking(bbimg,questions[4]))

		loop_action['saliences_list'] = saliences_list

	rearranged_actionlist.append(loop_action)

IN_LOOP = False
SELECTING = False
IN_WAIT = False
WAIT_FOR_1_ACTION = False
test_frames = sorted(os.listdir(path_to_imgs))
data=list()
with open(action_path + 'log_processed.txt','r') as input_file:
	for line in input_file:
		data.append(line.split(','))

OLD_TYPE ='NULL'
last_action=0

## transcipt each action
for each_action in actionlist:
	print each_action['class']
	try:
		if each_action['class'] == 'standby':

			standby_img = each_action['wait_img']
			print 'Question? : '
			print questions[7]
			looking_for_this_list = run_pygame_asking(each_action['wait_img']/255.,questions[7])
			looking_for_this = looking_for_this_list[0] #only one pattern for now
			padded_img = numpy.zeros(shape=(each_action['wait_img'].shape[0]+(2*size),each_action['wait_img'].shape[1]+(2*size),each_action['wait_img'].shape[2]))
			padded_img[size:-size,size:-size,:] = each_action['wait_img']
			template = padded_img[max(looking_for_this[1],0):min(looking_for_this[1]+(2*size)+1,padded_img.shape[0]),max(looking_for_this[0],0):min(looking_for_this[0]+(2*size)+1,padded_img.shape[1]),:]
			imname = ''.join([random.choice(string.letters+string.digits) for i in xrange(16)])
			HTML += '<span class=\"skw\">Stanby for</span> (<img src=\"{0}.png\"/>, at&#61({1},{2})) &#58;\n'.format(imname,looking_for_this[0],looking_for_this[1])
			plt.imsave(path_to_save+imname+'.png',template/255.)
			each_action['looking_for']=looking_for_this
			rearranged_actionlist.append(each_action)	
			IN_WAIT = True
			WAIT_FOR_1_ACTION = True
			standby_imgs_list = [standby_img]

		elif each_action['class'] == 'start_loop':
			IN_LOOP = True
			HTML += '<span class=\"skw\">For</span> each (<img src=\"{0}.png\"/>) in <span class=\"skw\">Instances List</span> &#58;\n'.format(example_imname)
			TEXT += 'for each_instance_loc in instance_locations:\n'
			rearranged_actionlist.append(each_action)

		elif each_action['class'] == 'end_loop':
			IN_LOOP = False
			SELECTING = True
			rearranged_actionlist.append(each_action)

		elif each_action['class'] == 'end_selection':
			SELECTING = False

		elif not SELECTING:
			if each_action['class'] != 'Typing':  ##Have image 
				if IN_LOOP: ##add tab to indicate the action is in for loop
					HTML += '&#09;'
					TEXT += '\t'
				if IN_WAIT: ##add tab to indicate the action is in for loop
					HTML += '&#09;'

				im_before_start_name = int(each_action['fname_start'][:-4])-1
				im_before_start = plt.imread(path_to_imgs + '%.6d.bmp'%im_before_start_name)
				each_action['removed_pt_img_str'] = im_before_start
				im_crop_start = im_before_start[max(each_action['pos_start'][1]-size,0):min(each_action['pos_start'][1]+size+1,im_before_start.shape[0]),max(each_action['pos_start'][0]-size,0):min(each_action['pos_start'][0]+size+1,im_before_start.shape[1]),:]
				each_action['removed_pt_img_end'] = im_before_start
				im_crop_end = im_before_start[max(each_action['pos_end'][1]-size,0):min(each_action['pos_end'][1]+size+1,im_before_start.shape[0]),max(each_action['pos_end'][0]-size,0):min(each_action['pos_end'][0]+size+1,im_before_start.shape[1]),:]

	#######################   Collect one more img to Train Standby Detector  ###############################
				if WAIT_FOR_1_ACTION:
					WAIT_FOR_1_ACTION = False
					if numpy.sqrt(numpy.dot([looking_for_this[1]-each_action['pos_start'][1],looking_for_this[0]-each_action['pos_start'][0]],[looking_for_this[1]-each_action['pos_start'][1],looking_for_this[0]-each_action['pos_start'][0]])) < 40:
						standby_imgs_list.append(each_action['removed_pt_img_str'])
						each_action['same_obj_as_standby'] = True
						print 'in'
	#######################   Collect one more img to Train Standby Detector  ###############################

				padded_img = numpy.zeros(shape=(im_before_start.shape[0]+(2*size),im_before_start.shape[1]+(2*size),im_before_start.shape[2]))
				padded_img[size:-size,size:-size,:] = im_before_start
				template = padded_img[max(each_action['pos_start'][1],0):min(each_action['pos_start'][1]+(2*size)+1,padded_img.shape[0]),max(each_action['pos_start'][0],0):min(each_action['pos_start'][0]+(2*size)+1,padded_img.shape[1]),:]
				matched = match_template(padded_img,template)[:,:,0]
				flat_index = numpy.argmax(matched)
				Y_max,X_max = numpy.unravel_index(flat_index,matched.shape)
				V_max = numpy.max(matched)
				thresh = 0.5
				
				if numpy.dot([Y_max-each_action['pos_start'][1],X_max-each_action['pos_start'][0]],[Y_max-each_action['pos_start'][1],X_max-each_action['pos_start'][0]]) < 10 and not TEDIOUS:
					rows,cols = utils.nonMaxSuppress_nonZero(matched,thresh=(0.8*V_max))
					rows = rows[1:] #delete the first element which is the one provided by user
					cols = cols[1:] #delete the first element which is the one provided by user

					if len(rows)>0 :
						bbimg = numpy.empty_like(im_before_start)
						bbimg[:] = im_before_start

						#draw bb on dectected object 
						for indx in range(len(rows)):
							bbimg=utils.draw_bounding_boxR(bbimg,cols[indx],rows[indx])

						bbimg=utils.draw_bounding_boxG(bbimg,each_action['pos_start'][0],each_action['pos_start'][1])
						print 'Question? : '
						print questions[10]
						saliences = run_pygame_asking(bbimg,questions[10])
						
						#when the user doesnt provide supporters
						if len(saliences) == 0:
							padded_img = numpy.zeros(shape=(im_before_start.shape[0]+(2*size),im_before_start.shape[1]+(2*size),im_before_start.shape[2]))
							padded_img[size:-size,size:-size,:] = im_before_start
							template = padded_img[max(each_action['pos_start'][1],0):min(each_action['pos_start'][1]+(2*size)+1,padded_img.shape[0]),max(each_action['pos_start'][0],0):min(each_action['pos_start'][0]+(2*size)+1,padded_img.shape[1]),:]

							####### 1st with NCC ###########
							matched = match_template(padded_img,template)[:,:,0]#matched = numpy.squeeze(signal.correlate2d(padded_img[:,:,0],template[:,:,0],mode='valid'))#
							predict = numpy.empty_like(matched)
							predict[:] = matched
							rowsI,colsI = utils.nonMaxSuppress_nonZero(predict,thresh=thresh)
							neglist=list()

							for X_max,Y_max in zip(colsI,rowsI):
								if numpy.dot([Y_max-each_action['pos_start'][1],X_max-each_action['pos_start'][0]],[Y_max-each_action['pos_start'][1],X_max-each_action['pos_start'][0]]) > 10:
									neglist.append((X_max,Y_max))

							if len(neglist)>0:
								####### 2nd with rf hard negative ###########
								action_instances_clf = utils.learn_img_classifier(im_before_start,each_action['pos_start'],hardnegatives = neglist, sample_percentage = 0, neighbour = 0)
								predictII = utils.locateIMG(im_before_start,action_instances_clf)
								rowsI,colsI = utils.nonMaxSuppress_nonZero(predictII,thresh=thresh)
					
								for X_max,Y_max in zip(colsI,rowsI):
									if numpy.dot([Y_max-each_action['pos_start'][1],X_max-each_action['pos_start'][0]],[Y_max-each_action['pos_start'][1],X_max-each_action['pos_start'][0]]) > 10:
										neglist.append((X_max,Y_max))
								
								if len(colsI)>0:
									####### 3nd with rf hard negative ###########  we keep adding hard negative until there is no more to add
									action_instances_clf = utils.learn_img_classifier(im_before_start,each_action['pos_start'],hardnegatives = neglist, sample_percentage = 0, neighbour = 0)

								each_action['action_clf']=action_instances_clf

					else:
						saliences = []

				else:		
					bbimg = numpy.empty_like(im_before_start)
					bbimg[:] = im_before_start
					bbimg=utils.draw_bounding_boxR(bbimg,X_max,Y_max)
					bbimg=utils.draw_bounding_boxG(bbimg,each_action['pos_start'][0],each_action['pos_start'][1])
					print 'Question? : '
					print questions[13]
					saliences = run_pygame_asking(bbimg,questions[13])

					if len(saliences) == 0:
						padded_img = numpy.zeros(shape=(im_before_start.shape[0]+(2*size),im_before_start.shape[1]+(2*size),im_before_start.shape[2]))
						padded_img[size:-size,size:-size,:] = im_before_start
						template = padded_img[max(each_action['pos_start'][1],0):min(each_action['pos_start'][1]+(2*size)+1,padded_img.shape[0]),max(each_action['pos_start'][0],0):min(each_action['pos_start'][0]+(2*size)+1,padded_img.shape[1]),:]

						####### 1st with NCC ###########
						matched = match_template(padded_img,template)[:,:,0]#matched = numpy.squeeze(signal.correlate2d(padded_img[:,:,0],template[:,:,0],mode='valid'))#
						predict = numpy.empty_like(matched)
						predict[:] = matched
						rowsI,colsI = utils.nonMaxSuppress_nonZero(predict,thresh=thresh)
						neglist=list()

						for X_max,Y_max in zip(colsI,rowsI):
							if numpy.dot([Y_max-each_action['pos_start'][1],X_max-each_action['pos_start'][0]],[Y_max-each_action['pos_start'][1],X_max-each_action['pos_start'][0]]) > 10:
								neglist.append((X_max,Y_max))

						if len(neglist)>0:
							####### 2nd with rf hard negative ###########
							action_instances_clf = utils.learn_img_classifier(im_before_start,each_action['pos_start'],hardnegatives = neglist, sample_percentage = 0, neighbour = 0)
							predictII = utils.locateIMG(im_before_start,action_instances_clf)
							rowsI,colsI = utils.nonMaxSuppress_nonZero(predictII,thresh=thresh)

							for X_max,Y_max in zip(colsI,rowsI):
								if numpy.dot([Y_max-each_action['pos_start'][1],X_max-each_action['pos_start'][0]],[Y_max-each_action['pos_start'][1],X_max-each_action['pos_start'][0]]) > 10:
									neglist.append((X_max,Y_max))

							if len(colsI)>0:
								####### 3nd with rf hard negative ###########  we keep adding hard negative until there is no more to add
								action_instances_clf = utils.learn_img_classifier(im_before_start,each_action['pos_start'],hardnegatives = neglist, sample_percentage = 0, neighbour = 0)

						each_action['action_clf']=action_instances_clf

				if DEBUG:
					detector_list=list()
					print 'training detectors'
					detector_list.append({'offset':(0,0),'clf':utils.learn_img_classifier(im_before_start,each_action['pos_start'])})#the appearance where the user is performing the action
					for each_salience in saliences:
						offset=(each_action['pos_start'][0]-each_salience[0],each_action['pos_start'][1]-each_salience[1])
						salience_clf = utils.learn_img_classifier(im_before_start,each_salience)
						detector_list.append({'offset':offset,'clf':salience_clf})

					predicted_list=list()
					print 'predicting'
					PREDICT=numpy.zeros(im_before_start.shape[:2])
					non = lambda s: s if s<0 else None
					mom = lambda s: max(0,s)

					for index_detector,each_detector in enumerate(detector_list):
						pred = utils.locateIMG(imremove_pt2,each_detector['clf'])
						ox,oy = each_detector['offset']
						shift_pred = numpy.zeros_like(pred)
						shift_pred[mom(oy):non(oy), mom(ox):non(ox)] = pred[mom(-oy):non(-oy), mom(-ox):non(-ox)]
						PREDICT += shift_pred
						predicted_list.append(pred)
						plt.imsave(result_path+'/detector{0}_at{1}.png'.format(index_detector,each_action['start']),pred)

					plt.imsave(result_path+'/detector_combined_at{0}.png'.format(each_action['start']),PREDICT)
					plt.imsave(result_path+'/imremove_pt_at{0}.png'.format(each_action['start']),imremove_pt)

				each_action['saliences']=saliences

			if each_action['class'] == 'Click':
				imname = ''.join([random.choice(string.letters+string.digits) for i in xrange(16)])

				if prev_time > 0:
					TEXT += 'sleep({0})\n'.format(0.033*(each_action['start']-prev_time))

				HTML += '<span class=\"skw\">Click</span>(<img src=\"{0}.png\" />)\n'.format(imname)

				if IN_LOOP:
					TEXT += '\tclick(each_instance_loc)\n'
				else:	
					TEXT += 'click("{0}")\n'.format(imname+'.png')

				plt.imsave(path_to_save+imname+'.png',im_crop_start)
				prev_time = each_action['end']
				OLD_TYPE ='NULL'

			elif each_action['class'] == 'RClick':
				imname = ''.join([random.choice(string.letters+string.digits) for i in xrange(16)])

				if prev_time > 0:
					TEXT += 'sleep({0})\n'.format(0.033*(each_action['start']-prev_time))

				HTML += '<span class=\"skw\">RightClick</span>(<img src=\"{0}.png\" />)\n'.format(imname)

				if IN_LOOP:
					TEXT += '\trightClick(each_instance_loc)\n'
				else:
					TEXT += 'rightClick("{0}")\n'.format(imname+'.png')

				plt.imsave(path_to_save+imname+'.png',im_crop_start)
				prev_time = each_action['end']
				OLD_TYPE ='NULL'

			elif each_action['class'] == 'DoubleClick':
				imname = ''.join([random.choice(string.letters+string.digits) for i in xrange(16)])

				if prev_time > 0:
					TEXT += 'sleep({0})\n'.format(0.033*(each_action['start']-prev_time))

				HTML += '<span class=\"skw\">DoubleClick</span>(<img src=\"{0}.png\" />)\n'.format(imname)

				if IN_LOOP:
					TEXT += '\tdoubleClick(each_instance_loc)\n'
				else:
					TEXT += 'doubleClick("{0}")\n'.format(imname+'.png')

				plt.imsave(path_to_save+imname+'.png',im_crop_start)
				prev_time = each_action['end']
				OLD_TYPE ='NULL'

			elif each_action['class'] == 'ClickDrag':
				imname = ''.join([random.choice(string.letters+string.digits) for i in xrange(16)])
				imname2 = ''.join([random.choice(string.letters+string.digits) for i in xrange(16)])

				if prev_time > 0:
					TEXT += 'sleep({0})\n'.format(0.033*(each_action['start']-prev_time))

				HTML += '<span class=\"skw\">Drag</span>(<img src=\"{0}.png\" />)<span class=\"skw\">to</span>(<img src=\"{1}.png\" />)\n'.format(imname,imname2)

				if IN_LOOP:
					TEXT += '\tdragDrop(each_instance_loc,"{0}")\n'.format(imname2+'.png')
				else:
					TEXT += 'dragDrop("{0}","{1}")\n'.format(imname+'.png',imname2+'.png')

				plt.imsave(path_to_save+imname+'.png',im_crop_start)
				plt.imsave(path_to_save+imname2+'.png',im_crop_end)
				prev_time = each_action['end']

				if numpy.dot([each_action['pos_start'][1]-each_action['pos_end'][1],each_action['pos_start'][0]-each_action['pos_end'][0]],[each_action['pos_start'][1]-each_action['pos_end'][1],each_action['pos_start'][0]-each_action['pos_end'][0]]) > 5 :
					size = 30
					padded_img = numpy.zeros(shape=(im_before_start.shape[0]+(2*size),im_before_start.shape[1]+(2*size),im_before_start.shape[2]))
					padded_img[size:-size,size:-size,:] = im_before_start
					template = padded_img[max(each_action['pos_end'][1],0):min(each_action['pos_end'][1]+(2*size)+1,padded_img.shape[0]),max(each_action['pos_end'][0],0):min(each_action['pos_end'][0]+(2*size)+1,padded_img.shape[1]),:]
					matched = match_template(padded_img,template)[:,:,0]
					flat_index = numpy.argmax(matched)
					Y_max,X_max = numpy.unravel_index(flat_index,matched.shape)

					if numpy.dot([Y_max-each_action['pos_end'][1],X_max-each_action['pos_end'][0]],[Y_max-each_action['pos_end'][1],X_max-each_action['pos_end'][0]]) < 10 and not TEDIOUS:
						rows,cols = utils.nonMaxSuppress_nonZero(matched,thresh=(0.8*V_max))
						rows = rows[1:] #delete the first element which is the one provided by user
						cols = cols[1:] #delete the first element which is the one provided by user

						if len(rows)>0 :
							bbimg = numpy.empty_like(im_before_start)
							bbimg[:] = im_before_start

							#draw bb on dectected object 
							for indx in range(len(rows)):
								bbimg=utils.draw_bounding_boxR(bbimg,cols[indx],rows[indx])

							bbimg=utils.draw_bounding_boxG(bbimg,each_action['pos_end'][0],each_action['pos_end'][1])
							print 'Question? : '
							print questions[16]
							saliences = run_pygame_asking(bbimg,questions[16])

						else:
							saliences = []

					else:		
						bbimg = numpy.empty_like(im_before_start)
						bbimg[:] = im_before_start
						bbimg=utils.draw_bounding_boxB(bbimg,each_action['pos_start'][0],each_action['pos_start'][1])
						bbimg=utils.draw_bounding_boxG(bbimg,each_action['pos_end'][0],each_action['pos_end'][1])
						bbimg=utils.draw_bounding_boxR(bbimg,X_max,Y_max)
						print 'Question? : '
						print questions[16]
						saliences = run_pygame_asking(bbimg,questions[16])

				else:
					each_action['same_as_start'] = True

				each_action['saliences_end'] = saliences
				OLD_TYPE ='NULL'

			elif each_action['class'] == 'Typing':
				TYPE = each_action['typing']

				if TYPE != OLD_TYPE:
					if IN_LOOP: ##add tab to indicate the action is in for loop
						HTML += '&#09;'
						TEXT += '\t'
					if IN_WAIT: ##add tab to indicate the action is in for loop
						HTML += '&#09;'
					if TYPE not in string.letters+string.digits+'.?!+-*/':
						HTML += '<span class=\"skw\">Type</span>(<span class=\"skw\">key</span>({0}))\n'.format(TYPE)

						if IN_LOOP:
							TEXT += '\ttype(Key.{0})\n'.format(TYPE)
						else:
							TEXT += 'type(Key.{0})\n'.format(TYPE)

					else:
						HTML += '<span class=\"skw\">Type</span>({0})\n'.format(TYPE)

						if IN_LOOP:
							TEXT += '\ttype({0})\n'.format(TYPE)
						else:
							TEXT += 'type({0})\n'.format(TYPE)

					OLD_TYPE = TYPE

			rearranged_actionlist.append(each_action)

	except RuntimeError:
		raise SystemExit('error in transcription line:763')

#train classifier for standby 
for each_action in rearranged_actionlist:
	if each_action['class'] == 'standby':
		#if len(standby_imgs_list) == 2: #first mouse click action click on the object we want to standby for
		thresh = 0.5
		padded_img = numpy.zeros(shape=(each_action['wait_img'].shape[0]+(2*size),each_action['wait_img'].shape[1]+(2*size),each_action['wait_img'].shape[2]))
		padded_img[size:-size,size:-size,:] = each_action['wait_img']
		template = padded_img[max(looking_for_this[1],0):min(looking_for_this[1]+(2*size)+1,padded_img.shape[0]),max(looking_for_this[0],0):min(looking_for_this[0]+(2*size)+1,padded_img.shape[1]),:]
		
		####### 1st with NCC ###########
		matched = match_template(padded_img,template)[:,:,0]
		predict = numpy.empty_like(matched)
		predict[:] = matched
		rowsI,colsI = utils.nonMaxSuppress_nonZero(predict,thresh=thresh)
		neglist=list()

		for X_max,Y_max in zip(colsI,rowsI):
			if numpy.dot([Y_max-looking_for_this[1],X_max-looking_for_this[0]],[Y_max-looking_for_this[1],X_max-looking_for_this[0]]) > 10:
				neglist.append((X_max,Y_max))

		####### 2nd with rf hard negative ###########
		standby_instances_clf = utils.learn_img_classifier_standby(standby_imgs_list,looking_for_this_list,hardnegatives = neglist, sample_percentage = 0, neighbour = 0)
		predictII = utils.locateIMG(each_action['wait_img'],standby_instances_clf)
		rowsI,colsI = utils.nonMaxSuppress_nonZero(predictII,thresh=thresh)
		for X_max,Y_max in zip(colsI,rowsI):
			if numpy.dot([Y_max-looking_for_this[1],X_max-looking_for_this[0]],[Y_max-looking_for_this[1],X_max-looking_for_this[0]]) > 10:
				neglist.append((X_max,Y_max))

		####### 3nd with rf hard negative ###########  we keep adding hard negative until there is no more to add
		standby_instances_clf = utils.learn_img_classifier_standby(standby_imgs_list,looking_for_this_list,hardnegatives = neglist, sample_percentage = 0, neighbour = 0)
		predictIII = utils.locateIMG(each_action['wait_img'],standby_instances_clf)

		each_action['standby_clf'] = standby_instances_clf

with open(path_to_save+save_path+'.py','w') as f:
	f.write(TEXT)

HTML = genHTML_tail(HTML)
with open(path_to_save+save_path+'.html','w') as f:
	f.write(HTML)

print 'done!!'

with open(action_path+'prepared_actionlist.pickle', 'wb') as handle:
	pickle.dump(rearranged_actionlist, handle)