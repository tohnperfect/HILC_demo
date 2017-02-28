"""Predict basic action for each segment."""
import sys
import argparse
import os
import numpy
import  utils
import pickle
#from sklearn.linear_model import SGDClassifier
import csv
from collections import OrderedDict
from itertools import izip

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['backend'] = 'TkAgg'

#######################################################pre-defined arguments###################################################
disc = 7. #displacement threshold the pointer is considered moving when sum of x displace and y displace is more than disc
epsilon = 5 #few frames after last change point first experiment till 23.10 17/12/15 using epsilon = 2
eps = 2
overlap = 0.3 # %overlap
threshold = -1
multi_p = 2

class_dict = {0:'Idle',1:'Click',2:'ClickDrag',3:'DoubleClick',4:'RClick'}
class_dict_reverse = {'Idle':0,'Click':1,'ClickDrag':2,'DoubleClick':3,'RClick':4}
class_name = ['Click','ClickDrag','DoubleClick','RClick']
#######################################################pre-defined arguments####################################################

parser = argparse.ArgumentParser()
#parser.add_argument('--p', type=str, help='path to dataset default is /media/localadmin/DATA1/DATASET/', default='')
parser.add_argument('--p', type=str, help='path to test data default all folders in /media/localadmin/DATA1/DATASET/TEST/', default='')
parser.add_argument('--s', type=str, help='path to save directory  default is empty string', default='')
parser.add_argument('--e', type=bool, help='Enter editing mode', default=True)
args = parser.parse_args()

savefolder = args.s
path_to_testset_ = args.p
#path_to_testset_ = args.t
#path = args.p
EDIT = args.e

path_to_this_script = sys.path[0] + '/'

if len(path_to_testset_) > 0:
	path_to_testset_list=[path_to_testset_]
	L_,T_,S_ = utils.detect_looping_signal_sniff(path_to_testset_)   # added standby
	loop_signal_list = [L_]
	typing_signal_list = [T_]
	standby_signal_list = [S_]   # added standby
else:
	raise SystemExit('please provide path to path_to_testset by --t=PATH')

### load intermediate model
num_states = 0
inter_MODEL_list={}
for class_ in class_name:
	with open(path_to_this_script+'{0}_intermediat_CLF.pickle'.format(class_), 'rb') as handle:
		inter_MODEL_list[class_] = pickle.load(handle)
		num_states+ = (inter_MODEL_list[class_]['preMODEL'][class_]['num_changes']+1)

pre_MODEL_list = inter_MODEL_list[class_]['preMODEL']
pre_MODEL_classes = inter_MODEL_list[class_]['preMODEL_classes']


for path_to_testset,loop_signal,typing_signal,standby_signal in izip(path_to_testset_list,loop_signal_list,typing_signal_list,standby_signal_list):   #added standby

	path_to_save = path_to_testset + savefolder + '/'

	if not os.path.exists(path_to_save) and len(savefolder) > 0:	
		os.mkdir(path_to_save)

	feature_matrix,change = utils.extract_feature(path_to_testset,disc)
	prediction =  numpy.zeros(shape=(feature_matrix.shape[0],)) #answer 
	changepoint_list=[i for i, j in enumerate(change) if j > 0]

	if feature_matrix.shape[0]>0 and len(changepoint_list)>0:
		cut = changepoint_list[-1]+epsilon
		test = { \
					'fol':'TEST', \
					'name':path_to_testset[-2], \
					'feature_matrix':feature_matrix, \
					'num_sections':numpy.sum(numpy.array(change)>0)+1, \
					'length':cut, \
					'change_graph':change \
				}

	DATA_STRUCT = list()
	""" DATA_STRUCT = [pos1,pos2,pos3,...]  ; 
	pos = [state1,state2,state3,...] ; 
	state = { 'name' , 'MODEL' , 'score' , 'input' , 'output4next' }"""

	max_score = numpy.zeros(shape=(num_states,test['num_sections']-1))
	parents = numpy.zeros(shape=(num_states,test['num_sections']-1))

	### load pairwise cost
	### (row,col) the score of the row appears after the col score(row|col)
	with open(path_to_this_script+'pairwise_addedRClick_noClicksRename.csv', 'rb') as f:
		reader = csv.reader(f)
		PW = list(reader)

	pairwise = numpy.zeros(shape=(num_states,num_states))
	for i_row,e_row in enumerate(PW):
		for i_col,e_col in enumerate(e_row):
			if int(e_col) == 0:
				tran = -1 * multi_p
			elif int(e_col) == 1:
				tran = 0
			elif int(e_col) == 2:
				tran = 1 * multi_p
			else:
				raise SystemExit('corrupt CSV file.')
			pairwise[i_row,i_col] = tran

	### pre_compute pre-classifiers score for each segment
	pre_compute_score = -1.*numpy.ones(shape=(len(pre_MODEL_classes),test['num_sections']-1))

	for i_class,each_class in enumerate(pre_MODEL_classes):
		this_MODEL = pre_MODEL_list[each_class]
		this_num_changes = this_MODEL['num_changes']
		for i_change in range(len(changepoint_list)-this_num_changes):

			clip_len = changepoint_list[i_change+this_num_changes] - changepoint_list[i_change] + epsilon

			if clip_len > this_MODEL['min_len']-eps and clip_len < this_MODEL['max_len']+eps:

				feature_matrix_cropped = test['feature_matrix'][changepoint_list[i_change] : changepoint_list[i_change+this_num_changes] + epsilon]
				changes_cropped = test['change_graph'][changepoint_list[i_change] : changepoint_list[i_change+this_num_changes] + epsilon]
				BOFfeat = utils.cal_one_BOF2context(feature_matrix_cropped,changes_cropped)

				score = numpy.squeeze(numpy.dot(this_MODEL['filter'],BOFfeat.T))
			else:
				score = -1

			for each_change_model in range(this_num_changes+1):	
				pre_compute_score[i_class,i_change+each_change_model] = score


	### compute unary cost
	unary = -100.*numpy.ones(shape=(num_states,test['num_sections']-1))
	index_state = 0
	for each_class in class_name:

		this_MODEL = inter_MODEL_list[each_class]
		this_num_changes = pre_MODEL_list[each_class]['num_changes']

		for i_change in range(len(changepoint_list)-this_num_changes):
			clip_len = changepoint_list[i_change+this_num_changes] - changepoint_list[i_change] + epsilon

			if clip_len > pre_MODEL_list[each_class]['min_len']-eps and clip_len < pre_MODEL_list[each_class]['max_len']+eps:

				feature_vec = numpy.zeros(shape=(1,len(pre_MODEL_classes)))
				feature_vec = pre_compute_score[:,i_change] 
				cls_predict=inter_MODEL_list[each_class]['classifier'].predict_proba(feature_vec.reshape(1,-1))

				if cls_predict.shape[1] == 2:
					score = cls_predict[0,1]
				elif cls_predict.shape[1] == 1:
					score = 0
				else:
					raise SystemExit('prediction error')

			else:
				score = -1

			for each_change_model in range(this_num_changes+1):	
				unary[index_state+each_change_model,i_change+each_change_model] = score

		index_state += (this_num_changes+1)

	### forward passing
	max_score[:,0] = unary[:,0]
	for position in range(1,test['num_sections']-1):
		for i_c_state in range(num_states):
			poss_path_score = numpy.zeros(shape=(num_states,))

			for i_prev_state in range(num_states):
				poss_path_score[i_prev_state] = max_score[i_prev_state,position-1]+pairwise[i_c_state,i_prev_state]

			max_parent = numpy.argmax(poss_path_score)
			max_score[i_c_state,position] = poss_path_score[max_parent]+unary[i_c_state,position]
			parents[i_c_state,position] = max_parent

	### backward passing
	bestPath = numpy.zeros(shape=(test['num_sections']-1,))
	maxlastpos = numpy.argmax(max_score[:,-1])
	bestPath[-1] = maxlastpos
	parent = parents[maxlastpos,-1]
	for index in range(test['num_sections']-3,-1,-1):
		bestPath[index] = parent
		parent = parents[int(parent),index]

	print 'unary'
	print unary
	print 'bestPath'
	print bestPath
	print 'max_score'
	print max_score
	print 'parents'
	print parents

	### translate to prediction
	index_numchange = 0
	for BEST in bestPath:
		if index_numchange < len(changepoint_list):
			prediction[changepoint_list[index_numchange]] = BEST+1
			index_numchange += 1

	### write the transcription

	################## make list of starting position and end position of each action to use to find image.
	action_list = list()
	code_dict = {}
	track_state = 1
	for each_class in class_name:
		code_dict[each_class] = {'start':track_state,'end':track_state+inter_MODEL_list[each_class]['preMODEL'][each_class]['num_changes']}
		track_state += (inter_MODEL_list[each_class]['preMODEL'][each_class]['num_changes']+1)

	HAS_LOOP = False
	if len(loop_signal) == 7:
		HAS_LOOP = True

	WAITING_FOR = False   #added standby
	if 'wait_index' in standby_signal:   #added standby
		WAITING_FOR = True   #added standby
		
	start = False
	c_index = -1
	transcript = numpy.zeros(shape=prediction.shape)
	class_ = ''
	print 'Transcribing ... '
	for index,each_frame in enumerate(prediction):
			if not start:
				if each_frame > 0:
					action = {}
					start = True
					for each_class in class_name:
						if code_dict[each_class]['start'] == each_frame:
							class_ = each_class

					### start_in = index
					action['class'] = class_
					action['start'] = index
					XY,fname,im,time = utils.getPos_fname_sniff(index,path_to_testset)
					action['pos_start'] = XY
					action['fname_start'] = fname
					action['image_start']= im
					action['CTRL_ON'] = False
					action['time_start'] = time
					transcript[index] = class_dict_reverse[class_]

				else:

					### Check Typing
					if typing_signal[index] != 'NULL':
						action_list.append({'class':'Typing', \
											'typing':typing_signal[index], \
											'start':index})

					### Check Looping	
					if HAS_LOOP:
						if index == loop_signal['start_i']:
							action_list.append({'class':'start_loop'})
						elif index == loop_signal['start2_i']:
							action_list.append({'class':'end_loop'})
						elif index == loop_signal['end_i']:
							action_list.append({'class':'end_selection'})

					### Check Waiting
					if WAITING_FOR and standby_signal['wait_index'] == index:    #added standby

						action_list.append({'class':'standby','wait_img':standby_signal['wait_img']})   #added standby

			else:

				if each_frame == code_dict[class_]['end']:
					start = False
					action['end'] = index
					XY,fname,im,time = utils.getPos_fname_sniff(index,path_to_testset)
					action['pos_end'] = XY
					action['fname_end'] = fname
					action['image_end'] = im
					action['time_end'] = time

					if HAS_LOOP:
						if index in loop_signal['CTRL_list']:
							action['CTRL_ON'] = True

					transcript[index] = transcript[index-1]
					action_list.append(action)

				elif (each_frame > code_dict[class_]['start'] and each_frame < code_dict[class_]['end']) or each_frame == 0:
					transcript[index] = transcript[index-1]

				else:
					raise SystemExit('error!! could not find the end of the action')

	### allow users to modify prediction result.
	if EDIT:

		### later will use the label provided by the user to rerun viterbi again.
		done = False
		while not done:
			### iput is the action which the user need to modify.
			for indx,e_action in enumerate(action_list):
				if e_action['class'] != 'Typing':
					print indx,e_action['class'],e_action['fname_start']

			iput = int(raw_input('Please enter the number of step that needed to be editted or {0} if the result is correct\n'.format(len(action_list))))

			if iput < len(action_list):
				c_iput = int(raw_input('Please enter\n 0 if it is a Click,\n 1 for Double Click,\n 2 for Click Drag, and \n 3 for Right Click\n'))
				if c_iput == 0:
					action_list[iput]['class'] = 'Click'
				elif c_iput == 1:
					action_list[iput]['class'] = 'DoubleClick'
				elif c_iput == 2:
					action_list[iput]['class'] = 'ClickDrag'
				else:
					action_list[iput]['class'] = 'RClick'

				### fix unary cost matrix.
				index_state = 0
				found = False
				for each_class in class_name:
					this_num_changes = pre_MODEL_list[each_class]['num_changes']
					if action_list[iput]['class'] != each_class and not found:
						index_state += (this_num_changes+1)
					else:
						found = True

				i_change = 0
				for indx,e_action in enumerate(action_list):
					if e_action['class'] != 'Typing':
						this_num_changes = pre_MODEL_list[e_action['class']]['num_changes']
						if indx != iput:
							i_change += (this_num_changes+1)
						else:
							for each_change_model in range(this_num_changes+1):	
								unary[index_state+each_change_model,i_change+each_change_model] = 100

				### run viterbi
				prediction =  numpy.zeros(shape=(feature_matrix.shape[0],)) #answer
				### forward passing
				max_score[:,0] = unary[:,0]
				for position in range(1,test['num_sections']-1):
					for i_c_state in range(num_states):
						poss_path_score = numpy.zeros(shape=(num_states,))
						for i_prev_state in range(num_states):
							poss_path_score[i_prev_state] = max_score[i_prev_state,position-1]+pairwise[i_c_state,i_prev_state]

						max_parent = numpy.argmax(poss_path_score)
						max_score[i_c_state,position] = poss_path_score[max_parent]+unary[i_c_state,position]
						parents[i_c_state,position] = max_parent


				### backward passing
				bestPath = numpy.zeros(shape=(test['num_sections']-1,))
				maxlastpos = numpy.argmax(max_score[:,-1])
				bestPath[-1] = maxlastpos
				parent = parents[maxlastpos,-1]
				for index in range(test['num_sections']-3,-1,-1):
					bestPath[index] = parent
					parent = parents[parent,index]

				print 'unary'
				print unary
				print 'bestPath'
				print bestPath
				print 'max_score'
				print max_score
				print 'parents'
				print parents

				### translate to prediction
				index_numchange = 0
				for BEST in bestPath:
					if index_numchange < len(changepoint_list):
						prediction[changepoint_list[index_numchange]]=BEST+1
						index_numchange += 1

				### transcribe
				################## make list of starting position and end position of each action to use to find image.
				action_list = list()

				code_dict = {}
				track_state = 1
				for each_class in class_name:
					code_dict[each_class]={'start':track_state,'end':track_state+inter_MODEL_list[each_class]['preMODEL'][each_class]['num_changes']}
					track_state+=(inter_MODEL_list[each_class]['preMODEL'][each_class]['num_changes']+1)

				HAS_LOOP = False
				if len(loop_signal) == 7:
					HAS_LOOP = True

				WAITING_FOR = False   #added standby
				if 'wait_index' in standby_signal:   #added standby
					WAITING_FOR = True   #added standby
					
				start = False
				c_index = -1
				transcript = numpy.zeros(shape=prediction.shape)
				class_ = ''
				print 'Transcribing ... '
				for index,each_frame in enumerate(prediction):
					
						if not start:
							if each_frame > 0:
								action = {}
								start = True
								for each_class in class_name:
									if code_dict[each_class]['start'] == each_frame:
										class_ = each_class

								#start_in = index
								action['class'] = class_
								action['start'] = index
								XY,fname,im,time = utils.getPos_fname_sniff(index,path_to_testset)
								action['pos_start'] = XY
								action['fname_start'] = fname
								action['image_start']= im
								action['CTRL_ON'] = False
								action['time_start'] = time
								transcript[index] = class_dict_reverse[class_]

							else:
								#Check Typing
								if typing_signal[index] != 'NULL':
									action_list.append({'class':'Typing', \
														'typing':typing_signal[index], \
														'start':index})

								#Check Looping	
								if HAS_LOOP:
									if index == loop_signal['start_i']:
										action_list.append({'class':'start_loop'})
									elif index == loop_signal['start2_i']:
										action_list.append({'class':'end_loop'})
									elif index == loop_signal['end_i']:
										action_list.append({'class':'end_selection'})

								#Check Waiting
								if WAITING_FOR and standby_signal['wait_index'] == index:    #added standby
									action_list.append({'class':'standby','wait_img':standby_signal['wait_img']})   #added standby

						else:
							if each_frame == code_dict[class_]['end']:
								#stop_in = index
								start = False
								action['end'] = index
								XY,fname,im,time = utils.getPos_fname_sniff(index,path_to_testset)
								action['pos_end'] = XY
								action['fname_end'] = fname
								action['image_end'] = im
								action['time_end'] = time
								if HAS_LOOP:
									if index in loop_signal['CTRL_list']:
										action['CTRL_ON'] = True
								transcript[index] = transcript[index-1]
								action_list.append(action)
							elif (each_frame > code_dict[class_]['start'] and each_frame < code_dict[class_]['end']) or each_frame == 0:
								transcript[index] = transcript[index-1]
							else:
								raise SystemExit('error!! could not find the end of the action')
			else:
				done = True
	with open(path_to_save+'transcript_actionlist.pickle', 'wb') as handle:
		pickle.dump([transcript,action_list], handle)
	print 'done'
	