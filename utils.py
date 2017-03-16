"""Utility functions"""
import matplotlib.pyplot as plt
import numpy
from sklearn.ensemble import RandomForestClassifier as RF
from skimage.util import view_as_windows
import string
import time
import os
from scipy.ndimage.filters import gaussian_filter

def cal_one_BOF2context(positive_feature,changes,no_sections=3):
	feature = numpy.zeros(shape=(1,8*(no_sections+1)))
	whole,col = positive_feature.shape
	changepoint_list = [i for i, j in enumerate(changes) if j > 0]
	cut = changepoint_list[-1]+1
	chunk = numpy.int(numpy.ceil(1.0*cut/no_sections))

	for each_sec in range(no_sections):
		if each_sec < no_sections-1:
			for each_im in range(chunk*(each_sec),chunk*(each_sec+1)):
				feature[0,int((each_sec*8)+numpy.dot(positive_feature[each_im,:],[2**2,2**1,2**0]))] += 1                        
			feature[0,:] /= (1.*chunk)
		else:
			for each_im in range(chunk*(each_sec),cut): #  the last section of action
				feature[0,int((each_sec*8)+numpy.dot(positive_feature[each_im,:],[2**2,2**1,2**0]))] += 1

			feature[0,:] /= (1.*chunk)
	for each_im in range(cut,whole):
		feature[0,int((no_sections*8)+numpy.dot(positive_feature[each_im,:],[2**2,2**1,2**0]))] += 1  # the rest

	return feature

def extract_feature(path_to_log,disc=7.,saveimg=False):
	data=list()
	with open(path_to_log+'log_processed.txt','r') as input_file:
		for line in input_file:
			data.append(line.split(','))
	feature_matrix = numpy.zeros(shape=(len(data),3))
	Y=list()
	x0=-10
	y0=-10
	for index,data_point in enumerate(data):
		fname = data_point[0]
		x1 = int(data_point[1])
		y1 = int(data_point[2])
		mouse_status = data_point[3]

		### mouse status : feature0->press left = true? , feature1->press right = true? 
		if mouse_status == 'press left':
			feature_matrix[index,0] = 1.
		elif mouse_status == 'press right':
			feature_matrix[index,1] = 1.
		elif mouse_status == 'no press':
			'do nothing'
		else:
			raise SystemExit('error in cal_one_BOF2context function')
		
		if x0 == -10 and y0 == -10:
			'do nothing'
			Y.append(0.)
		else:
			### displacement: feature2->have movement?
			if abs(x1-x0)+abs(y1-y0) > disc:
				feature_matrix[index,2] = 1.

			Y.append(1.*(numpy.dot(feature_matrix[index,:-1]-feature_matrix[index-1,:-1],feature_matrix[index,:-1]-feature_matrix[index-1,:-1])>0))
		
		x0 = x1
		y0 = y1

	if saveimg:
		plt.plot(range(len(Y)),Y)
		plt.savefig(path_to_log+'mousechange.jpg')

	return feature_matrix,Y

def detect_looping_signal_sniff(path_to_log):

	data=list()
	with open(path_to_log+'log_processed.txt','r') as input_file:
		for line in input_file:
			data.append(line.split(','))

	previousKEY=''
	loop = {}
	typing = list() 
	standby = {}

	for index,data_point in enumerate(data):
		fname = data_point[0]
		x1 = int(data_point[1])
		y1 = int(data_point[2])
		mouse_status = data_point[3]
		keyboard_input = data_point[4]
		currentKEY=keyboard_input.replace('Ctrl','Ctr')
		currentKEY=currentKEY.replace('-I','-l')
		currentKEY=currentKEY.replace('-i','-l')
		currentKEY=currentKEY.replace('\n','')

		if ('Ctr-Shift-w' not in previousKEY and 'Ctr-Shift-w' in currentKEY) or ('Ctr-Shift-Snapshot' not in previousKEY and 'Ctr-Shift-Snapshot' in currentKEY): 
			### 'Ctr-Shift-w' cause google chrome to crash
			standby['wait_index'] = index
			standby['wait_img'] = plt.imread(path_to_log + 'imgs/' + fname)

		if ('Ctr-Shift-l' in previousKEY and 'Ctr-Shift-l' not in currentKEY) or ('Ctr-Shift-Pause' in previousKEY and 'Ctr-Shift-Pause' not in currentKEY):
			if 'start_i' not in loop:
				loop['start_i'] = index-1
				loop['start_fname'] = fname
		elif ('Ctr-Shift-l' not in previousKEY and 'Ctr-Shift-l' in currentKEY and 'start_i' in loop) or ('Ctr-Shift-Pause' not in previousKEY and 'Ctr-Shift-Pause' in currentKEY and 'start_i' in loop):
			if 'start2_i' not in loop:
				loop['start2_i'] = index+1
				loop['start2_fname'] = fname
				loop['CTRL_list'] = list()
			elif 'end_i' not in loop:
				loop['end_i'] = index+1
				loop['end_fname'] = fname
			else:
				raise SystemExit('error in detect_looping_signal_sniff function : error in start')

		elif 'start2_i' in loop and 'end_i' not in loop:
			if 'Ctr' in currentKEY:
				loop['CTRL_list'].append(index)
		previousKEY=currentKEY

		if 'Ctr-' in currentKEY:
				temp = currentKEY.replace('Ctr-','')
				if temp in string.letters+string.digits+'.?!+-*/':
					typing.append('CTRL-'+temp)
					#typing.append('NULL') ##not implemented shortkey yet
				else:
					typing.append('NULL')
					
		elif 'Shift-' in currentKEY:
			temp = currentKEY.replace('Shift-','')
			if temp in ['Home','End']:
				typing.append('SHIFT-'+temp)
			else:
				typing.append('NULL')
		else:# ('Ctr' not in currentKEY) and ('Alt' not in currentKEY):
			if 'Return' in currentKEY:
				typing.append('enter')
			elif 'F2' in currentKEY:
				typing.append('f2')
			elif 'Tab' in currentKEY:
				typing.append('tab')
			elif 'Up' in currentKEY:
				typing.append('up')
			elif 'Down' in currentKEY:
				typing.append('down')
			elif 'Left' in currentKEY:
				typing.append('left')
			elif 'Right' in currentKEY:
				typing.append('right')
			elif 'Snapshot' in currentKEY:
				typing.append('printscreen')
			elif 'Space' in currentKEY:
				typing.append('spacebar')
			elif 'Back' in currentKEY:
				typing.append('backspace')
			elif 'Delete' in currentKEY:
				typing.append('delete')
			elif len(currentKEY) == 0:
				typing.append('NULL')
			else:
				if currentKEY in string.letters+string.digits+'.?!+-*/':
					typing.append(currentKEY)
				else:
					typing.append('NULL')

	return loop,typing,standby

def getPos_fname_sniff(index,path_to_log,w=50,h=30):
	data=list()
	with open(path_to_log+'log_processed.txt','r') as input_file:
		for line in input_file:
			data.append(line.split(','))
	x = int(data[index][1])
	y = int(data[index][2])
	fname = data[index][0]
	time = int(data[index][5])
	### get image
	### check if image exists. if the image does not exist, search for the nearest one instead
	if os.path.exists(path_to_log + 'imgs/' + fname):

		img=plt.imread(path_to_log + 'imgs/' + fname) #sniffer

	else:
		
		int_fname = int(fname[:-4])
		### search down
		down = 1
		while (not os.path.exists(path_to_log + 'imgs/' + '%.6d.bmp'%(int_fname-down))) and (int_fname-down >= int(data[0][0][:-4])) :
			down -=1
		### search up
		up = 1
		while (not os.path.exists(path_to_log + 'imgs/' + '%.6d.bmp'%(int_fname+up))) and (int_fname+up <= int(data[-1][0][:-4])):
			up +=1
		if down<up:
			fname = '%.6d.bmp'%(int_fname-down)
		else:
			fname = '%.6d.bmp'%(int_fname+up)

		img=plt.imread(path_to_log + 'imgs/' + fname) #sniffer

	if y-h < 0:
		h = y
	if x-w < 0:
		w = x
	if y+h >= img.shape[0]:
		h = img.shape[0]-y-1
	if x+w >= img.shape[1]:
		w = img.shape[1]-x-1
	im_cropped = img[y-h:y+h+1,x-w:x+w+1,:]

	return (x,y), fname, im_cropped, time

def learn_img_classifier(img, click_position, hardnegatives = list(), neighbour = 3, size = 30, sample_percentage = 0.01):
	t0 = time.time()
	if not isinstance(click_position,list):
		click_position = [click_position] 

	numdup = 1
	### collect positive and negative
	row,col,dim = img.shape

	padded_img = numpy.zeros(shape=(row+(2*size),col+(2*size),dim))
	padded_img[size:-size,size:-size,:] = img

	pos = list()
	neg = list()
	for r in range(row):
		for c in range(col):
			patch = padded_img[r:r+(2*size)+1,c:c+(2*size)+1,:]

			first_pass = True
			for each_position in click_position:

				if r >= each_position[1] - neighbour and r <= each_position[1] + neighbour and \
					c >= each_position[0] - neighbour and c <= each_position[0] + neighbour:
					first_pass = False
					pos.append(patch.flatten())


				elif first_pass:
					first_pass = False
					if len(hardnegatives) > 0:
						if (c,r) in hardnegatives:
							neg = neg +[patch.flatten() for _ in xrange(numdup)]
						elif numpy.random.choice([True,False],p=[sample_percentage,1.0-sample_percentage]) and len(neg) < 1000: # randomly pick negative example
							neg.append(patch.flatten())
					elif numpy.random.choice([True,False],p=[sample_percentage,1.0-sample_percentage]) or (c,r) in hardnegatives: # randomly pick negative example
						neg.append(patch.flatten())

	X=numpy.vstack((numpy.squeeze(numpy.array(pos)),numpy.squeeze(numpy.array(neg))))
	y=numpy.vstack((numpy.ones(shape=(len(pos),1)),numpy.zeros(shape=(len(neg),1))))

	print 'collecting time',
	print time.time()-t0
	t0 = time.time()
	randomforest = RF(n_estimators=50,max_depth=5,class_weight='balanced')
	randomforest.fit(X,numpy.squeeze(y))
	print 'training time',
	print time.time()-t0

	return randomforest

def learn_img_classifier_standby(img_list, click_position, hardnegatives = list(), neighbour = 3, size = 30, sample_percentage = 0.01):
	t0 = time.time()
	if not isinstance(click_position,list):
		click_position = [click_position] 

	numdup = 1

	### collect positive and negative
	row,col,dim = img_list[0].shape
	padded_img_0 = numpy.zeros(shape=(row+(2*size),col+(2*size),dim))
	padded_img_0[size:-size,size:-size,:] = img_list[0]

	if len(img_list) == 2:
		padded_img_1 = numpy.zeros(shape=(row+(2*size),col+(2*size),dim))
		padded_img_1[size:-size,size:-size,:] = img_list[1]

	pos = list()
	neg = list()
	for r in range(row):
		for c in range(col):
			patch_0 = padded_img_0[r:r+(2*size)+1,c:c+(2*size)+1,:]
			if len(img_list) == 2:
				patch_1 = padded_img_1[r:r+(2*size)+1,c:c+(2*size)+1,:]

			first_pass = True
			for each_position in click_position:

				if r >= each_position[1] - neighbour and r <= each_position[1] + neighbour and \
				   c >= each_position[0] - neighbour and c <= each_position[0] + neighbour:
					first_pass = False
					pos.append(patch_0.flatten())
					if len(img_list) == 2:
						pos.append(patch_1.flatten())

				elif first_pass:
					first_pass = False
					if len(hardnegatives) > 0:

						if (c,r) in hardnegatives:
							neg = neg +[patch_0.flatten() for _ in xrange(numdup)]
						elif numpy.random.choice([True,False],p=[sample_percentage,1.0-sample_percentage]) and len(neg) < 1000: # randomly pick negative example
							neg.append(patch_0.flatten())
							
					elif numpy.random.choice([True,False],p=[sample_percentage,1.0-sample_percentage]) and (c,r) in hardnegatives: # randomly pick negative example
						neg.append(patch_0.flatten())

	X=numpy.vstack((numpy.squeeze(numpy.array(pos)),numpy.squeeze(numpy.array(neg))))
	y=numpy.vstack((numpy.ones(shape=(len(pos),1)),numpy.zeros(shape=(len(neg),1))))

	print 'collecting time',
	print time.time()-t0
	t0 = time.time()
	randomforest = RF(n_estimators=50,max_depth=5,class_weight='balanced')
	randomforest.fit(X,numpy.squeeze(y))
	print 'training time',
	print time.time()-t0

	return randomforest

def locateIMG(img2search, MODEL, size = 30, max_win_size = 100, classifier = 'rf'):
	row,col,dim = img2search.shape

	if classifier == 'ada':
		img2search = img2search.astype(numpy.int)

	padded_img = numpy.zeros(shape=(row+(2*size),col+(2*size),dim))
	padded_img[size:-size,size:-size,:] = img2search
	img_predict = numpy.zeros(shape=(row,col))

	if row%(max_win_size-(2*size)) > 0:
		num_row_chunks = row/(max_win_size-(2*size))+1
	else:
		num_row_chunks = row/(max_win_size-(2*size))

	if col%(max_win_size-(2*size)) > 0:
		num_col_chunks = col/(max_win_size-(2*size))+1
	else:
		num_col_chunks = col/(max_win_size-(2*size))

	t0 = time.time()
	if classifier == 'rf':
		for each_r_sec in range(num_row_chunks):
			for each_c_sec in range(num_col_chunks):
				test_win = padded_img[(each_r_sec*(max_win_size-(2*size))):min((each_r_sec*(max_win_size-(2*size)))+max_win_size,padded_img.shape[0]), \
				(each_c_sec*(max_win_size-(2*size))):min((each_c_sec*(max_win_size-(2*size)))+max_win_size,padded_img.shape[1]),:]
				win_list = view_as_windows(test_win,((2*size)+1,(2*size)+1,dim))
				win_list_rs = win_list.reshape((numpy.prod(win_list.shape[:2]),numpy.prod(win_list.shape[2:])))
				img_predict[(each_r_sec*(max_win_size-(2*size))):min((each_r_sec*(max_win_size-(2*size)))+(max_win_size-(2*size)),row), \
				(each_c_sec*(max_win_size-(2*size))):min((each_c_sec*(max_win_size-(2*size)))+(max_win_size-(2*size)),col)] \
				= MODEL.predict_proba(win_list_rs)[:,1].reshape(win_list.shape[:2])

	elif classifier == 'ada':
		result_img = MODEL.predict_feature_matrix(padded_img,test_whole_cascade=True)
		if result_img.shape[0] > img_predict.shape[0]:
			img_predict = result_img[size:-size,size:-size]
		else:
			img_predict = result_img

	t1 = time.time()
	print t1-t0
	ij = numpy.unravel_index(numpy.argmax(img_predict), img_predict.shape)
	x, y = ij[::-1]

	return img_predict

 
def nonMaxSuppress(image,NHoodSize=(21,21)):
	#http://stackoverflow.com/questions/29057159/non-local-maxima-suppression-in-python
	dX, dY = NHoodSize
	M, N = image.shape
	for x in range(0,M-dX+1):
		for y in range(0,N-dY+1):
			window = image[x:x+dX, y:y+dY]
			if numpy.sum(window)==0:
				localMax=0
			else:
				localMax = numpy.amax(window)
			maxCoord = numpy.argmax(window)
			# zero all but the localMax in the window
			window[:] = 0
			window.flat[maxCoord] = localMax
	return image
	
def nonMaxSuppress_nonZero(image,thresh,NHoodSize=(21,21)):
    #http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
	rowsI,colsI = numpy.nonzero(image>thresh)
	
	if len(rowsI) == 0:
		return [],[]
		
	pick = list()
	
	x1 = 1.*colsI - (NHoodSize[0])
	y1 = 1.*rowsI - (NHoodSize[1])
	x2 = 1.*colsI + (NHoodSize[0])
	y2 = 1.*rowsI + (NHoodSize[1])
	
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	s = [image[r,c] for r,c in zip(rowsI,colsI)]
	indx = numpy.argsort(s)
	
	while len(indx) > 0 :
		last = len(indx)-1
		i = indx[last]
		pick.append((rowsI[i],colsI[i]))
		
		xx1 = numpy.maximum(x1[i],x1[indx[:last]])
		yy1 = numpy.maximum(y1[i],y1[indx[:last]])
		xx2 = numpy.minimum(x2[i],x2[indx[:last]])
		yy2 = numpy.minimum(y2[i],y2[indx[:last]])
		
		w = numpy.maximum(0,xx2-xx1+1)
		h = numpy.maximum(0, yy2-yy1+1)
		overlap = (w*h) / area[indx[:last]]
		indx = numpy.delete(indx, numpy.concatenate(([last],numpy.where(overlap > 0.25)[0])))
		
	ret_row,ret_col = zip(*pick)
 
	return ret_row,ret_col

def draw_bounding_boxR(img_white,c_corn,r_corn,temp_width=61,temp_height=61):
	c_max=c_corn-temp_width/2
	r_max=r_corn-temp_height/2
	try:
		Row,Col,Dim = numpy.shape(img_white)
		img_display = numpy.copy(img_white)   
		
	except:
		Row,Col = numpy.shape(img_white)
		img_display = numpy.zeros(shape=(Row,Col,3))
		img_display[:,:,0]=img_white
		img_display[:,:,1]=img_white
		img_display[:,:,2]=img_white

	if numpy.max(img_display) > 1:
		constant = 255.
	else:
		constant = 1.

	for r in range(temp_height):
		ROW=min(r_max+r,Row-1)
		img_display[ROW,min(Col-1,c_max-1),0]=constant
		img_display[ROW,min(Col-1,c_max),0]=constant
		img_display[ROW,min(Col-1,c_max+1),0]=constant
		img_display[ROW,min(Col-1,c_max+2),0]=constant
		img_display[ROW,min(Col-1,c_max+temp_width-1),0]=constant
		img_display[ROW,min(Col-1,c_max+temp_width),0]=constant
		img_display[ROW,min(Col-1,c_max+temp_width+1),0]=constant
		img_display[ROW,min(Col-1,c_max+temp_width+2),0]=constant
		
		img_display[ROW,min(Col-1,c_max-1),1]=0.
		img_display[ROW,min(Col-1,c_max),1]=0.
		img_display[ROW,min(Col-1,c_max+1),1]=0.
		img_display[ROW,min(Col-1,c_max+2),1]=0.
		img_display[ROW,min(Col-1,c_max+temp_width-1),1]=0.
		img_display[ROW,min(Col-1,c_max+temp_width),1]=0.
		img_display[ROW,min(Col-1,c_max+temp_width+1),1]=0.
		img_display[ROW,min(Col-1,c_max+temp_width+2),1]=0.
		
		img_display[ROW,min(Col-1,c_max-1),2]=0.
		img_display[ROW,min(Col-1,c_max),2]=0.
		img_display[ROW,min(Col-1,c_max+1),2]=0.
		img_display[ROW,min(Col-1,c_max+2),2]=0.
		img_display[ROW,min(Col-1,c_max+temp_width-1),2]=0.
		img_display[ROW,min(Col-1,c_max+temp_width),2]=0.
		img_display[ROW,min(Col-1,c_max+temp_width+1),2]=0.
		img_display[ROW,min(Col-1,c_max+temp_width+2),2]=0.
		
		
	for c in range(temp_width):
		COL=min(c_max+c,Col-1)
		img_display[min(Row-1,r_max-1),COL,0]=constant
		img_display[min(Row-1,r_max),COL,0]=constant
		img_display[min(Row-1,r_max+1),COL,0]=constant
		img_display[min(Row-1,r_max+2),COL,0]=constant
		img_display[min(Row-1,r_max+temp_height-1),COL,0]=constant
		img_display[min(Row-1,r_max+temp_height),COL,0]=constant
		img_display[min(Row-1,r_max+temp_height+1),COL,0]=constant
		img_display[min(Row-1,r_max+temp_height+2),COL,0]=constant
		
		img_display[min(Row-1,r_max-1),COL,1]=0
		img_display[min(Row-1,r_max),COL,1]=0
		img_display[min(Row-1,r_max+1),COL,1]=0
		img_display[min(Row-1,r_max+2),COL,1]=0
		img_display[min(Row-1,r_max+temp_height-1),COL,1]=0
		img_display[min(Row-1,r_max+temp_height),COL,1]=0
		img_display[min(Row-1,r_max+temp_height+1),COL,1]=0
		img_display[min(Row-1,r_max+temp_height+2),COL,1]=0
		
		img_display[min(Row-1,r_max-1),COL,2]=0
		img_display[min(Row-1,r_max),COL,2]=0
		img_display[min(Row-1,r_max+1),COL,2]=0
		img_display[min(Row-1,r_max+2),COL,2]=0
		img_display[min(Row-1,r_max+temp_height-1),COL,2]=0
		img_display[min(Row-1,r_max+temp_height),COL,2]=0
		img_display[min(Row-1,r_max+temp_height+1),COL,2]=0
		img_display[min(Row-1,r_max+temp_height+2),COL,2]=0
		
	return img_display/constant

def draw_bounding_boxB(img_white,c_corn,r_corn,temp_width=61,temp_height=61):
	c_max=c_corn-temp_width/2
	r_max=r_corn-temp_height/2

	try:
		Row,Col,Dim = numpy.shape(img_white)
		img_display = numpy.copy(img_white)   
		
	except:
		Row,Col = numpy.shape(img_white)
		img_display = numpy.zeros(shape=(Row,Col,3))
		img_display[:,:,0]=img_white
		img_display[:,:,1]=img_white
		img_display[:,:,2]=img_white

	if numpy.max(img_display) > 1:
		constant = 255.
	else:
		constant = 1.

	for r in range(temp_height):
		ROW=min(r_max+r,Row-1)
		img_display[ROW,min(Col-1,c_max-1),2]=constant
		img_display[ROW,min(Col-1,c_max),2]=constant
		img_display[ROW,min(Col-1,c_max+1),2]=constant
		img_display[ROW,min(Col-1,c_max+temp_width-1),2]=constant
		img_display[ROW,min(Col-1,c_max+temp_width),2]=constant
		img_display[ROW,min(Col-1,c_max+temp_width+1),2]=constant
		
		img_display[ROW,min(Col-1,c_max-1),1]=0.
		img_display[ROW,min(Col-1,c_max),1]=0.
		img_display[ROW,min(Col-1,c_max+1),1]=0.
		img_display[ROW,min(Col-1,c_max+temp_width-1),1]=0.
		img_display[ROW,min(Col-1,c_max+temp_width),1]=0.
		img_display[ROW,min(Col-1,c_max+temp_width+1),1]=0.
		
		img_display[ROW,min(Col-1,c_max-1),0]=0.
		img_display[ROW,min(Col-1,c_max),0]=0.
		img_display[ROW,min(Col-1,c_max+1),0]=0.
		img_display[ROW,min(Col-1,c_max+temp_width-1),0]=0.
		img_display[ROW,min(Col-1,c_max+temp_width),0]=0.
		img_display[ROW,min(Col-1,c_max+temp_width+1),0]=0.
		
	for c in range(temp_width):
		COL=min(c_max+c,Col-1)
		img_display[min(Row-1,r_max-1),COL,2]=constant
		img_display[min(Row-1,r_max),COL,2]=constant
		img_display[min(Row-1,r_max+1),COL,2]=constant
		img_display[min(Row-1,r_max+temp_height-1),COL,2]=constant
		img_display[min(Row-1,r_max+temp_height),COL,2]=constant
		img_display[min(Row-1,r_max+temp_height+1),COL,2]=constant
		
		img_display[min(Row-1,r_max-1),COL,1]=0
		img_display[min(Row-1,r_max),COL,1]=0
		img_display[min(Row-1,r_max+1),COL,1]=0
		img_display[min(Row-1,r_max+temp_height-1),COL,1]=0
		img_display[min(Row-1,r_max+temp_height),COL,1]=0
		img_display[min(Row-1,r_max+temp_height+1),COL,1]=0
		
		img_display[min(Row-1,r_max-1),COL,0]=0
		img_display[min(Row-1,r_max),COL,0]=0
		img_display[min(Row-1,r_max+1),COL,0]=0
		img_display[min(Row-1,r_max+temp_height-1),COL,0]=0
		img_display[min(Row-1,r_max+temp_height),COL,0]=0
		img_display[min(Row-1,r_max+temp_height+1),COL,0]=0
		
	return img_display/constant

def draw_bounding_boxG(img_white,c_corn,r_corn,temp_width=61,temp_height=61):
	c_max=c_corn-temp_width/2
	r_max=r_corn-temp_height/2

	try:
		Row,Col,Dim = numpy.shape(img_white)
		img_display = numpy.copy(img_white)   
		
	except:
		Row,Col = numpy.shape(img_white)
		img_display = numpy.zeros(shape=(Row,Col,3))
		img_display[:,:,0]=img_white
		img_display[:,:,1]=img_white
		img_display[:,:,2]=img_white

	if numpy.max(img_display) > 1:
		constant = 255.
	else:
		constant = 1.

	for r in range(temp_height):
		ROW=min(r_max+r,Row-1)
		img_display[ROW,min(Col-1,c_max-1),1]=constant
		img_display[ROW,min(Col-1,c_max),1]=constant
		img_display[ROW,min(Col-1,c_max+1),1]=constant
		img_display[ROW,min(Col-1,c_max+temp_width-1),1]=constant
		img_display[ROW,min(Col-1,c_max+temp_width),1]=constant
		img_display[ROW,min(Col-1,c_max+temp_width+1),1]=constant
		
		img_display[ROW,min(Col-1,c_max-1),2]=0.
		img_display[ROW,min(Col-1,c_max),2]=0.
		img_display[ROW,min(Col-1,c_max+1),2]=0.
		img_display[ROW,min(Col-1,c_max+temp_width-1),2]=0.
		img_display[ROW,min(Col-1,c_max+temp_width),2]=0.
		img_display[ROW,min(Col-1,c_max+temp_width+1),2]=0.
		
		img_display[ROW,min(Col-1,c_max-1),0]=0.
		img_display[ROW,min(Col-1,c_max),0]=0.
		img_display[ROW,min(Col-1,c_max+1),0]=0.
		img_display[ROW,min(Col-1,c_max+temp_width-1),0]=0.
		img_display[ROW,min(Col-1,c_max+temp_width),0]=0.
		img_display[ROW,min(Col-1,c_max+temp_width+1),0]=0.
		
	for c in range(temp_width):
		COL=min(c_max+c,Col-1)
		img_display[min(Row-1,r_max-1),COL,1]=constant
		img_display[min(Row-1,r_max),COL,1]=constant
		img_display[min(Row-1,r_max+1),COL,1]=constant
		img_display[min(Row-1,r_max+temp_height-1),COL,1]=constant
		img_display[min(Row-1,r_max+temp_height),COL,1]=constant
		img_display[min(Row-1,r_max+temp_height+1),COL,1]=constant
		
		img_display[min(Row-1,r_max-1),COL,2]=0
		img_display[min(Row-1,r_max),COL,2]=0
		img_display[min(Row-1,r_max+1),COL,2]=0
		img_display[min(Row-1,r_max+temp_height-1),COL,2]=0
		img_display[min(Row-1,r_max+temp_height),COL,2]=0
		img_display[min(Row-1,r_max+temp_height+1),COL,2]=0
		
		img_display[min(Row-1,r_max-1),COL,0]=0
		img_display[min(Row-1,r_max),COL,0]=0
		img_display[min(Row-1,r_max+1),COL,0]=0
		img_display[min(Row-1,r_max+temp_height-1),COL,0]=0
		img_display[min(Row-1,r_max+temp_height),COL,0]=0
		img_display[min(Row-1,r_max+temp_height+1),COL,0]=0
		
	return img_display/constant

def locateIMG_with_supporter(img2search, supporters, max_supporters_dectected_score, MODEL_main_supporter, DEBUG=False, size = 30, max_win_size = 100, classifier = 'rf'):
	[MODEL,MODEL_supporter] = MODEL_main_supporter
	row,col,dim = img2search.shape

	padded_img = numpy.zeros(shape=(row+(2*size),col+(2*size),dim))
	padded_img[size:-size,size:-size,:] = img2search
	img_predict = numpy.zeros(shape=(row,col))

	### compute supporter probability maps
	w_stary = 1.*numpy.arange(img2search.shape[0])
	postFuny = numpy.zeros(shape=img2search.shape[:2])
	w_starx = 1.*numpy.arange(img2search.shape[1])
	postFunx = numpy.zeros(shape=img2search.shape[:2])

	for each_supporter,each_score in zip(supporters,max_supporters_dectected_score):
		predVary = 10*each_score
		predVarx = 10*each_score
		predMeanY = each_supporter[1]
		predMeanX = each_supporter[0]

		for e_y in range(img2search.shape[0]):
			postFuny[e_y,:] = (1./numpy.sqrt(2*numpy.pi*predVary))*numpy.exp(-0.5*((w_stary[e_y]-predMeanY)**2)/predVary)
	
		for e_x in range(img2search.shape[1]):
			postFunx[:,e_x] = (1./numpy.sqrt(2*numpy.pi*predVarx))*numpy.exp(-0.5*((w_starx[e_x]-predMeanX)**2)/predVarx)

	postFunx[each_supporter[1],each_supporter[0]] = 0
	postFuny[each_supporter[1],each_supporter[0]] = 0

	### pass gaussian filters
	postFunx = gaussian_filter(postFunx,20)
	postFuny = gaussian_filter(postFuny,20)

	combined = (postFunx + postFuny)

	### compute the main pattern probability map
	pattern_prob = locateIMG(img2search, MODEL)

	### combine X and Y supporters
	img_predict = pattern_prob*combined

	### pass through softmax
	img_predict_pp = normalize(img_predict)
	
	if DEBUG:
		print supporters
		print max_supporters_dectected_score

		plt.imshow(pattern_prob)
		plt.title('pattern_prob')
		plt.show()
			
		plt.imshow(combined)
		plt.title('combined')
		plt.show()
			
		plt.imshow(img_predict)
		plt.title('img_predict')
		plt.show()

		plt.imshow(img_predict_pp)
		plt.title('img_predict after post-processed')
		plt.show()
	return img_predict_pp

def normalize(x):

	maxV = numpy.max(x)
	minV = numpy.min(x)
	rangeV = maxV - minV

	x_pp = (x - minV) / rangeV

	return x_pp


def learn_img_classifier_with_supporter(img, click_position, hardnegatives = list(), supporters = list(), neighbour = 3, size = 30, sample_percentage = 0.01):
	t0 = time.time()
	if not isinstance(click_position,list):
		click_position = [click_position] 

	### collect positive and negative
	row,col,dim = img.shape

	padded_img = numpy.zeros(shape=(row+(2*size),col+(2*size),dim))
	padded_img[size:-size,size:-size,:] = img

	pos = list()
	neg = list()
	for r in range(row):
		for c in range(col):
			patch = padded_img[r:r+(2*size)+1,c:c+(2*size)+1,:]
			first_pass = True

			for each_position in click_position:
				if r >= each_position[1] - neighbour and r <= each_position[1] + neighbour and \
					c >= each_position[0] - neighbour and c <= each_position[0] + neighbour:
					first_pass = False
					feature = numpy.zeros(shape=(11163 +(2*len(supporters)),))
					feature[:11163] = patch.flatten()

					for ind_sup,each_sup in enumerate(supporters):
						feature[11163+(2*ind_sup)] = c - each_sup[0]
						feature[11163+(2*ind_sup)+1] = r - each_sup[1]

					pos.append(feature)

				elif first_pass:
					first_pass = False
					if len(hardnegatives) > 0:
						if (c,r) in hardnegatives:
							feature = numpy.zeros(shape=(11163 +(2*len(supporters)),)) #11163 = (3*(((2*size)+1)^2))
							feature[:11163] = patch.flatten()
							for ind_sup,each_sup in enumerate(supporters):
								feature[11163+(2*ind_sup)] = c - each_sup[0]
								feature[11163+(2*ind_sup)+1] = r - each_sup[1]

							neg.append(feature)

						elif numpy.random.choice([True,False],p=[sample_percentage,1.0-sample_percentage]) and len(neg) < 1000: # randomly pick negative example
							feature = numpy.zeros(shape=(11163 +(2*len(supporters)),)) #11163 = (3*(((2*size)+1)^2))
							feature[:11163] = patch.flatten()

							for ind_sup,each_sup in enumerate(supporters):
								feature[11163+(2*ind_sup)] = c - each_sup[0]
								feature[11163+(2*ind_sup)+1] = r - each_sup[1]

							neg.append(feature)

					elif numpy.random.choice([True,False],p=[sample_percentage,1.0-sample_percentage]) or (c,r) in hardnegatives: # randomly pick negative example
						feature = numpy.zeros(shape=(11163 +(2*len(supporters)),)) #11163 = (3*(((2*size)+1)^2))
						feature[:11163] = patch.flatten()

						for ind_sup,each_sup in enumerate(supporters):
							feature[11163+(2*ind_sup)] = c - each_sup[0]
							feature[11163+(2*ind_sup)+1] = r - each_sup[1]

						neg.append(feature)

	X=numpy.vstack((numpy.squeeze(numpy.array(pos)),numpy.squeeze(numpy.array(neg))))
	y=numpy.vstack((numpy.ones(shape=(len(pos),1)),numpy.zeros(shape=(len(neg),1))))

	print 'collecting time',
	print time.time()-t0

	t0 = time.time()
	randomforest = RF(n_estimators=50,max_depth=5,class_weight= 'balanced')
	randomforest.fit(X[:,:11163],numpy.squeeze(y))

	if len(supporters)>0:
		sup_forest = RF(n_estimators=50,max_depth=5,class_weight= 'balanced')
		sup_forest.fit(X[:,11163:],numpy.squeeze(y))
	else:
		sup_forest = []

	print 'training time',
	print time.time()-t0

	return [randomforest,sup_forest]