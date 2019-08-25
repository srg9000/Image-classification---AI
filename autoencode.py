import numpy as np
import cv2
import os
import random
import keras
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model, load_model
from keras.optimizers import RMSprop, SGD, Adadelta
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from keras import metrics


def normalize():
	for _ in list(os.listdir()):
		l = str.upper(_).split('_')
		if l[2][0:3] == 'ANG' or l[2][0:1] == 'A':
			l[2] = 'ANGRY'
		elif l[2][0:2] == 'SU' or l[2][0:2] == 'SH':
			l[2] = 'SURPRISE'
		elif l[2][0:3] == 'SAD':
			l[2] = 'SAD'
		elif l[2][0:1] == 'N':
			l[2] = 'NEUTRAL'
		elif l[2][0:3] == 'HAP':
			l[2] = 'HAPPY'
		elif l[2][0:3] == 'DIS':
			l[2] = 'DISGUSTED'
		else:
			l[2] = 'FEAR'
		dst = '_'.join(l)
		os.rename(_, dst)


def shutdown():
	os.system("shutdown /s /t 1");


def getModel(size):
	input_img = Input(shape=(200, 200, 1))
	x = Conv2D(128, (3, 3), activation='relu', padding='same')(input_img)
	x = MaxPooling2D((2, 2), padding='same')(x)
	x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
	x = MaxPooling2D((2, 2), padding='same')(x)
	x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
	encoded = MaxPooling2D((2, 2), padding='same')(x)

	# at this point the representation is (4, 4, 8) i.e. 128-dimensional

	x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
	x = UpSampling2D((2, 2))(x)
	x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
	x = UpSampling2D((2, 2))(x)
	x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
	x = UpSampling2D((2, 2))(x)
	decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

	autoencoder = Model(input_img, decoded)
	opt = Adadelta(lr = 0.3)#SGD(lr=0.05, momentum=0.5, nesterov=True)
	autoencoder.compile(optimizer=opt, loss='binary_crossentropy', metrics=['acc', 'mae'])
	# print(autoencoder.summary())
	return autoencoder


direc = "F:/Semester 6/AI/proj/converted2/"
train = list(os.listdir(direc))
test = list(os.listdir(direc))
test_num = int(0.2 * len(train))

while (len(test) != test_num):
	test.pop(random.randint(0, len(test) - 1))

[train.remove(x) for x in test]

train_label = [str.upper(x.split('_')[1]) for x in train]
test_label = [str.upper(x.split('_')[1]) for x in test]

# classes_train = {'ANGRY': [], 'DISGUSTED': [], 'SAD': [], 'HAPPY': [], 'NEUTRAL': [], 'FEAR': [], 'SURPRISE': []}
# classes_test = {'ANGRY': [], 'DISGUSTED': [], 'SAD': [], 'HAPPY': [], 'NEUTRAL': [], 'FEAR': [], 'SURPRISE': []}
classes_train = { "MALE":[], "FEMALE" : []}
classes_test = { "MALE":[], "FEMALE" : []}
# train_angry = []
# train_disg = []
# train_sad = []
# train_happy = []
# train_neut = []
# train_fear = []
# train_surprise = []
f = open("save.txt", 'w+')
trained = {}
for i in train:
	k = i.split('_')[1]
	classes_train[k].append(i)
for i in test:
	k = i.split('_')[1]
	classes_test[k].append(i)

for _ in classes_train.keys():
	autoencoder = getModel(len(classes_train[_]))
	l = []
	ll = []
	for xx in classes_train[_]:
		img = cv2.imread(direc + xx, 0)
		img = np.array(img).astype('float32') / 255
		img = np.reshape(img, (200, 200, 1))
		l.append(img)
	#	print(xx)
	for xx in test:
		img = cv2.imread(direc + xx, cv2.IMREAD_GRAYSCALE)
		img = np.array(img).astype('float32') / 255
		img = np.reshape(img, (200, 200, 1))
		ll.append(img)
	print(_)
	autoencoder.fit(np.array(l), np.array(l),
	                epochs=50,
	                batch_size=len(classes_train[_])//4,
	                # shuffle=True,
	                validation_split=0.05,
	                #	                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')],
	                verbose=2)
	#	autoencoder = load_model("ANGRY")
	decoded_imgs = autoencoder.predict(np.array(ll))
	#	print(decoded_imgs)
	lst = {}
	for xv in range(len(ll)):
		try:
			lst[test_label[xv]].append(
				(mean_absolute_error(np.array(ll[xv]).reshape((200, 200)), decoded_imgs[xv].reshape(200, 200))))
		except:
			lst[test_label[xv]] = (
				mean_absolute_error(np.array(ll[xv]).reshape((200, 200)), decoded_imgs[xv].reshape(200, 200)))
	# print(lst, file=f, end='\n')
	trained[_] = autoencoder
	autoencoder.save(_)

    
f.close()

male = load_model("MALE")
female = load_model("FEMALE")
output = []
output2 = []
n=0
for xv in range(len(test)):
	opm = male.predict(np.array([ll[xv]]))
	opf = female.predict(np.array([ll[xv]]))
	if n<3:
		cv2.imwrite('decm'+test[xv], (opm*255).reshape((200,200,)).astype('int16'))
		cv2.imwrite('decf'+test[xv], (opf*255).reshape((200,200,)).astype('int16'))
		n+=1
	merr = mean_absolute_error(np.array(ll[xv]).reshape((200, 200)), opm.reshape(200, 200))
	mereu = np.linalg.norm(np.array(ll[xv]).reshape((200, 200)) -  opm.reshape(200, 200))
	ferr = mean_absolute_error(np.array(ll[xv]).reshape((200, 200)), opf.reshape(200, 200))
	fereu = np.linalg.norm(np.array(ll[xv]).reshape((200, 200)) -  opf.reshape(200, 200))
	if merr<ferr:
		output.append("MALE")
	else:
		output.append("FEMALE")
	if mereu<fereu:
		output2.append("MALE")
	else:
		output2.append("FEMALE")	
count=0
for xx in range(len(output)):
		if output[xx]==test_label[xx]:
			count+=1
print(count,len(output),(count/len(output)*100))
count=0
for xx in range(len(output2)):
		if output2[xx]==test_label[xx]:
			count+=1
print(count,len(output2),(count/len(output2)*100))