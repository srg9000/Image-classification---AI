import numpy as np
import cv2
import os
import random
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

direc = "F:/Semester 6/AI/lab/l6 data/"

train = list(os.listdir('F:/Semester 6/AI/lab/l6 data/'))												#Create train and test set
test = list(os.listdir('F:/Semester 6/AI/lab/l6 data'))
test_num = int(0.2*len(train))

while(len(test)!=test_num):	
	test.pop(random.randint(0,len(test)-1))

[train.remove(x) for x in test]
train_label = [ x[0:8] for x in train ]
test_label = [ x[0:8] for x in test ]

file_name = []
face_matrix = np.ndarray((len(train), 600 * 600))
face_matrix_test = np.ndarray((len(test), 600 * 600))
res_list = {}


'''haar cascading '''
kernel = np.ones((5, 5), np.float32)
face_cascade = cv2.CascadeClassifier('F:/downloads/opencv-master/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml')


def convert(file):
	''' Convert given file to normalized form and size'''
	img = cv2.imread(direc + file)
	scale_percent = 50  # percent of original size
	width = int(img.shape[1] * scale_percent / 600)
	height = int(img.shape[0] * scale_percent / 600)
	dim = (width, height)
	img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)  											# prescale image

	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	faces = face_cascade.detectMultiScale(img, 1.3, 7, minSize=(128, 128)) 								# cascade
	if len(faces)==0:																					#if Haar-cascade fails to detect face
		smaller_dim=min(img.shape[0:2])
		center = (img.shape[0]//2, img.shape[1]//2)
		x = center[1] - smaller_dim//2
		w = smaller_dim
		y = center[0] - smaller_dim//2
		h = smaller_dim
		img = img[y:y + h, x:x + w]
		img = cv2.resize(img, (128, 128))
	else:
		for k in faces:  # cascade parameters
			areax = 0 
			try:																						#if multiple parts detected as faces, pick one with max area
				for elem in k:
					area = elem[-1]*elem[-2]
					if area > areax:
						areax=area
						mat = elem
				(x,y,w,h) = mat
				img = img[y:y + h, x:x + w]
				
				img = cv2.resize(img, (128, 128))
				cv2.imshow(file,img)
			except:																						#Face properly detected
				(x,y,w,h) = k
				img = img[y:y + h, x:x + w]
				img = cv2.resize(img, (128, 128))
				
				return img	
	
	file_name.append(file)
	return img

count = 0
for file in train:																						#Fill matrix with flattened images
	img = convert(file)
	face_matrix[count] = np.array(img.reshape(1, -1))
	count += 1
face_matrix_transpose = np.transpose(face_matrix)

pcax = PCA(n_components=20)																				#Perform PCA
pcay = pcax.fit(face_matrix)

eigen_vectors = pcay.components_
eigen_values = pcay.explained_variance_

pca_train = pcay.transform(face_matrix)																	#transform face matrix

train_label = np.transpose(np.array(train_label))

neigh = KNeighborsClassifier(n_neighbors=7)																# KNN classification with 7 nearest neighbors
neigh.fit((pca_train), np.ravel(np.array(train_label)))

count = 0
for file in test:																						#Pre-process and test images in test set
	img = convert(file)																					
	test_face = np.array(img.reshape(1, -1))
	test_face_transpose = np.transpose(test_face)
	face_matrix_test[count] = np.array(img.reshape(1, -1))
	count += 1

face_matrix_test_transpose = np.transpose(face_matrix_test)
count=0
pred=[]
for row in range(face_matrix_test.shape[0]):															#predict the class of image
	test_point = pcay.transform(face_matrix_test[row].reshape(1,-1))
	prediction = neigh.predict(np.array(test_point).reshape(1,-1))
	print(prediction , test_label[row])
	pred.append(prediction)
	if prediction[0]==str(test_label[row]):												
		count+=1
print ("Accuracy = ",count/test_num)																	#Accuracy

print(classification_report(test_label,pred , target_names=list(set(test_label))))						#precision, recall, f1-score and other reports
print(confusion_matrix(test_label, pred))																#Confusion matrix


