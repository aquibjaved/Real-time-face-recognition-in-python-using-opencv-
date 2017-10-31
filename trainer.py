# facerec.py
import cv2, sys, numpy, os,pyttsx,time
size = 4
fn_haar = 'haarcascade_frontalface_default.xml'
fn_dir = 'database'

recognizer = cv2.createLBPHFaceRecognizer()
# Part 1: Create fisherRecognizer
print('Training...')
# Create a list of images and a list of corresponding names
(images, lables, names, id) = ([], [], {}, 0)
for (subdirs, dirs, files) in os.walk(fn_dir):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(fn_dir, subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            lable = id
	    images.append(cv2.imread(path, 0))
            lables.append(int(lable))
        id += 1
(im_width, im_height) = (112, 92)

# Create a Numpy array from the two lists above
(images, lables) = [numpy.array(lis) for lis in [images, lables]]


# model = cv2.reateFisherFaceRecognizer()
recognizer.train(images, lables)
recognizer.save('trainer.yml')

