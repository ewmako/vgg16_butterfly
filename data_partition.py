import os
import cv2
import numpy as np
from scipy.misc import imread, imresize

############################### PODZIAŁ DANYCH NA TESTOWE I TRENINGOWE ###############################

#sciezka do folderu ze zdjeciami
path_imgs = 'leedsbutterfly/images'

#sciezka do folderu w ktorym zapiszemy dane testowe UWAGA TRZEBA STWORZYC FOLDER- SKRYPT TEGO NIE ROBI
path_test_data = 'leedsbutterfly/test_data'
	
#sciezka do folderu w ktorym zapiszemy dane treningowe UWAGA TRZEBA STWORZYC FOLDER- SKRYPT TEGO NIE ROBI
path_train_data = 'leedsbutterfly/train_data'

if __name__ == '__main__':

    data = []
    
    for _ in range(10):
        data.append([])
     
    for filename in os.listdir(path_imgs):
        if not filename.endswith('png'):
            continue
	#etykieta to nr motyla-1 (zeby je numerowac od 0 do 9)
        label = int(filename[:3]) - 1
        data[label].append(filename)
                
    for i in range(10):
	#permutacja listy
        np.random.shuffle(data[i])
        num_data = len(data[i])
        print 'Klasa: %03d Licza zdjec: %03d' % (i, num_data)
	#podzial 20:80 z dokladnoscia do calych zdjec (zbior testowy zaokraglamy w dol)
        num_test_data = int(num_data * 0.2)
        print 'Klasa: %03d Licza zdjec test: %03f ' % (i, num_test_data)
        counter = 0
        for n in data[i]:
            img = cv2.imread(os.path.join(path_imgs, n), 1)
		#zapisz w folderze testowym
            if counter < num_test_data:
                cv2.imwrite(os.path.join(path_test_data , n), img)
                counter += 1
		#zapisz w folderze treningowym
            else:
                cv2.imwrite(os.path.join(path_train_data , n), img)
