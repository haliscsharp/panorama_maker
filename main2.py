import os
import numpy as np
import cv2

mainFolder = 'Images'
myFolders = os.listdir(mainFolder)
print(myFolders)

finished=0
wrong=0

for folder in myFolders:
    path = mainFolder +'/'+folder
    images =[]
    myList = os.listdir(path)
    print(f'Total no of images detected {len(myList)}')
    for imgN in myList:
        curImg = cv2.imread(f'{path}/{imgN}')
        curImg = cv2.resize(curImg, (0,0),None,0.4,0.4,None)
        images.append(curImg)

    stitcher = cv2.Stitcher.create()
    (status,result)= stitcher.stitch(images)
    if status==cv2.Stitcher_OK:
        print('Done')
        finished=finished+1
        cv2.imshow(folder,result)
        cv2.waitKey()
        cv2.destroyAllWindows()
    else:
        print('Something went wrong')
        wrong=wrong+1

print('Successful stitches: ',finished, 'Failed stitches: ',wrong)


