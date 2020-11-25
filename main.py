import os
import cv2
import numpy as np

# A kep fekete szeleinek levagasa
def trim(frame):
    if not np.sum(frame[0]):
        return trim(frame[1:])

    if not np.sum(frame[-1]):
        return trim(frame[:-2])

    if not np.sum(frame[:,0]):
        return trim(frame[:,1:])

    if not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame

# Kepek beolvasasa, atmeretezese es szurkearnyalatossa tetele
img_ = cv2.imread('2.jpg')
img_ = cv2.resize(img_, (0,0),None,0.4,0.4,None)
img1 = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)

img = cv2.imread('1.jpg')
img = cv2.resize(img, (0,0),None,0.4,0.4,None)
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# Kulcspontok kiszamolasa SIFT algoritmus segitsegevel
sift = cv2.SIFT_create()
keypoint1, des1 = sift.detectAndCompute(img1, None)
keypoint2, des2 = sift.detectAndCompute(img2, None)

# Kulcspontok osszehasonlitasa FLANN(Gyors Konyvtar a Legkoyelebbi Szomszedok megtalalasara) segitsegevel
match=cv2.BFMatcher()
matches=match.knnMatch(des1,des2,k=2)

# Jo egyezesek kigyujtese es ezek abrazolasa a kepek kozott
good=[]
for m,n in matches:
    if m.distance<0.35*n.distance:
        good.append(m)

draw_parameters = dict(matchColor=(0,255,0), singlePointColor=None, flags=2)

img3=cv2.drawMatches(img_,keypoint1,img,keypoint2,good,None,**draw_parameters)

# A Kep homography tehat a ket kep kozos metszetenek megtalalasa
MIN_MATCH_COUNT=10
if len(good)>MIN_MATCH_COUNT:
    src_pts=np.float32([keypoint1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([keypoint2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts,dst_pts, cv2.RANSAC, 5.0)
    h,w=img1.shape
    pts=np.float32([ [0,0], [0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
    dst=cv2.perspectiveTransform(pts, M)
    img2=cv2.polylines(img2, [np.int32(dst)],True,255,3,cv2.LINE_AA)
    cv2.imshow("overlapping_image", img2)
else:
    print("Not enough matches for approppriate stitching")

# A kep perspektivajanak megvaltoztatasa
dst=cv2.warpPerspective(img_,M,(img.shape[1]+img_.shape[1], img.shape[0]))
dst[0:img.shape[0], 0:img.shape[1]]=img

# Az egyes lepesek es a vegeredmeny megjelenitese
cv2.imshow('sift_left', cv2.drawKeypoints(img_, keypoint1, None))
cv2.imshow('sift_right', cv2.drawKeypoints(img, keypoint2, None))
cv2.imshow('drawMatches',img3)
cv2.imshow('stitched',dst)
cv2.imshow('cropped',trim(dst))
cv2.waitKey()
cv2.destroyAllWindows()


