#keypoint detection, local invariant desctiptors, keypoint matching, RANSAC, perspective warping
#different versions of opencv handle keypoints differently(SIFT, SURF)
#
#1. Detect keypoints(DoG, Harris) and extract local invariant descriptors(SIFT, SURF), from the input images
#2. Match the descriptors between the two images
#3. Use the RANSAC algorithm to estimate the homography matrix using matched feature vectors
#4. Apply a warping transformation using the homography matrix obtained from step 3

import cv2
import numpy as np
import imutils

class Stitcher:
	def __init__(self):

		self.isv3=imutils.is_cv3(or_better=True)
	def stitch(self, images, ratio=0.75, reprojThresh=4.0, showMatches=False):
			
		(left, right)=images
		(kpsleft, featuresleft)=self.detectAndDescribe(left)
		(kpsright, featuresright)=self.detectAndDescribe(right)

		M=self.matchKeypoints(kpsleft,kpsright,featuresleft,featuresright,ratio,reprojThresh)

		if M is None:
			return None
		else:
			(matches, H, status)=M
			result=cv2.warpPerspective(left, H, (left.shape[1]+right.shape[1],left.shape[0]))
			result[0:right.shape[0], 0:right.shape[1]]=right

			if showMatches:
				vis = self.drawMatches(left, right, kpsleft,kpsright,matches,status)
				
		return(result,vis)

	def detectAndDescribe(self, image):
		gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		if self.isv3:
			descriptor=cv2.xfeatures2d.SIFT_create()
			(kps, features)=descriptor.detectAndCompute(image, None)
		else:
			detector=cv2.FeatureDetector_create("SIFT")
			kps=detector.detect(gray)
			extractor=cv2.DescriptorExtractor_create("SIFT")
			(kps,features)=extractor.compute(gray, kps)
			kps=np.float32([kp.pt for kp in kps])
			return(kps, features)

	def matchKeypoints(self, kpsleft,kpsright,featuresleft,featuresright,ratio,reprojThresh):
		matcher=cv2.DescriptorMatcher_create("BruteForce")
		rawMatches=matcher.knnMatch(featuresleft,featuresright,2)
		matches=[]

		for m in rawMatches:
			if len(m)==2 and m[0].distance<m[1].distance*ratio:
				matches.append((m[0].trainIdx, m[0].queryIdx))
		if len(matches)>4:
			ptsleft=np.float32([kpsleft[i] for (_, i)in matches])
			ptsright=np.float32([kpsright[i] for (i, _)in matches])
		
		(H,status)=cv2.findHomography(ptsleft,ptsright,cv2.RANSAC,reprojThresh)
		return (matches,H,status)

		return None

	def drawMatches(self, left,right,kpsleft,kpsright,matches,status):
		(hleft,wleft)=left.shape[:2]
		(hright,wright)=right.shape[:2]
		vis=np.zeos((mac(hleft,hright),wleft,wright,3),dtype="uint8")
		vis[0:hleft,0:wleft]=left
		vis[0:hright,wleft]=right

		for((trainIdx, queryIdx), s) in zip(matches, status):
			if s==1:
				ptleft = (int(kpsleft[queryIdx][0]), int(kpsleft[queryIdx][1]))
				ptright = (int(kpsright[trainIdx][0]) + wA, int(kpsright[trainIdx][1]))
				cv2.line(vis, ptleft, ptright, (0, 255, 0), 1)
		return vis

imageLeft=cv2.imread("left.jpg")
imageRight=cv2.imread("right.jpg")
imageLeft = imutils.resize(imageLeft, width=400)
imageRight = imutils.resize(imageRight, width=400)

stitcher=Stitcher()
(result,vis)=stitcher.stitch([imageLeft,imageRight],showMatches=True)

cv2.imshow('ImageLeft', imageLeft)
cv2.imshow('ImageRight', imageRight)
cv2.imshow('Keypoint Matches', vis)
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()