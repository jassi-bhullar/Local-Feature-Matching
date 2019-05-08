#-------Author : Sanchit Jalan-----------------

import cv2
import numpy as np
import sys
import getopt
import scipy
from scipy import ndimage
import math
import operator
import datetime
from numpy import zeros
from student_sift import get_features


def readImage(filename):
    """
     Read in an image file, errors out if we can't find the file
    :return: Img object if filename is found
    """

    img = cv2.imread(filename, 0)
    if img is None:
        print('Invalid image:' + filename)
        return None
    else:
        return img

#---------------Finding corners--------------------
def findCorners(img, window_shape, k, thresh):
    """
    Finds and returns list of corners and new image with corners drawn

    """
    #Find x and y derivatives

    dy, dx = np.gradient(img)
    Ixx = dx**2
    Ixy = dy*dx
    Iyy = dy**2
    height = img.shape[0]
    width = img.shape[1]

    cornerList = []
    newImg = img.copy()
    color_img = cv2.cvtColor(newImg, cv2.COLOR_GRAY2RGB)
    # cv2.imwrite("lo",color_img)
    offset = window_shape/2

    #Loop through image and find our corners
    print "Finding Corners..."
    for y in range(offset, height-offset):
        for x in range(offset, width-offset):
            #Calculate sum of squares
            windowIxx = Ixx[y-offset:y+offset+1, x-offset:x+offset+1]
            windowIxy = Ixy[y-offset:y+offset+1, x-offset:x+offset+1]
            windowIyy = Iyy[y-offset:y+offset+1, x-offset:x+offset+1]
            Sxx = windowIxx.sum()
            Sxy = windowIxy.sum()
            Syy = windowIyy.sum()

            #Find determinant and trace, use to get corner response
            det = (Sxx * Syy) - (Sxy**2)
            trace = Sxx + Syy
            r = det - k*(trace**2)

            #If corner response is over threshold, color the point and add to corner list
            if r > thresh:
                # print x, y, r
                cornerList.append([x, y, r])
                # color_img.itemset((y, x, 0), 255)
                # color_img.itemset((y, x, 1), 0)
                # color_img.itemset((y, x, 2), 0)
    # cv2.imwrite("x",color_img)
         
    # print "Height and width of image is: "+ str(height) + "  "+ str(width) 

    return color_img, cornerList

#-------adaptive non-maximal suppression---------------------------------
def anms(cornerList):
    print "Applying anms, Suppressing corners...."
    l = len(cornerList)
    finalcornerList=[]
    finalcornerList.append([cornerList[l-1][0],cornerList[l-1][1],10000])
    for i in range(l-2,-1,-1):
        mini=10000
        for j in range(i+1,l,1):
            dist= math.sqrt( (cornerList[i][0]-cornerList[j][0])**2 + (cornerList[i][1]-cornerList[j][1])**2 )
            if dist<mini:
                mini=dist
        finalcornerList.append([cornerList[i][0], cornerList[i][1], mini])
    return finalcornerList



#----------euclidean distance-----------------
def euclidean_dist(f1,f2):
    return np.linalg.norm(f1 - f2)

#---------Ration test---------------
def ratioTest(d1,d2):
    ratio=d1/d2
    # print "hello world " + str(ratio)
    if ratio<=0.965:
        return 1
    else:
        return 0

#-----------Matcing number of features--------------------
def match_features(features1, features2):
    l1=features1.shape[0]
    l2=features2.shape[0]
    match=[]
    confidence=[]
    for i in range(0,l2):
        dist=[]
        for j in range(0,l1):
            dist.append([i,j,euclidean_dist(features2[i],features1[j])])
        dist.sort(key=operator.itemgetter(2))
        r=ratioTest(dist[0][2],dist[1][2])
        # print "hello world " + str(r)
        if r==1:
            match.append([dist[0][0],dist[0][1]])
            confidence.append(dist[0][2])
    # return match.shape[0],confidence
    return len(confidence)

def obtainFeatures(filename):


    # print ("Script started at :" +  str(datetime.datetime.now()) )
    window_shape=2;
    k_corner_response=0.06;
    corner_threshold=2000000000;

    # print("Image Name: " + str(filename))
    # print("Window Size: " + str(window_shape))
    # print("K Corner Response: " + str(k_corner_response))
    # print("Corner Response Threshold:" + str(corner_threshold))

    img = readImage(filename)

    if img is not None:
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if len(img.shape) == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        # print "Shape: " + str(img.shape)
        # print "Size: " + str(img.shape)
        # print "Type: " + str(img.dtype)
        # print "Printing Original Image..."
        # print(img)
        finalImg, cornerList = findCorners(img, int(window_shape), float(k_corner_response), int(corner_threshold))
        fimg=finalImg.copy()

        # Write top 100 corners to file
        cornerList.sort(key=operator.itemgetter(2))
        outfile = open('corners.txt', 'w')
        print "Size of cornerlist is: "+ str(len(cornerList))


        # print("Writing corners to txt file.....")
        l=len(cornerList)
        for i in range(l-1,l-1000,-1):
            outfile.write(str(cornerList[i][0]) + ' ' + str(cornerList[i][1]) + ' ' + str(cornerList[i][2]) + '\n')
            x=cornerList[i][0]
            y=cornerList[i][1]
            finalImg.itemset((y, x, 0), 0)
            finalImg.itemset((y, x, 1), 0)
            finalImg.itemset((y, x, 2), 255)
        outfile.close()

        if finalImg is not None:
            cv2.imwrite("finalimage.png", finalImg)

        l=len(cornerList)
        print str(l) + " corner points detected "

        finalcornerList=anms(cornerList)
        finalcornerList.sort(key=operator.itemgetter(2))
        l=len(finalcornerList)

        # print("Writing finalcorners to file finalcorners.txt.....")
        
        # outfile = open('finalcorners.txt', 'w')
        # for i in range(l-1,-1,-1):
        #     outfile.write(str(finalcornerList[i][0]) + ' ' + str(finalcornerList[i][1]) + ' ' + str(finalcornerList[i][2]) + '\n')
        # outfile.close()


        x_v=[]
        y_v=[]
        for i in range(l-1,l-1-500,-1):
            x=finalcornerList[i][0]
            y=finalcornerList[i][1]
            x_v.append(x)
            y_v.append(y)
            fimg.itemset((y, x, 0), 0)
            fimg.itemset((y, x, 1), 0)
            fimg.itemset((y, x, 2), 255)

        x_v=np.array(x_v)
        y_v=np.array(y_v)

        if fimg is not None:
            cv2.imwrite("fimg.png", fimg)

        feature_width=16
        # x_v=[item[0] for item in finalcornerList]
        # y_v=[item[1] for item in finalcornerList]

        feature_descriptors=get_features(img, x_v, y_v, feature_width)
        print feature_descriptors.shape

        # outfile = open('checkk.txt', 'w')
        # for i in range(0,feature_descriptors.shape[0]):
        #     outfile.write(str(feature_descriptors[i]) + '\n')
        # outfile.close()

        # print ("Script completed at :" +  str(datetime.datetime.now()) )
        return feature_descriptors


def main():
    # image1 = 'ear_image.bmp'

    #----------------Checking score between two images--------------------
    # image1 = "125/113_1.bmp"
    # image2 = '125/113_3.bmp'
    # feature1 = obtainFeatures(image1)
    # feature2 = obtainFeatures(image2)
    # # print feature1.shape
    # # print feature2.shape
    # outfile = open('f1.txt', 'w')
    # for i in range(0,feature1.shape[0]):
    #         outfile.write(str(feature1[i]) + '\n')
    # outfile.close()
    # outfile = open('f2.txt', 'w')
    # for i in range(0,feature2.shape[0]):
    #         outfile.write(str(feature2[i]) + '\n')
    # outfile.close()    
    # score=match_features(feature1,feature2)
    # print "final score "
    # print score


    #-------------------Training algorithm--------------------------
    trained_features=[[[0]*180]*500]*126
    for i in range(1,126):
        print i
        if i<=9:
            file = "125/00"+ str(i) +"_1.bmp"
        elif i<=99:
            file = "125/0"+str(i)+"_1.bmp"
        else :
            file = "125/"+str(i)+"_1.bmp"
        feature=obtainFeatures(file)
        trained_features[i] = feature



    #-------------------------Testing datasets---------------------
    similarity_threshold=200
    true_positive=0
    false_negative=0
    false_positive=0
    true_negative=0

    for i in range(1,126):
        for j in range(2,4):
            if i<=9:
                file = "125/00"+ str(i) +"_" + str(j) + ".bmp"
            elif i<=99:
                file = "125/0"+ str(i) +"_" + str(j) + ".bmp"
            else :
                file = "125/"+ str(i) +"_" + str(j) + ".bmp"

            img = cv2.imread(filename, 0)
            if img is None:
                continue
            else:
                feature1=obtainFeatures(filename)
                for k in range(1,126):
                    feature2=trained_features[k]
                    score=match_features(feature1,feature2)
                    if score > similarity_threshold:
                        if i==k:
                            true_positive+=1
                        else:
                            false_positive+=1
                    else :
                        if i==k:
                            false_negative+=1
                        else:
                            true_negative+=1




    
    #------------------Classification performance--------------------------
    true_positive_rate = true_positive/(true_positive + false_negative)
    false_positive_rate = false_positive/(false_positive + true_negative)
    positives = true_positive + false_positive
    negatives = true_negative + false_negative 
    accuracy = true_positive + true_negative / (positives + negatives)

    print "Performance evauation parameters are as follows : "
    print "Accuracy is : " + accuracy
    print "True positive rate is(TPR) or Sensitivity :  " + str(true_positive_rate)
    print "False positive rate is(FPR) : " + str(false_positive_rate)
    print "Specificity is : " + str(1- false_positive_rate )


if __name__ == "__main__":
    main()