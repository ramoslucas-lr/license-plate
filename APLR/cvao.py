import cv2 as cv
import imutils
import numpy as np
from skimage.segmentation import clear_border

from APLR.frame_alpr import FrameAlpr
from APLR.vehicle import Vehicle


class cvao:
    @staticmethod
    def resize(res, frame):

        if res.lower() == 'hd':
            b = cv.resize(frame.mat, (1280, 720), fx=0, fy=0, interpolation=cv.INTER_CUBIC)
            frame.mat = b.copy()

        return frame

    @staticmethod
    def preprocess(frame_before, frame, debug=False):
        output = FrameAlpr(None)

        grayA = cv.cvtColor(frame_before.mat, cv.COLOR_BGR2GRAY)
        grayB = cv.cvtColor(frame.mat, cv.COLOR_BGR2GRAY)

        diff_image = cv.absdiff(grayB, grayA)
        output.mat = diff_image

        if debug:
            output.debug_imshow('diff')

        ret, thresh = cv.threshold(diff_image, 20, 255, cv.THRESH_BINARY)
        output.mat = thresh

        if debug:
            output.debug_imshow('thresh')

        dst = cv.medianBlur(thresh, 5)
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv.dilate(dst, kernel, iterations=5)
        output.mat = dilated

        if debug:
            output.debug_imshow('dilated')

        output.mat = dilated.copy()

        return output

    @staticmethod
    def get_vehicles(frame):
        contours, hierarchy = cv.findContours(frame.mat.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        valid_cntrs = []
        vehicles = []
        for i, cntr in enumerate(contours):

            x, y, w, h = cv.boundingRect(cntr)

            if (cv.contourArea(cntr) >= 10000):
                M = cv.moments(cntr)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                invalid = False

                for v in vehicles:
                    if (cX > v.x and cX < (v.x + v.w)) and (cY > v.y and cY < (v.y + v.h)):
                        if v.w * v.h > h * w:
                            invalid = True

                if not invalid:
                    valid_cntrs.append(cntr)
                    vehicles.append(Vehicle(x, y, w, h, (cX, cY)))

        frame.contours = valid_cntrs
        frame.vehicles = vehicles
        return frame

    @staticmethod
    def locate_license_plate_candidates(gray, keep=5, debug=False):

        rectKern = cv.getStructuringElement(cv.MORPH_RECT, (10, 5))
        blackhat = cv.morphologyEx(gray, cv.MORPH_BLACKHAT, rectKern)
        if debug:
            FrameAlpr(blackhat).debug_imshow("Blackhat", True)

        squareKern = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        light = cv.morphologyEx(gray, cv.MORPH_CLOSE, squareKern)
        if debug:
            FrameAlpr(light).debug_imshow("light", True)
        light = cv.threshold(light, 85, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
        if debug:
            FrameAlpr(light).debug_imshow("light", True)

        gradX = cv.Sobel(blackhat, ddepth=cv.CV_32F, dx=1, dy=0, ksize=-1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
        gradX = gradX.astype("uint8")
        if debug:
            FrameAlpr(gradX).debug_imshow("Scharr", True)

        gradX = cv.GaussianBlur(gradX, (5, 5), 0)
        gradX = cv.morphologyEx(gradX, cv.MORPH_CLOSE, rectKern)
        thresh = cv.threshold(gradX, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
        if debug:
            FrameAlpr(thresh).debug_imshow("Grad Thresh", True)

        thresh = cv.erode(thresh, None, iterations=2)
        thresh = cv.dilate(thresh, None, iterations=2)
        if debug:
            FrameAlpr(thresh).debug_imshow("Grad Erode/Dilate", True)

        if debug:
            FrameAlpr(thresh).debug_imshow("Final", waitKey=True)

        cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,
                               cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv.contourArea, reverse=True)[:keep]

        return cnts

    @staticmethod
    def locate_license_plate(cropped, candidates, clearBorder=False):

        lpCnt = None
        roi = None

        for c in candidates:

            (x, y, w, h) = cv.boundingRect(c)
            ar = w / float(h)
            if ar >= 3 and ar <= 6:

                lpCnt = c
                licensePlate = cropped[y:y + h, x:x + w]

                roi = licensePlate

                if clearBorder:
                    roi = clear_border(roi)
                break

        return (roi, lpCnt)

