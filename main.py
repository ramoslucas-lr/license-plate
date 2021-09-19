import os
import re

import cv2 as cv
import numpy as np
import copy
import pytesseract as ocr

detected_plates = set()
from APLR.frame_alpr import FrameAlpr
from APLR.cvao import cvao
l = 0
def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

def build_tesseract_options(psm=7):
    # tell Tesseract to only OCR alphanumeric characters
    alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    options = "-c tessedit_char_whitelist={}".format(alphanumeric)
    # set the PSM mode
    options += " --user-patterns /home/prim/Documents/lUCAS/xxx.patterns"
    options += " --psm {}".format(psm)
    options += " bazaar"



    # return the built options string
    return options

path  = 'C:/Users/ramos/dev/videos/videos/Camera1'
files = os.listdir(path)

cap = cv.VideoCapture(path+files[0])

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

size = (frame_width, frame_height)
print(size)

result = cv.VideoWriter('output.avi',
                                cv.VideoWriter_fourcc(*'MJPG'),
                                24, size)

frame_before = None
a = 0

while cap.isOpened():

    ret, frame = cap.read()

    input = FrameAlpr(frame)
    output = FrameAlpr(frame.copy())
    resized_frame = FrameAlpr(frame.copy())

    resized_frame = cvao.resize(res='hd', frame=resized_frame)

    if ret:
        if a == 0:
            frame_before = copy.deepcopy(resized_frame)

        pp_frame = cvao.preprocess(frame_before=frame_before, frame=resized_frame, debug=False)
        pp_frame = cvao.get_vehicles(pp_frame)

        scaled_vehicles = []
        for v in pp_frame.vehicles:
            scaled_vehicles.append(v.scale_vehicle(1.5))

        output.vehicles = scaled_vehicles
        output.contours = copy.copy(pp_frame.contours)
        output.scale_contours(1.5)
        output.draw_vehicles()


        for v in scaled_vehicles:
            bg = np.zeros(input.mat.shape, dtype="uint8")

#            cv.rectangle(bg, (v.x, v.y), (v.x + v.w, v.y + v.h), 255, -1)
            plates = FrameAlpr(input.mat.copy())
#            bitwiseAnd = cv.bitwise_and(bg, plates)

            crop_img = plates.mat[v.y:v.y+v.h, v.x:v.x+v.w]


            crop_img_gray = cv.cvtColor(crop_img, cv.COLOR_BGR2GRAY)
            candidates = cvao.locate_license_plate_candidates(crop_img_gray,)



            a = cvao.locate_license_plate(crop_img_gray, candidates)
            if a[0] is not None:
                b = FrameAlpr(a[0].copy())

                options = build_tesseract_options(psm=7)
                b = unsharp_mask(a[0])
                lpText = ocr.image_to_string(b, config=options)

                pattern = re.compile("(^\D{3}\d{4}$)")
                lpText = lpText.replace('\n', '').replace('\f', '')

                if pattern.match(lpText):

                    lpText = lpText[:3] + '-' + lpText[3:]
                    if lpText not in detected_plates:
                        detected_plates.add(lpText)
                        output.draw_license_plate(a[1], (v.x, v.y), lpText, v)
                        for i in range(48):
                            result.write(output.mat)






           # cv.imshow('output', output.mat)

        #        cv.imshow('Original', dmy)
#        cv.imshow('Diff', diff_image)
#        cv.imshow('Dilated', dilated)
#        cv.imshow('Thresh', thresh)
        #cv.imshow('Median', median)


        #if cv.waitKey(1) & 0xFF == ord('s'):
        #    break

        result.write(output.mat)
        frame_before = copy.deepcopy(resized_frame)
        a = 1
        l += 1
        print(l)
        if l == 24*60*1:
            break

    else:
        break

cap.release()
result.release()
cv.destroyAllWindows()

