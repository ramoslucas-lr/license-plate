import cv2 as cv


class FrameAlpr:
    contours = []
    vehicles = []

    def __init__(self, mat_frame):
        self.mat = mat_frame


    def debug_imshow(self, title, waitKey=False):
        cv.imshow(title, self.mat)

        if waitKey:
            cv.waitKey(0)
        pass

    def draw_vehicles(self, debug=False):
        for v in self.vehicles:
            cv.rectangle(self.mat, (int(v.x), int(v.y)), (int(v.x) + int(v.w), int(v.y) + int(v.h)), (153, 51, 255), 2)
            cv.circle(self.mat, (int(v.c[0]), int(v.c[1])), 7, (255, 255, 255), -1)
            #cv.putText(self.mat, "center", (int(v.c[0]) - 20, int(v.c[1]) - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if debug:
            cv.drawContours(self.mat, self.contours, -1, (127, 200, 0), 2)

    def scale_contours(self, factor):
        scaled_contours = []

        for contour in self.contours:
            contour[:, :, 0] = contour[:, :, 0] * factor
            contour[:, :, 1] = contour[:, :, 1] * factor

            scaled_contours.append(contour)

        self.contours = scaled_contours

    def draw_license_plate(self, cntr, offset, lpText, v):
        #cv.drawContours(self.mat, cntr, -1, (0, 255, 0), 3, offset=offset)
        cv.putText(self.mat, lpText, (int(v.c[0]) - 20, int(v.c[1]) - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)




