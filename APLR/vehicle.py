class Vehicle:
    def __init__(self, x, y, w, h, c):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

        self.c = c

    def scale_vehicle(self, factor):
        self.x = int(self.x * factor)
        self.y = int(self.y * factor)
        self.w = int(self.w * factor)
        self.h = int(self.h * factor)

        self.c = self.c[0] * factor, self.c[1] * factor,

        return self
