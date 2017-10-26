import cv

import kalcommon as kc
from kalcommon import lerp

class Drawing():
    """
    Helps us draw frames and write video
    """

    def __init__(self, output_filename='sim.avi', draw_estimates=['cam1', 'cam2', 'kalman', 'average']):
        """
        """
        self._vidWriter = cv.CreateVideoWriter(output_filename,
                                               kc.codec,
                                               kc.fps,
                                               kc.img_size)

        self._draw_estimates = draw_estimates

        self._font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, .5, .5)
        (co,cw,cs) = (10,30,60)
        (w,h) = kc.img_size
        self.cam1center = (co, co)
        self.cam2center = (w-co, co)
        self.cam3center = (co, h-co)


        # status texts and stuffs:
        self.status_text = ""
        self.ball_coords = (0,0)
        self.cam1_estimate = (0,0)
        self.cam2_estimate = (0,0)
        self.cam3_estimate = (0,0)
        self.frame_num = 0

    def write_frame(self, img):
        """
        Write a frame to our video
        """
        self.frame_num += 1
        # cv.SaveImage("testimage%d.jpg" % self.frame_num, img)
        cv.WriteFrame(self._vidWriter, img)


    def get_base_image(self):
        """
        Returns image with cameras drawn on background
        """
        img = cv.CreateImage(kc.img_size, kc.img_depth, kc.img_channels)
        self.paint_cameras(img)
        self.paint_text(img)
        self.paint_errors(img)
        self.paint_ball(img)
        self.paint_estimates(img)
        return img

    def paint_cameras(self, img):
        "paints the cameras onto the image"
        (co,cw,cs) = (10,30,60)
        (w,h) = kc.img_size
        coords = [self.cam1center, (cs,cw), (cw, cs)]
        cv.FillConvexPoly(img, coords, kc.cam1color)
        coords = [self.cam2center, (w-cs,cw), (w-cw, cs)]
        cv.FillConvexPoly(img, coords, kc.cam2color)
        # coords = [self.cam3center, (cs,h-cw), (cw, h-cs)]
        # cv.FillConvexPoly(img, coords, kc.cam3color)

    def paint_text(self, img):
        "paints the text onto the image"
        cv.PutText(img, self.status_text, (10,100), self._font, kc.font_color)
        cv.PutText(img, "Actual ball coordinates: (%d,%d)" % self.ball_coords, (10,150), self._font, kc.font_color)
        cv.PutText(img, "      Camera 1 estimate: (%d,%d)" % self.cam1_estimate, (10,170), self._font, kc.font_color)
        cv.PutText(img, "      Camera 2 estimate: (%d,%d)" % self.cam2_estimate, (10,190), self._font, kc.font_color)
        # cv.PutText(img, "      Camera 3 estimate: (%d,%d)" % self.cam3_estimate, (10,210), self._font, kc.font_color)
        cv.PutText(img, " Kalman filter estimate: (%d,%d)" % self.ball_coords, (10,230), self._font, kc.font_color)
        cv.PutText(img, "                  Frame: %d" % self.frame_num, (10,250), self._font, cv.RGB(0,0,0))

    def paint_errors(self, img):
        error_range = (0.0,100.0)
        output_range = (0.0,30.0)
        xoffset = 10
        cam1error = kc.get_dist_between_2_points(self.ball_coords, self.cam1_estimate)
        cam2error = kc.get_dist_between_2_points(self.ball_coords, self.cam2_estimate)
        # cam3error = kc.get_dist_between_2_points(self.ball_coords, self.cam3_estimate)
        cv.Rectangle(img, (xoffset + int(lerp(cam1error, error_range, output_range)), 165), (xoffset, 175), kc.cam1color, cv.CV_FILLED)
        cv.Rectangle(img, (xoffset + int(lerp(cam2error, error_range, output_range)), 185), (xoffset, 195), kc.cam2color, cv.CV_FILLED)
        # cv.Rectangle(img, (xoffset + int(lerp(cam3error, error_range, output_range)), 205), (xoffset, 215), kc.cam3color, cv.CV_FILLED)

    def paint_ball(self, img):
        "Paints the ball"
        cv.Circle(img, self.ball_coords, 50, kc.ballcolor, -1)

    def paint_estimates(self, img):
        "paints the camera estimates"
        circles = []
        if 'cam1' in self._draw_estimates:
            cv.Circle(img, self.cam1_estimate, 10, kc.cam1color, -1)
            cv.Line(img, self.cam1_estimate, self.cam1center, kc.cam1color, 1, cv.CV_AA)
            circles.append({'color':kc.cam1color, 'text':'Camera 1'})

        if 'cam2' in self._draw_estimates:
            cv.Circle(img, self.cam2_estimate, 10, kc.cam2color, -1)
            cv.Line(img, self.cam2_estimate, self.cam2center, kc.cam2color, 1, cv.CV_AA)
            circles.append({'color':kc.cam2color, 'text':'Camera 2'})

        # cv.Circle(img, self.cam3_estimate, 10, kc.cam3color, -1)
        # cv.Line(img, self.cam3_estimate, self.cam3center, kc.cam3color, 1, cv.CV_AA)

        if 'kalman' in self._draw_estimates:
            cv.Circle(img, self.kalman_estimate, 10, kc.kalmancolor, -1)
            circles.append({'color':kc.kalmancolor, 'text':'Kalman'})

        if 'average' in self._draw_estimates:
            cv.Circle(img, self.average_estimate, 10, kc.averagecolor, -1)
            circles.append({'color':kc.averagecolor, 'text':'Average'})

        circles.reverse()
        for cnt,c in enumerate(circles):
            yloc = kc.img_size[1] - 20 - cnt*25
            cloc = (200, yloc)
            cv.Circle(img, cloc, 10, c['color'], -1)
            tloc = (220, yloc+5)
            cv.PutText(img, c['text'], tloc, self._font, kc.font_color)


if __name__ == '__main__':
    import random

    print 'creating an instance of Drawing...'
    d = Drawing()

    print 'writing out some circle frames...'
    seconds = 5
    numframes = seconds * kc.fps
    img = d.get_base_image()
    # create 5 seconds worth of a circle bouncing around:
    for i in xrange(numframes):
        coords = (lerp(i, (0,numframes), (0,kc.img_size[0])), lerp(i, (0,numframes), (0,kc.img_size[1])))
        cv.Circle(img, coords, 10 + int(random.gauss(0,2)), cv.RGB(255,100,0), -1)
        d.paint_cameras(img)
        d.write_frame(img)
    seconds = 5
    numframes = seconds * kc.fps
    # create 5 seconds worth of a circle bouncing around:
    for i in xrange(numframes):
        coords = (lerp(i, (numframes,0), (0,kc.img_size[0])), lerp(i, (0,numframes), (0,kc.img_size[1])))
        cv.Circle(img, coords, 10, cv.RGB(0,100,255), -1)
        d.paint_cameras(img)
        d.write_frame(img)

    print 'all done!'
