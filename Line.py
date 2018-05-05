import numpy as np

class Line():
    """This class tracks """
    def __init__(self):
        #was there lines detected in previous frames?
        self.detected = False  
        #Array of recent y and x points for left and right lines
        self.leftx = None
        self.lefty = None
        self.rightx = None
        self.righty = None
        #are the last added values valid
        self.valid_new = False
        #dimensions of the images
        self.dim = None
        #conversion parameters from pixel to actual dimesions
        self.ym_per_pix = None
        self.xm_per_pix = None
        #curvature of left and right lanes
        self.right_curv = None
        self.left_curv = None
        #right and left polynomials of the most recent fit in actual dimensions
        self.right_poly = [np.array([False])]  
        self.left_poly = [np.array([False])]  
        #average curvature of left and right lanes in actual dimensions
        self.avg_right_curv = None
        self.avg_left_curv = None
        #radius of curvature of the line in  actual dimensions
        self.radius_of_curvature = None 
        #polynomial coefficients averaged in actual dimensions
        self.avg_right_poly = [np.array([False])]  
        self.avg_right_poly = [np.array([False])]
        

    def find_curvature(self, mode='recent'):
        #accepts polynomial fit and pixel to actual dimension conversion parameters and returns curveture on the polynomial in actual dimensions 

        #defining evaluation point for y
        y_eval = self.dim[0]

        if (mode=='recent'):
            # Calculate the new radii of curvature of left and right lines
            self.right_curv = ((1 + (2*self.right_poly [0]*y_eval*self.ym_per_pix + self.right_poly [1])**2)**1.5) / np.absolute(2*self.right_poly[0])
            self.left_curv = ((1 + (2*self.left_poly [0]*y_eval*self.ym_per_pix + self.left_poly [1])**2)**1.5) / np.absolute(2*self.left_poly[0])
        elif (mode=='avg'):
            # Calculate the new radii of curvature of left and right lines
            self.avg_right_curv = ((1 + (2*self.right_poly [0]*y_eval*self.ym_per_pix + self.right_poly [1])**2)**1.5) / np.absolute(2*self.right_poly[0])
            self.avg_left_curv = ((1 + (2*self.left_poly [0]*y_eval*self.ym_per_pix + self.left_poly [1])**2)**1.5) / np.absolute(2*self.left_poly[0])
            self.radius_of_curvature = (self.avg_right_curv + self.avg_left_curv) / 2
         
    def sanity_check(self):

        self.find_curvature(mode='recent')
        #comparing left and right curvs
        if ((self.right_curv > 4000) & (self.right_curv > 4000)):
            #straigh line case
            curv_error = 0
        else:
            curv_error = abs((self.right_curv - self.left_curv) / ((self.right_curv + self.left_curv)/2))
        
        #comparing to previous values
        #defining error values
        right_curv_error = 0
        left_curv_error = 0

        if (self.detected):
            right_curv_error = abs( (self.avg_right_curv - self.right_curv) / self.avg_right_curv )
            left_curv_error = abs( (self.avg_left_curv - self.left_curv) / self.avg_left_curv )

        #appending new values
        if ( (curv_error < 0.65) & (right_curv_error < 0.1) & (left_curv_error < 0.1) ):
            self.valid_new = True

    def find_avg(self):
        if (self.detected):
            self.avg_right_poly = 0.9 * self.avg_right_poly + 0.1 * self.right_poly
            self.avg_left_poly = 0.9 * self.avg_left_poly + 0.1 * self.left_poly
        else:
            self.avg_right_poly = self.right_poly
            self.avg_left_poly = self.left_poly
            self.detected = True

    def fit_poly(self):
        # Fit a second order polynomial to each
        self.right_poly = np.polyfit(self.lefty, self.leftx, 2)
        self.left_poly = np.polyfit(self.righty, self.rightx, 2)

        self.sanity_check()
        if(self.valid_new):
            self.find_avg()
            self.find_curvature(mode='avg')

    def add_points(self, lefty, leftx, righty, rightx):
        self.leftx = leftx
        self.lefty = lefty
        self.rightx = rightx
        self.righty = righty

        self.fit_poly()
        
    
            
