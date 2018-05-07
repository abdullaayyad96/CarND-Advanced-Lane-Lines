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
        #number of frames since last valid reading 
        self.last_valid_frame = 0
        #dimensions of the images
        self.dim = None
        #conversion parameters from pixel to actual dimesions
        self.ym_per_pix = None
        self.xm_per_pix = None
        #vector to convert polynomial from pixel to actual dimensions
        self.cvrt_mtx = None
        #curvature of recent left and right lanes in pixel dimensions
        self.right_curv = None
        self.left_curv = None
        #right and left polynomials of the most recent fit in pixel dimensions
        self.right_poly = [np.array([False])]  
        self.left_poly = [np.array([False])]  
        #average curvature of left and right lanes in pixel dimensions
        self.avg_right_curv = None
        self.avg_left_curv = None
        #average curvature of left and right lanes in actual dimensions
        self.act_avg_right_curv = None
        self.act_avg_left_curv = None
        #radius of curvature of the line in  actual dimensions
        self.radius_of_curvature = None 
        #polynomial coefficients averaged in pixel dimensions
        self.avg_right_poly = [np.array([False])]  
        self.avg_left_poly = [np.array([False])]
        #polynomial coefficients averaged in actual dimensions
        self.act_avg_right_poly = [np.array([False])]  
        self.act_avg_left_poly = [np.array([False])]
        
    
    def set_param(self, image_shape, ym_per_pix, xm_per_pix):
        self.dim = image_shape
        self.ym_per_pix = ym_per_pix
        self.xm_per_pix = xm_per_pix

        self.cvrt_mtx = np.diag([ (self.xm_per_pix / self.ym_per_pix**2),  (self.xm_per_pix / self.ym_per_pix), self.xm_per_pix  ])

    def find_curvature(self, mode='recent'):
        #accepts polynomial fit and pixel to actual dimension conversion parameters and returns curveture on the polynomial in actual dimensions 

        #defining evaluation point for y
        y_eval = self.dim[0]

        if (mode=='recent'):
            # Calculate the new radii of curvature of left and right lines in pixel dimensions
            self.right_curv = ((1 + (2*self.right_poly[0]*y_eval + self.right_poly[1])**2)**1.5) / np.absolute(2*self.right_poly[0])
            self.left_curv = ((1 + (2*self.left_poly[0]*y_eval + self.left_poly[1])**2)**1.5) / np.absolute(2*self.left_poly[0])
        elif (mode=='avg'):
            # Calculate the new radii of curvature of left and right lines in pixel dimesions
            self.avg_right_curv = ((1 + (2*self.avg_right_poly[0]*y_eval + self.avg_right_poly[1])**2)**1.5) / np.absolute(2*self.avg_right_poly[0])
            self.avg_left_curv = ((1 + (2*self.avg_left_poly[0]*y_eval + self.avg_left_poly[1])**2)**1.5) / np.absolute(2*self.avg_left_poly[0])
            # Calculate the new radii of curvature of left and right lines in actual dimesions
            self.act_avg_right_curv = ((1 + (2*self.act_avg_right_poly[0]*y_eval*self.ym_per_pix + self.act_avg_right_poly[1])**2)**1.5) / np.absolute(2*self.act_avg_right_poly[0])
            self.act_avg_left_curv = ((1 + (2*self.act_avg_left_poly[0]*y_eval*self.ym_per_pix + self.act_avg_left_poly[1])**2)**1.5) / np.absolute(2*self.act_avg_left_poly[0])
            self.radius_of_curvature = (self.avg_right_curv + self.avg_left_curv) / 2

            #for straight lines radius of curvs would be very high
            if(self.radius_of_curvature > 10000):
                self.radius_of_curvature = np.inf

         
    def sanity_check(self):
            
        self.find_curvature(mode='recent')

        if (self.detected==False):
            self.valid_new = True
        else:
            #comparing left and right curvs
            if ((self.right_curv > 3000) & (self.right_curv > 3000)):
                #straigh line case
                curv_error = 0
            else:
                curv_error = abs((self.right_curv - self.left_curv) / ((self.right_curv + self.left_curv)/2))
        
            #comparing to previous values
            #defining error values
            if(self.avg_right_curv > 3000):
                right_curv_error = 0
            else:
                right_curv_error = abs( (self.avg_right_curv - self.right_curv) / self.avg_right_curv )
            if(self.avg_left_curv > 3000):
                left_curv_error = 0
            else:
                left_curv_error = abs( (self.avg_left_curv - self.left_curv) / self.avg_left_curv )

            #position error
            y_eval = self.dim[0]
            leftx_base = self.left_poly[0]*y_eval**2 + self.left_poly[1]*y_eval + self.left_poly[2]
            rightx_base = self.right_poly[0]*y_eval**2 + self.right_poly[1]*y_eval + self.right_poly[2]
            base_diff_pix = rightx_base - leftx_base
            base_diff_act = self.xm_per_pix * base_diff_pix
            base_error = abs(base_diff_act - 3.7) / 3.7

            #appending new values
            if ((right_curv_error < 0.5) & (left_curv_error < 0.5) & (curv_error < 0.8) & (base_error < 0.1)):
                self.valid_new = True
                self.last_valid_frame=0
            else: 
                self.valid_new = False
                self.last_valid_frame += 1
                if(self.last_valid_frame>=10):
                    self.detected = False
                    self.valid_new = True
                    self.last_valid_frame=0
        
    def cvrt_2_act(self):
        #convert polynomials from pixel to actual dimensions in meter
        self.act_avg_right_poly = np.matmul(self.cvrt_mtx, self.avg_right_poly)
        self.act_avg_left_poly = np.matmul(self.cvrt_mtx, self.avg_left_poly)

    def find_avg(self):
        if (self.detected):
            self.avg_right_poly = np.add(np.multiply(0.8, self.avg_right_poly), np.multiply(0.2, self.right_poly ) )
            self.avg_left_poly = np.add(np.multiply(0.8, self.avg_left_poly), np.multiply(0.2, self.left_poly ) )
        else:
            self.avg_right_poly = self.right_poly
            self.avg_left_poly = self.left_poly
            self.detected = True
        
        self.cvrt_2_act()

    def fit_poly(self):
        # Fit a second order polynomial to each
        self.left_poly = np.polyfit(self.lefty, self.leftx, 2)
        self.right_poly = np.polyfit(self.righty, self.rightx, 2)

        self.sanity_check()
        if(self.valid_new):
            self.find_avg()
            self.find_curvature(mode='avg')

    def add_points(self, lefty, leftx, righty, rightx):
        if((len(lefty)>0) & (len(leftx)>0)): 
            self.leftx = leftx
            self.lefty = lefty
        if((len(righty)>0) & (len(rightx)>0)): 
            self.rightx = rightx
            self.righty = righty

        self.fit_poly()
        
    
            
