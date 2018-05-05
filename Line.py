class Line():
    """This class tracks """
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        #dimensions of the images
        self.dim = None
        #curvature of left and right lanes
        self.right_curv = None
        self.left_curv = None
        #right and left polynomials of the most recent fit
        self.right_poly = [np.array([False])]  
        self.left_poly = [np.array([False])]  
        #average curvature of left and right lanes
        self.avg_right_curv = None
        self.avg_left_curv = None
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #polynomial coefficients averaged over the last n iterations
        self.avg_right_poly = [np.array([False])]  
        self.avg_right_poly = [np.array([False])]
        #conversion parameters from pixel to actua dimesions
        self.ym_per_pix = None
        self.xm_per_pix = None

        def find_curvature(self, mode='recent'):
            #accepts polynomial fit and pixel to actual dimension conversion parameters and returns curveture on the polynomial in actual dimensions 

            #defining evaluation point for y
            y_eval = self.image_dim[0]

            if (mode=='recent'):
                # Calculate the new radii of curvature
                self.right_curv = ((1 + (2*self.right_poly [0]*y_eval*self.ym_per_pix + self.right_poly [1])**2)**1.5) / np.absolute(2*self.right_poly [0])


