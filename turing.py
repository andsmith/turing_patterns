import numpy as np
from scipy.signal import convolve
from scipy.ndimage import laplace
import cv2
from scipy.stats import multivariate_normal
import pylab as plt


class TuringPatternGen(object):
    def __init__(self, 
                 shape=(512,1024), 
                 da = .5,
                 rad_e = 2.,
                 rad_i = 2.5,  # should probably be bigger than rad_e
                 delta_t = .05, 
                 disp_speed = 2):

        self._shape = shape
        self._rad_e, self._rad_i = rad_e, rad_i
        self._da = da 
        self._dt  = delta_t
        self._speed = disp_speed
        self._t = 0

        self._a = np.random.rand(np.prod(shape)).reshape(shape)
        
        self._e_kern = make_gaussian_envelope(self._rad_e)
        self._i_kern = make_gaussian_envelope(self._rad_i)

        print("Excitatory field has shape:  %s" % (self._e_kern.shape, ))
        print("Inhibitory field has shape:  %s" % (self._i_kern.shape, ))

    def _step(self):
        ae_conv = cv2.filter2D(self._a,1, self._e_kern)
        ai_conv = cv2.filter2D(self._a,1, self._i_kern)
        al = laplace(self._a, mode='wrap')        
        da = ae_conv - ai_conv + self._da * al
        self._a += self._dt * da
        clip_ui(self._a)


    def run(self, save_prefix=None):
        frame_num = 0
        while True:

            if self._t % self._speed == 0:
                self._display()
                if save_prefix is not None:
                    name = "%s%.8i.png" %  (save_prefix, frame_num)
                    cv2.imwrite(name, self.render())
                    print("Wrote image: %s" % (name, ))
                    frame_num += 1
            else:
                print("skip")
            self._step()
            self._t += 1

    def _display(self):
        print("display iteration %s"  % (self._t, ))
        a_img = np.uint8(self._a*255)
        pic = self.render()
        cv2.imshow("A concentration", a_img)
        cv2.imshow("pic", pic)
        
        cv2.waitKey(1)

    def render(self):
        pic = np.uint8(self._a>0.5) * 254
        return pic

            
def make_gaussian_envelope(scale, tol=1e-5, max_size=50):
    """
    """
    x, y = np.meshgrid(np.arange(-max_size,max_size+1,dtype=np.float64), 
                       np.arange(-max_size,max_size+1,dtype=np.float64))
    x /= scale
    y /= scale
    n = multivariate_normal(mean=[0,0], cov=None)
    points = np.dstack((x, y))
    e = n.pdf(points)
    e = e/np.sum(e)
    largest_rows = np.max(e, axis=0).reshape(-1)
    first_sig = np.where(largest_rows >= tol)[0][0]
    e = e[first_sig:e.shape[0]-first_sig, 
          first_sig:e.shape[1]-first_sig]
    e = e/np.sum(e)
    return e

def clip_ui(x):
    """
    Change matrix elements so values are clipped to unit interval.
    """
    x[x<0.] = 0.0
    x[x>1.] = 1.0
    
if __name__=="__main__":

    t = TuringPatternGen()

    t.run(save_prefix="test1_")
