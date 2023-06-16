# call this function for every image of the training set. 
import numpy as np
from scipy.fftpack import dct, idct

def method1_DCT(im):
    """
    - "Discrte cosine transform" (DCT) method_1 function.
    - Apply the discrte cosine transform algo to the input image. 
    - (data points in terms of a sum of cosine functions oscillating at different frequencies)
    Params: 
    * img: input image. 
    """
    import numpy as np
    image = np.zeros_like(im, dtype=np.uint8)
    for i in range(3):
        # apply DCT to every dimension of the image
        DCT = dct(dct(im[:, :, i], axis=0, norm='ortho'), axis=1, norm='ortho')
        d1 = DCT.copy() # a copy for unmodified pixel in position 1. 
        # set some pixel to 0
        # changing the probability vector you can change the selection
        zero_indices = np.random.choice([False, True], size=DCT.shape, p=[0.4, 0.6]) # 0.6 probability of TRUE and 0.4 probability of FALSE. 
        #----------------------------------------------------------------------------
        # set all the pixels that have a value higher than (mode + std*2) or lower than (mode - std*2) to a value of zero
        values, counts = np.unique(im, return_counts=True)
        mode_value = values[np.argmax(counts)]
        std_dev = np.std(im)
        condition1 = np.abs(DCT) < (mode_value - std_dev*2)
        condition2 = np.abs(DCT) > (mode_value + std_dev*2)
        zero_indices = np.logical_or(condition1, condition2) # all pixels lower than threshold are setted to zero. 
        DCT[zero_indices] = 0
        #----------------------------------------------------------------------------        
        # leave unmodified pixel in position (1,1)
        DCT[0, 0] = d1[0, 0]
        # apply inverse DCT
        image[:, :, i] = idct(idct(DCT, axis=0, norm='ortho'), axis=1, norm='ortho')
    # return the image
    return image.astype(np.uint8)
