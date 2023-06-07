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
        DCT = dct(dct(im[:, :, i], axis=0), axis=1)
        d1 = DCT.copy() # a copy for unmodified pixel in opsition 1. 
        # set some pixel to 0
        # changing the probability vector you can change the selection
        zero_indices = np.random.choice([False, True], size=DCT.shape, p=[0.6, 0.4]) # 0.6 probability of TRUE and 0.4 probability of FALSE. 
        DCT[zero_indices] = 0
        # leave unmodified pixel in position (1,1)
        DCT[0, 0] = d1[0, 0]
        # apply inverse DCT
        image[:, :, i] = idct(idct(DCT, axis=0), axis=1)
    # return the image
    return image.astype(np.uint8)