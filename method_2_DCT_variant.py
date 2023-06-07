import numpy as np
from scipy.fftpack import dct, idct

def method2dct(im):
    y, x, z = im.shape
    
    # Apply standard deviation for every level
    sigma2 = np.std(im.astype(float))
    sigma1 = np.std(sigma2)
    sigma = np.std(sigma1) / 2
    
    image = np.zeros_like(im, dtype=np.uint8)
    
    for channel in range(z):
        # Apply DCT
        DCTim = dct(dct(im[:, :, channel], axis=0), axis=1)
        
        for riga in range(y):
            for colonna in range(x):
                # DTC(1,1) unmodified
                if riga == 0 and colonna == 0:
                    continue
                else:
                    # Calculate random number between (-0.5, 0.5). Use np.random()
                    random_z = np.random.random()
                    while random_z > 0.5:
                        random_z = np.random.random()
                    
                    prob = np.random.random()
                    if prob > 0.5:
                        random_z *= 1 
                    else:
                        random_z *= -1
                    
                    # Modify DCT image
                    DCTim[riga, colonna] += sigma * random_z
            
        # Inverse DCT
        image[:, :, channel] = idct(idct(DCTim, axis=0), axis=1)
    
    return image.astype(np.uint8)
