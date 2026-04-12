import numpy as np
import math
def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # Write code here
    array = np.asarray(x,dtype = float)
    array = 1/(1+math.e**(-array))
    return array