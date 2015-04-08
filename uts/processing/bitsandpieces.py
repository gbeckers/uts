def smoothmat(mat, sd=(3,3), mode='same'):

    '''smoothmat -- smooth matrix by convolution with a 2-dimensional
    Gaussian kernel. The size of the Gaussian kernel is truncated at 10 times
    the standard deviation or the size of the input matrix, whichever is
    smaller.

    Inputs:
      mat -- matrix to be smoothed
      sd -- tuple with standard deviation (in points) of Gaussian in two
            dimensions: (rows, columns). A value of zero indicates no
            smoothing.
      mode -- a flag indicating the size of the output
              'valid'  (0): The output consists only of those elements that
                              are computed by scaling the larger array with all
                              the values of the smaller array.
              'same'   (1): The output is the same size as the largest input
                              centered with respect to the 'full' output.
              'full'   (2): The output is the full discrete linear convolution

    Outputs:  (out,)

      out -- a 2-dimensional array containing a smoothed version of the input.
             matrix
    '''

    sdy,sdx = sd

    if sdy > 0:
        my = int(np.round(sdy * 10))
        my = min(my, mat.shape[0])
        kerny = scipy.signal.gaussian(my,sdy).reshape(my,1)
    else:
        kerny = np.ones([1,1])

    if sdx > 0:
        mx = int(np.round(sdx * 10))
        mx = min(mx, mat.shape[1])
        kernx = scipy.signal.gaussian(mx,sdx).reshape(1,mx)
    else:
        kernx = np.ones([1,1])

    kern = kerny * kernx

    factor = float(kern.mean() * kern.size)
    return scipy.signal.convolve(mat, kern, mode=mode) / factor