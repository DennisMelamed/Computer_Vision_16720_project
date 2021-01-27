import numpy as np
import cv2


def createGaussianPyramid(im, sigma0=1, k=np.sqrt(2), levels=[-1,0,1,2,3,4]):
    if len(im.shape)==3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.max()>10:
        im = np.float32(im)/255
    im_pyramid = []
    for i in levels:
        sigma_ = sigma0*k**i
        im_pyramid.append(cv2.GaussianBlur(im, (0,0), sigma_))
    im_pyramid = np.stack(im_pyramid, axis=-1)
    return im_pyramid


def displayPyramid(im_pyramid):
    im_pyramid = np.split(im_pyramid, im_pyramid.shape[2], axis=2)
    im_pyramid = np.concatenate(im_pyramid, axis=1)
    im_pyramid = cv2.normalize(im_pyramid, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imshow('Pyramid of image', im_pyramid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def createDoGPyramid(gaussian_pyramid, levels=[-1,0,1,2,3,4]):
    '''
    Produces DoG Pyramid
    INPUTS
        gaussian_pyramid - A matrix of grayscale images of size
                            [imH, imW, len(levels)]
        levels           - the levels of the pyramid where the blur at each level is
                            outputs

    OUTPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
                   created by differencing the Gaussian Pyramid input
        DoG_levels  - all but the very first item from the levels vector
    '''

    DoG_pyramid = []
    DoG_pyramid = np.diff(gaussian_pyramid, axis=2)

    DoG_levels = levels[1:]
    return DoG_pyramid, DoG_levels


def computePrincipalCurvature(dog_pyramid):
    '''
    takes in dogpyramid generated in createdogpyramid and returns
    principalcurvature,a matrix of the same size where each point contains the
    curvature ratio r for the corre-sponding point in the dog pyramid

    inputs
        dog_pyramid - size (imh, imw, len(levels) - 1) matrix of the dog pyramid

    outputs
        principal_curvature - size (imh, imw, len(levels) - 1) matrix where each
                          point contains the curvature ratio r for the
                          corresponding point in the dog pyramid

    '''
    principal_curvature = []

    for i in range(dog_pyramid.shape[2]):
        dx = cv2.Sobel(dog_pyramid[:,:,i], cv2.CV_32F, 1,0, borderType=cv2.BORDER_REPLICATE)
        dxx = cv2.Sobel(dx, cv2.CV_32F, 1,0, borderType=cv2.BORDER_REPLICATE)
        dy = cv2.Sobel(dog_pyramid[:,:,i], cv2.CV_32F, 0,1, borderType=cv2.BORDER_REPLICATE)
        dyy = cv2.Sobel(dy, cv2.CV_32F, 0,1, borderType=cv2.BORDER_REPLICATE)
        dxy = cv2.Sobel(dx, cv2.CV_32F, 0,1, borderType=cv2.BORDER_REPLICATE)

        tr = (dxx+dyy)**2
        det = (dxx*dyy) - (dxy*dxy)
        r_computed = tr/(det + 1e-10)
        principal_curvature.append(r_computed)


    principal_curvature = np.array(principal_curvature)
    principal_curvature = principal_curvature.transpose(1,2,0)


    return principal_curvature

def getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature,
        th_contrast=0.03, th_r=12):
    '''
    Returns local extrema points in both scale and space using the DoGPyramid

    INPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
        DoG_levels  - The levels of the pyramid where the blur at each level is
                      outputs
        principal_curvature - size (imH, imW, len(levels) - 1) matrix contains the
                      curvature ratio R
        th_contrast - remove any point that is a local extremum but does not have a
                      DoG response magnitude above this threshold
        th_r        - remove any edge-like points that have too large a principal
                      curvature ratio
     OUTPUTS
        locsDoG - N x 3 matrix where the DoG pyramid achieves a local extrema in both
               scale and space, and also satisfies the two thresholds.
    '''
    locsDoG = []

    def extrema(x,y,l):
        x_start = x-1
        x_end = x+2
        y_start = y-1
        y_end = y+2
        l_start = l-1
        l_end = l+2

        if x-1 < 0:
            x_start = 0
        if x+1 > DoG_pyramid.shape[0]:
            x_end = DoG_pyramid.shape[0]
        if y-1 < 0:
            y_start = 0
        if y+1 > DoG_pyramid.shape[1]:
            y_end = DoG_pyramid.shape[1]
        if l-1 < 0:
            l_start = 0
        if l+1 > DoG_pyramid.shape[2]:
            l_end = DoG_pyramid.shape[2]
        if DoG_pyramid[x,y,l] == np.max(DoG_pyramid[x_start:x_end, y_start:y_end, l_start:l_end]) or \
           DoG_pyramid[x,y,l] == np.min(DoG_pyramid[x_start:x_end, y_start:y_end, l_start:l_end]):
            return True
        else:
            return False

    principal_curvature_thr = np.abs(principal_curvature) < th_r
    dog_thc = np.abs(DoG_pyramid) > th_contrast
    okay_points = np.logical_and(principal_curvature_thr, dog_thc)

    for i in range(0, DoG_pyramid.shape[0]):
        for j in range(0, DoG_pyramid.shape[1]):
            for k in range(0,DoG_pyramid.shape[2]):
                if okay_points[i,j,k] and extrema(i,j,k):
                    locsDoG.append([j,i,k])

    return locsDoG


def DoGdetector(im, sigma0=1, k=np.sqrt(2), levels=[-1,0,1,2,3,4],
                th_contrast=0.03, th_r=12):
    '''
    Putting it all together

    INPUTS          Description
    --------------------------------------------------------------------------
    im              Grayscale image with range [0,1].

    sigma0          Scale of the 0th image pyramid.

    k               Pyramid Factor.  Suggest sqrt(2).

    levels          Levels of pyramid to construct. Suggest -1:4.

    th_contrast     DoG contrast threshold.  Suggest 0.03.

    th_r            Principal Ratio threshold.  Suggest 12.


    OUTPUTS         Description
    --------------------------------------------------------------------------

    locsDoG         N x 3 matrix where the DoG pyramid achieves a local extrema
                    in both scale and space, and satisfies the two thresholds.

    gauss_pyramid   A matrix of grayscale images of size (imH,imW,len(levels))
    '''

    ##########################
    # TO DO ....
    # compupte gauss_pyramid, locsDoG here
    gauss_pyramid = createGaussianPyramid(im, sigma0, k, levels)
    DoG_pyramid, DoG_levels = createDoGPyramid(gauss_pyramid, levels)
    #print(DoG_levels)
    #displayPyramid(DoG_pyramid)
    #print(DoG_pyramid[:,:,0])
    principal_curvature = computePrincipalCurvature(DoG_pyramid)
    #displayPyramid(principal_curvature)
    locsDoG = getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature, th_contrast, th_r)

    return locsDoG, gauss_pyramid


if __name__ == '__main__':
    # test gaussian pyramid
    levels = [-1,0,1,2,3,4]
    im = cv2.imread('../data/model_chickenbroth.jpg') #incline_L.png')

    # test DoG detector
    locsDoG, gaussian_pyramid = DoGdetector(im)


    x_pt = [b[0] for b in locsDoG]
    y_pt = [b[1] for b in locsDoG]
