import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
import scipy
from keypointDetect import DoGdetector


def integralImage(im):
    iim = np.zeros_like(im)
    iim = np.cumsum(im.astype(np.int32), axis=0)
    iim = np.cumsum(iim.astype(np.int32), axis=1)
    #iim = iim#/np.max(np.abs(iim))
    return iim

def boxFilterDxy(iim, L):
#    iim = np.pad(iim, 2*L, mode='symmetric')
    Gamma1 = [1,L,1,L]
    Gamma2 = [-L,-1,1,L ]
    Gamma3 = [-L,-1,-L,-1 ]
    Gamma4 = [1,L,-L,-1]
    term1 = genericFilter(iim, Gamma1)
    term2 = genericFilter(iim, Gamma2)
    term3 = genericFilter(iim, Gamma3)
    term4 = genericFilter(iim, Gamma4)

    conv = term1 + term3 - term2 - term4
    #conv[conv<0] = 0
    conv = conv[52:-52, 52:-52]
    #conv = conv - conv.min()
    #conv = ( conv /conv.max())*255
    #conv = np.uint8(conv)#/conv.max())

    return conv
def boxFilterDxx(iim, L):
#    iim = np.pad(iim, 2*L, mode='symmetric')
    Gamma1 = [-(3*L-1)/2, (3*L-1)/2, -L+1, L-1]
    Gamma2 = [-(L-1)/2, (L-1)/2, -L+1, L-1]
    term1 = genericFilter(iim, Gamma1)
    term2 = genericFilter(iim, Gamma2)

    conv = term1 - 3*term2 #term1 + term2 - term3 - term4 - term5 - term6 + term7 + term8
    #conv[conv<0] = 0
    conv = conv[52:-52, 52:-52]
    #conv = conv - conv.min()
    #conv = ( conv /conv.max())*255
    #conv = np.uint8(conv)#/conv.max())

    return conv

def boxFilterDyy(iim, L):
#    iim = np.pad(iim, 2*L, mode='symmetric')
    Gamma1 = [-L+1, L-1, -(3*L-1)/2, (3*L-1)/2 ]
    Gamma2 = [-L+1, L-1, -(L-1)/2, (L-1)/2]
    term1 = genericFilter(iim, Gamma1)
    term2 = genericFilter(iim, Gamma2)

    conv = term1 - 3*term2 #term1 + term2 - term3 - term4 - term5 - term6 + term7 + term8
    #conv[conv<0] = 0
    conv = conv[52:-52, 52:-52]
    #conv = conv - conv.min()
    #conv = ( conv /conv.max())*255
    #conv = np.uint8(conv)#/conv.max())

    return conv

#iim should be padded and all that, this will just do the additions
def genericFilter(iim, control):
    control = [int(x) for x in control]
    a,b,c,d = control

    term1 = rollLR(rollUD(iim, -a), -c)
    term2 = rollLR(rollUD(iim, -b-1), -d-1)
    term3 = rollLR(rollUD(iim, -a), -d-1)
    term4 = rollLR(rollUD(iim, -b-1), -c)
    return term1 + term2 - term3 - term4





def boxFilterDy(iim, L):
    l = int(np.round(0.8*L))
    #iim = np.pad(iim, l, mode='constant')#, mode='symmetric')
    term1 = rollLR(rollUD(iim, l), l)
    term2 = rollLR(rollUD(iim, -l-1), 0)
    term3 = rollLR(rollUD(iim, l), 0)
    term4 = rollLR(rollUD(iim, -l-1), l)
    term5 = rollLR(rollUD(iim, l), -1)
    term6 = rollLR(rollUD(iim, -l-1), -l-1)
    term7 = rollLR(rollUD(iim, l), -l-1)
    term8 = rollLR(rollUD(iim, -l-1), -1)
    conv = term1 + term2 - term3 - term4 - term5 - term6 + term7 + term8
#    conv[conv<0] = 0
    conv = conv[52:-52, 52:-52]
    #conv = conv - conv.min()
    #conv = (conv/conv.max())*255
    #conv = np.uint8(conv)#/conv.max())

    return conv

def boxFilterDx(iim, L):
    l = int(np.round(0.8*L))
    term1 = rollLR(rollUD(iim, l), l)
    term2 = rollLR(rollUD(iim, 0), -l-1)
    term3 = rollLR(rollUD(iim, l), -l-1)
    term4 = rollLR(rollUD(iim, 0), l)
    term5 = rollLR(rollUD(iim, -1), l)
    term6 = rollLR(rollUD(iim, -l-1), -l-1)
    term7 = rollLR(rollUD(iim, -1), -l-1)
    term8 = rollLR(rollUD(iim, -l-1), l)
    conv = term1 + term2 - term3 - term4 - term5 - term6 + term7 + term8
 #   conv[conv<0] = 0
    conv = conv[52:-52, 52:-52]
    #print(conv.min())
    #print(iim.shape)
    #conv = conv - conv.min()
    #conv = (conv/conv.max())*255
    #conv = np.uint8(conv)#/conv.max())

    return conv

def boxFilterConv(iim, n):
    iim = np.pad(iim, n, mode='symmetric')
    b = rollUp(iim, n)
    #cv2.imshow("b", b)
    c = rollLeft(iim, n)
    #cv2.imshow("c", c)
    a = rollUp(rollLeft(iim, n), n)
    #cv2.imshow("a", a)
    d = iim
    conv = np.uint8((d + a - b - c) / (n*n))

    return conv[n:-n, n:-n]

def rollLR(matrix, n):
    if n > 0:
        return rollLeft(matrix, abs(n))
    elif n <  0:
        return rollRight(matrix, abs(n))
    else:
        return matrix
def rollUD(matrix, n):
    if n > 0:
        return rollUp(matrix, abs(n))
    elif n < 0:
        return rollDown(matrix, abs(n))
    else:
        return matrix


def rollLeft(matrix, n):
    return np.concatenate((matrix[:, n:], np.zeros((matrix.shape[0], n))), axis=1)

def rollUp(matrix, n):
    return np.concatenate((matrix[n:, :], np.zeros((n, matrix.shape[1]))), axis=0)

def rollRight(matrix, n):
    return np.concatenate(( np.zeros((matrix.shape[0], n)), matrix[:, :-n]), axis=1)

def rollDown(matrix, n):
    return np.concatenate(( np.zeros((n, matrix.shape[1])),matrix[:-n, :]), axis=0)


def feature_detect_DoH(im, iim, o, i, dxx, dyy, dxy):
    im = im[52:-52, 52:-52]
    iim = iim[52:-52, 52:-52]
    #print(im.shape)
    #print(iim.shape)
    step = 2**(o-1)
    L = (2**o)*i + 1
    w = 0.912
    doh = np.zeros_like(iim, dtype=int)
    dxx = dxx.astype(np.int32)
    dyy = dyy.astype(np.int32)
    dxy = dxy.astype(np.int32)
    for x in range(0, im.shape[0], step):
        for y in range(0, im.shape[1], step):
            doh[x,y] = (1/(L**4))*(dxx[x,y]*dyy[x,y] - (w*dxy[x,y])**2)
    #doh[0:step:im.shape[0], 0:step:im.shape[1]] = (1/(L**4))*(dxx[0:step:im.shape[0], 0:step:im.shape[1]]*dyy[0:step:im.shape[0], 0:step:im.shape[1]] - (w*dxy[0:step:im.shape[0], 0:step:im.shape[1]])**2)
    return doh

#def make_overall_pyramids(im_gray, integral_image):
#    dxs = []
#    dys = []
#    doh = []
#    o_s = [1,2,3,4]
#    i_s = [1,2,3,4]
#    for o in o_s:
#        for i in i_s:
#            L = (2**o)*i + 1
#            dx = boxFilterDx(integral_image, L)
#            dy = boxFilterDy(integral_image, L)
#            dxs.append(dx)
#            dys.append(dy)
#            dxx = boxFilterDxx(integral_image, L)
#            dyy = boxFilterDyy(integral_image, L)
#            dxy = boxFilterDxy(integral_image, L)
#            dohL = feature_detect_DoH(im_gray, integral_image, o,i, dxx, dyy, dxy)
#            doh.append(dohL)
#    return dxs,dys, doh



def make_first_deriv_pyramid(im_gray, integral_image):
    dxs = []
    dys = []
    o_s = [1,2,3,4]
    i_s = [1,2,3,4]
    for o in o_s:
        for i in i_s:
            L = (2**o)*i + 1
            dx = boxFilterDx(integral_image, L)
            dy = boxFilterDy(integral_image, L)
            dxs.append(dx)
            dys.append(dy)
    return dxs,dys


def make_pyramid(im_gray, integral_image):
    doh = []
    o_s = [1,2,3,4]
    i_s = [1,2,3,4]
    for o in o_s:
        for i in i_s:
            L = (2**o)*i + 1
    #        print("L: {}".format(L))
            dxx = boxFilterDxx(integral_image, L)
            dyy = boxFilterDyy(integral_image, L)
            dxy = boxFilterDxy(integral_image, L)

            #fig, axs = plt.subplots(3,1)
            #axs[0].imshow(dxx, cmap='gray')
            #axs[1].imshow(dyy, cmap='gray')
            #axs[2].imshow(dxy, cmap='gray')
            #plt.show()

            dohL = feature_detect_DoH(im_gray, integral_image, o,i, dxx, dyy, dxy)
            doh.append(dohL)
    return doh

def select_features(iim, o,i, doh, thresh):
    L = (2**o)*i + 1
    doh_index = (o-1)*4 + (i-1)
   # print("doh index {}".format(doh_index))
    step = 2**(o-1)
    keypoints = []
    iim = iim[52:-52,52:-52]
    for x in range(1, iim.shape[0]-1, step):
        for y in range(1, iim.shape[1]-1, step):
            if doh[doh_index][x,y] > thresh:
                if doh[doh_index-1][x,y] <= doh[doh_index][x,y] and \
                   doh[doh_index+1][x,y] <= doh[doh_index][x,y] and \
                   doh[doh_index][x+1,y] <= doh[doh_index][x,y] and \
                   doh[doh_index][x-1,y] <= doh[doh_index][x,y] and \
                   doh[doh_index][x,y+1] <= doh[doh_index][x,y] and \
                   doh[doh_index][x,y-1] <= doh[doh_index][x,y]:

                    keypoints.append([x,y,L])


    return keypoints






def surfMatch(desc1, desc2, ratio=0.7):
    '''
    performs the descriptor matching
    INPUTS
        desc1, desc2 - m1 x n and m2 x n matrix. m1 and m2 are the number of keypoints in image 1 and 2.
                                n is the number of bits in the brief
    OUTPUTS
        matches - p x 2 matrix. where the first column are indices
                                        into desc1 and the second column are indices into desc2
    '''

    D = cdist(np.float32(desc1), np.float32(desc2))
    #print(D)
    # find smallest distance
    ix2 = np.argmin(D, axis=1)
    d1 = D.min(1)
    # find second smallest distance
    d12 = np.partition(D, 2, axis=1)[:,0:2]
    d2 = d12.max(1)
    r = d1/(d2+1e-10)
    #print(r)
    is_discr = r<ratio
    ix2 = ix2[is_discr]
    ix1 = np.arange(D.shape[0])[is_discr]

    matches = np.stack((ix1,ix2), axis=-1)
    return matches


def plotMatches(im1, im2, matches, locs1, locs2):
    fig = plt.figure()
    # draw two images side by side
    imH = max(im1.shape[0], im2.shape[0])
    im = np.zeros((imH, im1.shape[1]+im2.shape[1]), dtype='uint8')
    im[0:im1.shape[0], 0:im1.shape[1]] = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im[0:im2.shape[0], im1.shape[1]:] = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    plt.imshow(im, cmap='gray')
    for i in range(matches.shape[0]):
        pt1 = locs1[matches[i,0], 0:2]
        pt2 = locs2[matches[i,1], 0:2].copy()
        pt2[1] += im1.shape[1]
        x = np.asarray([pt1[1], pt2[1]])
        y = np.asarray([pt1[0], pt2[0]])
        plt.plot(x,y,'r')
        plt.plot(x,y,'g.')
    plt.show()


def Gkernel(sigma, x, y, k):
    Ck = 0
    for i in range(-k,k+1):
        for j in range(-k,k+1):
            Ck = Ck + (1/(2*np.pi*sigma))*np.exp(-(i**2 + j**2)/(2*sigma**2))
    result = (1/(2*np.pi*sigma*Ck))*np.exp(-(x**2 + y**2)/(2*sigma**2))
    return result


def orientation(image, doh, x,y, L, first_deriv_x, first_deriv_y):
    np.set_printoptions(linewidth=500)
    sigma= int(np.round(0.4*L))
    phi = np.zeros((13,13, 2))
    scaler_array = np.zeros((13,13))
    for i in range(-6,6+1):
        for j in range(-6,6+1):
            if i**2 + j**2 <= 36:
                scaler = Gkernel(1,i/2, j/2, 6)
                scaler_array[i+6, j+6] = scaler
                if x+i*sigma >= image.shape[0] or y+j*sigma >= image.shape[1]:
                    continue
                phi[i+6, j+6, :] = [first_deriv_x[L][x+i*sigma, y+j*sigma]*scaler, first_deriv_y[L][x+i*sigma, y+j*sigma]*scaler]
    Phi = np.zeros((40,2))
    #print(phi[:,:,0])
    #print(phi[:,:,1])
    #print(scaler_array)
    score = np.zeros((40, 1))
    for k in range(0,40):
        theta_k = (k)*np.pi/20
     #   print(theta_k)
        for i in range(-6,6):
            for j in range(-6,6):
                angle = np.arctan2(phi[i+6, j+6][1], phi[i+6, j+6][0]) + np.pi
                if theta_k-(np.pi/6) < angle and angle < theta_k + (np.pi/6):
                    Phi[k, :] = Phi[k, :] + phi[i+6, j+6]
        score[k] = np.linalg.norm(Phi[k,:])
    #print(score)
    #fig, axs = plt.subplots(2,1)
    #axs[0].imshow(first_deriv_x[L][x-6*sigma:x+6*sigma, y-6*sigma:y+6*sigma])
    #axs[1].imshow(first_deriv_y[L][x-6*sigma:x+6*sigma, y-6*sigma:y+6*sigma])
    #plt.show()
    #quit()
    best_theta_index = np.argmax(score)
    best_theta = (best_theta_index)*np.pi/20
    return best_theta


def make_descriptor(image, x,y,L, doh, dxs, dys):
    image = image[52:-52,52:-52]
    sigma= int(np.round(0.4*L))
    best_theta = orientation(image, doh, x,y,L, dxs,  dys)
    #print("best theta: {}".format(best_theta))
    #plt.imshow(image[int(x-6*sigma):int(x+6*sigma), int(y-6*sigma):int(y+6*sigma)])
    start = [int(6*sigma), int(6*sigma - 6*sigma*np.cos(best_theta)) ]
    end = [int(6*sigma), int(6*sigma + 6*sigma*np.sin(best_theta)) ]
    #plt.plot(start, end)
    #plt.colorbar()
    #plt.show()
    R_best_theta = np.array([[np.cos(best_theta), -np.sin(best_theta)], [np.sin(best_theta), np.cos(best_theta)]])
    values = np.zeros((20,20, 2))
    mu = np.zeros((16,4))
    for u in np.arange(-9.5,9.5, 1):
        for v in np.arange(-9.5, 9.5, 1):
            rotated = R_best_theta@np.array([[u],[v]])
            xc = int(np.round( rotated[0] + x  ))
            yc = int(np.round( rotated[1] + y  ))
            if xc >= dxs[L].shape[0] or yc >= dxs[L].shape[1] or xc >= dys[L].shape[0] or yc >= dys[L].shape[1]:
                continue
            g_kernel_scale = Gkernel(1, u/3.3, v/3.3, 20)
            local_deriv_vector = np.array([dxs[L][xc,yc], dys[L][xc,yc]])
            scaled_local_deriv_vector = R_best_theta.T @ local_deriv_vector.T
            scaled_local_deriv_vector = g_kernel_scale * scaled_local_deriv_vector
            values[int(u+9.5), int(v+9.5), :] = scaled_local_deriv_vector
    for i in range(0,4):
        for j in range(0,4):
            idx = i*4 + j
            u_start = i*4
            v_start = j*4
            patch = values[u_start:u_start+4, v_start:v_start+4,:]
            mu[idx,0] = np.sum(patch[:,:,0])
            mu[idx,1] = np.sum(patch[:,:,1])
            mu[idx,2] = np.sum(np.abs(patch[:,:,0]))
            mu[idx,3] = np.sum(np.abs(patch[:,:,1]))
    mu = mu/np.linalg.norm(mu)
    mu = mu.flatten()
    return mu


def improve_point(x0,y0,L0, doh, o):
    p = 2**(o-1)

    Hxx = (1/(p*p))*(  doh[L0][x0+p, y0] + doh[L0][x0-p, y0] - 2*doh[L0][x0,y0]   )
    Hyy = (1/(p*p))*(  doh[L0][x0, y0+p] + doh[L0][x0, y0-p] - 2*doh[L0][x0,y0]   )
    Hxy = (1/(4*p*p))*(  doh[L0][x0+p, y0+p] + doh[L0][x0-p, y0-p] - doh[L0][x0-p,y0+p] - doh[L0][x0+p, y0-p]  )
    HxL = (1/(8*p*p))*(  doh[L0+2*p][x0+p, y0] + doh[L0-2*p][x0-p, y0] - doh[L0+2*p][x0-p,y0] - doh[L0-2*p][x0+p, y0]  )
    HyL = (1/(8*p*p))*(  doh[L0+2*p][x0, y0+p] + doh[L0-2*p][x0, y0-p] - doh[L0+2*p][x0,y0-p] - doh[L0-2*p][x0, y0+p]  )
    HLL = (1/(4*p*p))*(  doh[L0+2*p][x0, y0] + doh[L0-2*p][x0, y0] - 2*doh[L0][x0,y0]  )
    H0 = np.array([ [Hxx, Hxy, HxL], [Hxy, Hyy, HyL], [HxL, HyL, HLL] ])

    dx = (1/(2*p*p))*(  doh[L0][x0+p, y0] - doh[L0][x0-p, y0]    )
    dy = (1/(2*p*p))*(  doh[L0][x0, y0+p] - doh[L0][x0, y0-p]    )
    dL = (1/(4*p*p))*(  doh[L0+2*p][x0, y0] - doh[L0-2*p][x0, y0]    )

    d0 = np.array([[dx],[dy],[dL]  ])
    #print(d0.shape)

    try:
        epsilon = -np.linalg.inv(H0)@d0
    except:
        return None, None, None
    #print(epsilon.shape)
    if np.max(np.abs(epsilon)) < p:
        return [int(x0+epsilon[0, 0]), int(y0+epsilon[1, 0]), int(L0+epsilon[2, 0])]
    else:
        return None, None, None




def compute_features(im):
    from scipy.ndimage import gaussian_filter
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gray_padded = np.pad(im_gray, 52, mode='symmetric')
    integral_image = integralImage(im_gray_padded)


    dxs, dys = make_first_deriv_pyramid(im_gray_padded, integral_image)
    doh = make_pyramid(im_gray_padded, integral_image)
    #dxs, dyx, doh = make_overall_pyramids(im_gray_padded, integral_image)
#    fig, axs = plt.subplots(2,16)
#    for i in range(len(dxs)):
#        axs[0, i].imshow(dxs[i], aspect='auto')
#        axs[1 ,i].imshow(dys[i], aspect='auto')
#    fig2, axs2 = plt.subplots(2,16)
#    o_s = [1,2,3,4]
#    i_s = [1,2,3,4]
#    idx = 0
#    for o in o_s:
#        for i in i_s:
#            L = (2**o)*i + 1
#            sigma = 0.4*L
#            filtered =gaussian_filter(im_gray, sigma, [1,0])
#            axs2[0,idx].imshow(filtered, aspect='auto')
#            filtered =gaussian_filter(im_gray, sigma, [0,1])
#            axs2[1,idx].imshow(filtered, aspect='auto')
#            idx = idx + 1
#    plt.show()



    import time
    t0 = time.time()
    thresh = 5000
    features = []
    for o in [1,2,3,4]:
        for i in [2,3]:
            L = (2**o)*i + 1
      #      print(L)
            feature = select_features(integral_image, o,i, doh, thresh)
            fixed_feature = []
            for ft in feature:
                x,y,L = ft
                x,y,L = improve_point(x,y,L,doh, o)
                if x is not None:
                    fixed_feature.append([x,y,L])
            features.extend(fixed_feature)
    thetas = []
    mus = []
    for feature in features:
        print(feature)
        x,y,L = feature
        theta = orientation(im, doh, x,y, L, dxs, dys)
    #    cv2.line(im, (y,x), (int(y+np.sin(theta)*L*10), int(x+np.cos(theta)*L*10)), (0,255,0), 1)

        #if x >= im_gray.shape[0] or y >= im_gray.shape[1]:
        #    continue
        mu = make_descriptor(im_gray_padded, x,y,L, doh, dxs, dys)
        mus.append(mu)
    #    fig, axs = plt.subplots(2,1)
    #    axs[0].imshow(dxs[L])
    #    axs[1].imshow(dys[L])
    #    plt.show()
    mus = np.array(mus)
    features = np.array(features)
    t1 = time.time()
    print(t1-t0)
    return features, mus









if __name__ == '__main__':

    #test integral image
    im = cv2.imread('../data/model_chickenbroth.jpg')
    features1, mus1 = compute_features(im)
    counts = []

    for theta in range(180,360,10):
        print(theta)
        rot_mat = cv2.getRotationMatrix2D((im.shape[1]//2, im.shape[0]//2), theta, 1)
        rot_im = cv2.warpAffine(im, rot_mat, (im.shape[1], im.shape[0]))

        features2, mus2 = compute_features(rot_im)

#    for feature in features1:
#        x,y,L = feature
#        cv2.circle(im, (y,x), 2, (0,0,255), -1)
#    for feature in features2:
#        x,y,L = feature
#        cv2.circle(rot_im, (y,x), 2, (255,0,0), -1)
    #cv2.imshow("orig", im)
    #cv2.imshow("rot", rot_im)

        matches = surfMatch(mus1, mus2)
        plotMatches(im, rot_im, matches, features1, features2)
        quit()
    #    counts.append(len(matches))
    #    print(len(matches))
    #print(counts)
    #print(features1)
    #print(features2)


    #cv2.waitKey(0)
