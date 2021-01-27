import cv2
import BRIEF
import matplotlib.pyplot as plt
im1 = cv2.imread('../data/model_chickenbroth.jpg')
im2 = cv2.imread('../data/model_chickenbroth.jpg')#'../data/chickenbroth_01.jpg')
locs1, desc1 = BRIEF.briefLite(im1)
#print("desc:")
#print(desc1.shape)
thetas = range(0,360,10)
matchn = []
for theta in range(0,360, 10):
    rot_mat = cv2.getRotationMatrix2D((im2.shape[1]//2, im2.shape[0]//2), theta, 1)
    rot_im2 = cv2.warpAffine(im2, rot_mat, (im2.shape[1], im2.shape[0]))
    locs2, desc2 = BRIEF.briefLite(rot_im2)
    matches = BRIEF.briefMatch(desc1, desc2)
    matchn.append(matches.shape[0])
plt.bar(thetas, matchn)
plt.show()

