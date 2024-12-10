import cv2
import numpy as np
print(cv2.__version__)



#zadanie3

img = cv2.imread('test.png')

img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)




##show
img = cv2.resize(img,(28,28))
# cv2.imshow('Sandaly.png',img)
cv2.imwrite(filename='Sandaly.png',img=img)
cv2.waitKey(0)
cv2.destroyAllWindows()