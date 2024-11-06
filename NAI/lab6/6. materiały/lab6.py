import cv2
import numpy as np
print(cv2.__version__)
#wczytywanie
imageTest = cv2.imread('img.jpg')
image = cv2.imread('img.jpg')
#cv2.imshow('image',image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#cv2.imwrite(filename='result1.jpg',img=image)
#zmiana jasnosci i kontrastu
img_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
value=88
##dodawanie
matrix_add = np.ones(img_rgb.shape,dtype="uint8")*value
img_rgb_brighter = cv2.add(img_rgb,matrix_add)
img_rgb_darker = cv2.subtract(img_rgb,matrix_add)
##mnożenie
valueMulti=3
#cv2.imshow('img_rgb',img_rgb_brighter)
matrix_mult = np.ones(img_rgb.shape,)*valueMulti
img_rgb_darker2 = np.uint8(cv2.multiply(np.float64(img_rgb),1/matrix_mult))
img_rgb_brighter2 = np.uint8(cv2.multiply(np.float64(img_rgb),matrix_mult))
#rysowanie
color =(255,0,0)
color2 = (255,255,255)
##linia
image2 = cv2.line(image,(0,0),(255,255),color)
##prostokąt
image3= cv2.rectangle(image,(0,0),(255,255),color2)



#zadanie1
result1 = cv2.imread('img.jpg')
result1= cv2.subtract(result1,matrix_add)
result1 = cv2.rectangle(result1,(100,100),(300,300),color)
cv2.imwrite('result1.jpg',result1)
#zadanie2
result2= cv2.imread('noise.jpg')
edges_img = cv2.Canny(result2,100,200)
edges_img_smooth = cv2.medianBlur(edges_img,5)
result2 = cv2.medianBlur(result2,5)
result2 = cv2.Canny(result2,100,200)
#zadanie3
pjatk = cv2.imread('pjatk.jpg')
logo = cv2.imread('logo.jpg')
white = (255,255,255)


matrixLogo = np.ones(logo.shape,dtype="uint8")*100
logo_dark = cv2.cvtColor(logo,cv2.COLOR_BGR2GRAY)

logo_dark = cv2.subtract(logo,matrixLogo)

#mask = cv2.inRange(logo,(100,100,100),(255,255,255))
mask = cv2.inRange(logo_dark,(250,250,250),(255,255,255))
#logo = cv2.bitwise_not(mask)

logo = cv2.subtract(logo,logo)

#logo = cv2.cvtColor(pjatk,cv2.COLOR_BGR2RGB)
matrix =  np.ones(pjatk.shape,dtype="uint8")*100
pjatk = cv2.add(pjatk,matrix)
#pjatkResult = cv2.add(pjatk,logo)
pjatkResult = cv2.bitwise_and(pjatk,pjatk,mask=mask)

##show
#cv2.imshow('image',image)
cv2.imshow('image',imageTest)
cv2.imshow('result1',result1)
cv2.imshow('result2_before',result2)
cv2.imshow('result2_after',edges_img_smooth)
cv2.imshow('logo',logo)
#cv2.imshow('pjatk',pjatk)
cv2.imshow('pjatk_final',pjatkResult)



#cv2.imshow('img_rgb',img_rgb)
#cv2.imshow('img_rgb_brighter',img_rgb_brighter)
#cv2.imshow('img_rgb_darker',img_rgb_darker)
#cv2.imshow('img_rgb_darker2',img_rgb_darker2)
#cv2.imshow('img_rgb_brighter2',img_rgb_brighter2)
cv2.waitKey(0)
cv2.destroyAllWindows()