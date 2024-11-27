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
result1 = cv2.cvtColor(result1,cv2.COLOR_BGR2GRAY)
#result1= cv2.subtract(result1,matrix_add)
result1 = cv2.cvtColor(result1,cv2.COLOR_GRAY2BGR)
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

mask = cv2.inRange(logo, (200, 200, 200), (255, 255, 255))
mask_inv = cv2.bitwise_not(mask)
matrix =  np.ones(pjatk.shape,dtype="uint8")*100
pjatk = cv2.add(pjatk,matrix)
logo_no_white = cv2.bitwise_and(logo, logo, mask=mask_inv)

pjatk_bg = cv2.bitwise_and(pjatk, logo, mask=mask)
pjatkResult= cv2.add(pjatk_bg, logo_no_white)

#zadanieDomowe
film= cv2.VideoCapture('video.mp4')

frame_width = int(film.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(film.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(film.get(cv2.CAP_PROP_FPS))
result_video = cv2.VideoWriter("wynik_Film.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))


while film.isOpened():
    ret, frame = film.read()
    if not ret:
        break
    mask = cv2.inRange(frame, (1, 0, 0), (255, 120, 120))
    mask_inv = cv2.bitwise_not(mask)
    frame= cv2.bitwise_and(frame, frame, mask=mask_inv)
    result_video.write(frame)



##show
#cv2.imshow('image',image)
cv2.imshow('image',imageTest)
cv2.imshow('result1',result1)
cv2.imshow('result2_before',result2)
cv2.imshow('result2_after',edges_img_smooth)
cv2.imshow('logo',logo)
#cv2.imshow('pjatk',pjatk)
cv2.imshow('pjatk_final',pjatkResult)
# cv2.imshow('video',film)



#cv2.imshow('img_rgb',img_rgb)
#cv2.imshow('img_rgb_brighter',img_rgb_brighter)
#cv2.imshow('img_rgb_darker',img_rgb_darker)
#cv2.imshow('img_rgb_darker2',img_rgb_darker2)
#cv2.imshow('img_rgb_brighter2',img_rgb_brighter2)
cv2.waitKey(0)
cv2.destroyAllWindows()