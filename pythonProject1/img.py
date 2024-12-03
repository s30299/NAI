import cv2
import numpy as np
print(cv2.__version__)



#zadanie1
result1 = cv2.imread('img/shirt.png')
if result1 is None:
    print("Error: Could not load the image. Check the file path.")
else:
    # Konwersja do odcieni szarości
    gray_image = cv2.cvtColor(result1, cv2.COLOR_BGR2GRAY)
    print("Image converted to grayscale.")
    # Wyświetlenie obrazu (opcjonalnie)
    cv2.imshow("Grayscale Image", gray_image)
result1 = cv2.cvtColor(result1, cv2.COLOR_BGR2GRAY)

cv2.imwrite('result1.png',result1)


##show
#cv2.imshow('image',image)
# cv2.imshow('image',imageTest)
cv2.imshow('result1',result1)


cv2.waitKey(0)
cv2.destroyAllWindows()