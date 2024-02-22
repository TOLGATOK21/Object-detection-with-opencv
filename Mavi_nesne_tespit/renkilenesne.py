# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 14:49:11 2024

@author: TOLGA
"""
import cv2
import numpy as np 
from collections import deque

# tespit ettiğimiz objenin merkezini depolamak için deque kullanıyoruz

buffer_size = 16
pts = deque(maxlen=buffer_size)

# mavi renk aralığı
blueLower = (84, 98, 0)
blueUpper = (179, 255, 255)

# capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2060)

while True:
    success, imgOriginal = cap.read()
    if success:
        # blur ile detayı azaltıp noise eliminate ediyoruz
        blurred = cv2.GaussianBlur(imgOriginal, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        cv2.imshow("HSV ımage", hsv)
        
        # mavi renk için maske oluştur
        mask = cv2.inRange(hsv, blueLower, blueUpper)
        
        # gürültüleri silmemiz lazım (erozyon-genişleme)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        cv2.imshow("Mask + erozyon ve genişleme", mask)
        
        # KONTURLERİ BULMA İŞLEMİ YAPIYORUZ 
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center = None  # cismin merkezini aldık
        if len(contours) > 0:
            # en büyük konturu al
            c = max(contours, key=cv2.contourArea)
            # dikdörtgene çevir 
            rect = cv2.minAreaRect(c)
            
            ((x, y), (width, height), rotation) = rect
            s = "x: {}, y: {}, width: {}, height: {}, rotation: {}".format(np.round(x), np.round(y), np.round(width), np.round(height), np.round(rotation))
            print(s)
            box = cv2.boxPoints(rect)
            box = np.int64(box)
            
            # moment hazırlıyoruz = bazı görüntü pixel özelliklerini bulmaya yarar
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                
            # Konturu çizdirme işlevi
            cv2.drawContours(imgOriginal, [box], 0, (0, 255, 255), 2)
            # merkeze bi tane nokta ekle renk = pembe
            cv2.circle(imgOriginal, center, 5, (255, 0, 255), -1)
            # bilgileri ekrana yazdırıyoruz
            cv2.putText(imgOriginal, s, (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0))
            
        #noktalarımızı deque içine ekliyoruz
        pts.appendleft(center)
        for i in range (1,len(pts)):
            if pts[i-1] is None or pts[i] is None: continue
            cv2.line(imgOriginal,pts[i-1],pts[i],(0,255,0),5)
        
        cv2.imshow("Orijinal Tespit", imgOriginal)
     
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
