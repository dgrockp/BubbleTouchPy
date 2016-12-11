import numpy as np
import cv2

def detect_color():
    # Creamos una variable de camara y asigamos la primera camara disponible con "0"
    cap = cv2.VideoCapture(0)

    # Iniciamos el bucle de captura, en el que leemos cada frame de la captura
    while (True):
        ret, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Convertimos imagen a HSV

        #Definimos rango minimo y maximo del color Azul en [H,S,V] 
        lower_blue = np.array([103,100,100])
        upper_blue = np.array([130,255,255])

        # Aqui mostramos la imagen en blanco o negro segun el rango de colores.
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        # Limpiamos la imagen de imperfecciones con los filtros erode y dilate
        blue_mask = cv2.erode(blue_mask, None, iterations=4)
        blue_mask = cv2.dilate(blue_mask, None, iterations=4)

        lower_red = np.array([160,100,100])
        upper_red = np.array([190,255,255])

        red_mask = cv2.inRange(hsv, lower_red, upper_red)
        red_mask = cv2.erode(red_mask, None, iterations=2)
        red_mask = cv2.dilate(red_mask, None, iterations=2)

        
        # Localizamos la posicion del objeto
        M_blue = cv2.moments(blue_mask)
        cx = 0
        cy = 0
        if M_blue['m00'] > 50000:
            cx = int(M_blue['m10'] / M_blue['m00'])
            cy = int(M_blue['m01'] / M_blue['m00'])
            # Mostramos un circulo azul en la posicion en la que se encuentra el objeto
            cv2.circle(frame, (cx, cy), 20, (255, 0, 0), 2)

        M_red = cv2.moments(red_mask)
        cx = 0
        cy = 0
        if M_red['m00'] > 50000:
            cx = int(M_red['m10'] / M_red['m00'])
            cy = int(M_red['m01'] / M_red['m00'])
            # Mostramos un circulo rojo en la posicion en la que se encuentra el objeto
            cv2.circle(frame, (cx, cy), 20, (0, 0, 255), 2)



        # Creamos las ventanas de salida y configuracion
        cv2.imshow('Salida', frame)
        cv2.imshow('inRangeBlue', blue_mask)
        cv2.imshow('inRangeRed', red_mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Indicamos que al pulsar "q" el programa se cierre
            break

    cap.release()
    cv2.destroyAllWindows()

detect_color()