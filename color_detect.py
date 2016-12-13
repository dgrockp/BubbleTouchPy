import numpy as np
import cv2

def detect_color():
    # Creamos una variable de camara y asigamos la primera camara disponible con "0"
    cap = cv2.VideoCapture(0)

    # Iniciamos el bucle de captura, en el que leemos cada frame de la captura
    while (True):
        ret, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Convertimos imagen a HSV

        #Definimos rango minimo y maximo del color Verde en [H,S,V] 
        lower_green = np.array([110,100,70])
        upper_green = np.array([130,255,255])

        # Aqui mostramos la imagen en blanco o negro segun el rango de colores.
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        # Limpiamos la imagen de imperfecciones con los filtros erode y dilate
        green_mask = cv2.erode(green_mask, None, iterations=6)
        green_mask = cv2.dilate(green_mask, None, iterations=6)

        lower_red1 = np.array([0,70,50])
        upper_red1 = np.array([8,255,255])
        lower_red2 = np.array([170,70,50])
        upper_red2 = np.array([180,255,255])

        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1,red_mask2)
        red_mask = cv2.erode(red_mask, None, iterations=10)
        red_mask = cv2.dilate(red_mask, None, iterations=10)

        
        # Localizamos la posicion del objeto
        M_green = cv2.moments(green_mask)
        cx = 0
        cy = 0
        if M_green['m00'] > 50000:
            cx = int(M_green['m10'] / M_green['m00'])
            cy = int(M_green['m01'] / M_green['m00'])
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
        cv2.imshow('inRangeBlue', green_mask)
        cv2.imshow('inRangeRed', red_mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Indicamos que al pulsar "q" el programa se cierre
            break

    cap.release()
    cv2.destroyAllWindows()

detect_color()