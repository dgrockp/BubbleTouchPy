import numpy as np
import cv2
import random


class Bubble:

	def __init__(self, img, x, y):
		self.img = img
		self.shape = img.shape
		self.x = x
		self.y = y
		self.cont = 0

#convierte el fondo blanco de la imagen a negro
def setWhiteToBlack(bubble):
	for c in range(0,2):
		for y in range(0,bubble.shape[0]):
			for x in range(0,bubble.shape[1]):
				if(bubble[y,x,c]>=250):
					bubble[y,x,c] = 0

#dibuja una burbuja en el frame capturado por la camara
def drawBubble(bubble):

	x_offset = bubble.x
	y_offset = bubble.y

	if bubble.cont >= -1 and bubble.cont <= 0:
		x_offset = x_offset + 1
		y_offset = y_offset + 1
		bubble.cont = bubble.cont + 1
	elif bubble.cont <= 1 and bubble.cont >0:
		x_offset = x_offset - 1
		y_offset = y_offset - 1
		bubble.cont = bubble.cont - 1

	for c in range(0,2):
		frameflip[y_offset:y_offset+bubble.shape[0], x_offset:x_offset+bubble.shape[1], c] = bubble.img[:,:,c] * (bubble.img[:,:,c]/255.0) +  frameflip[y_offset:y_offset+bubble.shape[0], x_offset:x_offset+bubble.shape[1], c] * (1.0 - bubble.img[:,:,c]/255.0)

#Verifica si el obetjo de color que se esta detectando esta sobre algun bubble 
def isOnBubble(cx, cy):
	i=0
	for b in bubbles:
		h,w,_ = b.shape
		if (cx >= b.x and cx<= b.x+w) and (cy >= b.y and cy<= b.y+h):
			print cx,cy,b.x,b.y,w,h
			return True, b, i
		
		i = i+1

	return False, 0, 0

#remover del frameflip un bubble
def deleteBubble(b,index):
	h,w,_ = b.shape
	frameflip[b.y:h, b.x:w] = frameflipcpy[b.y:h, b.x:w]
	bubbles.pop(index)

#detecta un objeto de color verde, dibuja un circulo sobre el y retorna la posicion 
def detectGreenColor():

	hsv = cv2.cvtColor(frameflip, cv2.COLOR_BGR2HSV)  # Convertimos imagen a HSV
   	
   	#Definimos rango minimo y maximo del color Verde en [H,S,V] 
	lower_green = np.array([49,100,54])
	upper_green = np.array([90,255,183])

    # Aqui mostramos la imagen en blanco o negro segun el rango de colores.
	green_mask = cv2.inRange(hsv, lower_green, upper_green)
    # Limpiamos la imagen de imperfecciones con los filtros erode y dilate
	green_mask = cv2.erode(green_mask, None, iterations=4)
	green_mask = cv2.dilate(green_mask, None, iterations=4)

    # Localizamos la posicion del objeto
	M_green = cv2.moments(green_mask)
	if M_green['m00'] > 50000:
		cx = int(M_green['m10'] / M_green['m00'])
		cy = int(M_green['m01'] / M_green['m00'])
		# Mostramos un circulo azul en la posicion en la que se encuentra el objeto
		cv2.circle(frameflip, (cx, cy), 20, (255, 0, 0), 2)
		return True, cx, cy

	return False, 0, 0

def addBubbles():
	n = random.choice(nb)
	for i in range(n):
		# se crean posiciones aleatorias para ubicar las burbujas en el frameflip
		h,w,_ = bubble.shape
		x = random.randint(5, frameflip.shape[1]-w-5)
		y = random.randint(5+h, frameflip.shape[0]-h-5)
		bubbles.append(Bubble(bubble, x, y))


# def isPosEnabled(x, y, w, h):
# 	offset = 5
# 	if (x+w >= frameflip.shape[1]-offset) or (y+h >= frameflip.shape[0]-offset):
# 		return False

# 	for b in bubbles:
# 		if ((x+w >= b.x-offset) and x <= b.x+b.shape[1]+offset) and ((y+h >= b.y-offset) and y <= b.y+b.shape[0]+offset):
# 			return False

# 	return True
	
nb = [1,1,1,1,2,2,2,3,3]

cap = cv2.VideoCapture(0)
bubble = cv2.imread('res/bubble.jpg')
setWhiteToBlack(bubble)

bubbles = []
start = True
while(True):
	#capturar frame por frame
	ret, frame, = cap.read()
	frame = cv2.resize(frame, (0,0), fx=2, fy=2)
	frameflip = cv2.flip(frame.copy(),1)
	frameflipcpy = frameflip.copy()

	if start:
		addBubbles()
		start = False
	
	detected, cx, cy = detectGreenColor()
	if detected:
		isOn,b,i = isOnBubble(cx, cy)
		if isOn:
			deleteBubble(b, i)
			if len(bubbles)<=5:
				addBubbles()

	for b in bubbles: #se dibujan las burbujas que han sido creadas
		drawBubble(b)

	if cv2.waitKey(1) & 0xFF == ord('q'):  # Indicamos que al pulsar "q" el programa se cierre
		break

	#mostrar el frameflip resultante
	cv2.namedWindow("frameflip", cv2.WND_PROP_FULLSCREEN)          
	cv2.setWindowProperty("frameflip",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
	cv2.imshow('frameflip', frameflip)


cap.release()
cv2.destroyAllWindows() 