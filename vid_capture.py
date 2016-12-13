import numpy as np
import cv2
import random
import time

class Bubble:

	def __init__(self, img, x, y):
		self.img = img
		self.shape = img.shape
		self.x = x
		self.y = y
		self.cont = 0

class Player:

	def __init__(self, color):
		self.color = color
		self.nbubbles = 0

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
			return True, b, i
		
		i = i+1

	return False, 0, 0

#remover del frameflip un bubble
def deleteBubble(b,index):
	h,w,_ = b.shape
	frameflip[b.y:h, b.x:w] = frameflipcpy[b.y:h, b.x:w]
	bubbles.pop(index)

#retorna la posicion del objeto color verde   
def detectGreenColor():
	return getColorPos(frameflip.copy(), lower_green, upper_green)

#retorna la posicion del objeto color verde   
def detectBlueColor():
	return getColorPos(frameflip.copy(), lower_blue, upper_blue)

#retorna la posicion del objeto color rojo   
def detectRedColor():
	#return getColorPos(frameflip.copy(), lower_red, upper_red)
	hsv = cv2.cvtColor(frameflip.copy(), cv2.COLOR_BGR2HSV)  # Convertimos imagen a HSV
   	
	red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
	red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
	red_mask = cv2.bitwise_or(red_mask1,red_mask2)
	red_mask = cv2.erode(red_mask, None, iterations=11)
	red_mask = cv2.dilate(red_mask, None, iterations=11)

	M_red = cv2.moments(red_mask)
	cx = 0
	cy = 0
	if M_red['m00'] > 50000:
		cx = int(M_red['m10'] / M_red['m00'])
		cy = int(M_red['m01'] / M_red['m00'])
		# Mostramos un circulo rojo en la posicion en la que se encuentra el objeto
		cv2.circle(frameflip, (cx, cy), 20, (0, 0, 255), 2)
		return True, cx, cy

	return False, 0, 0
	
#detecta un objeto de un determinado color, dibuja un circulo sobre el y retorna la posicion
def getColorPos(framecp, lowerColor, upperColor):
	hsv = cv2.cvtColor(framecp, cv2.COLOR_BGR2HSV)  # Convertimos imagen a HSV
   	
    # Aqui mostramos la imagen en blanco o negro segun el rango de colores.
	mask = cv2.inRange(hsv, lowerColor, upperColor)
    # Limpiamos la imagen de imperfecciones con los filtros erode y dilate
	mask = cv2.erode(mask, None, iterations=6)
	mask = cv2.dilate(mask, None, iterations=6)

    # Localizamos la posicion del objeto
	M = cv2.moments(mask)
	if M['m00'] > 50000:
		cx = int(M['m10'] / M['m00'])
		cy = int(M['m01'] / M['m00'])
		# Mostramos un circulo azul en la posicion en la que se encuentra el objeto
		cv2.circle(frameflip, (cx, cy), 20, (0, 255, 0), 2)
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

lower_green = np.array([49,100,54])
upper_green = np.array([90,255,183])
lower_blue = np.array([110,100,70])
upper_blue = np.array([130,255,255])
lower_red1 = np.array([0,0,50])
upper_red1 = np.array([8,255,255])
lower_red2 = np.array([175,70,50])
upper_red2 = np.array([180,255,255])

cap = cv2.VideoCapture(0)
bubble = cv2.imread('res/bubble.jpg')
setWhiteToBlack(bubble)

pg = Player('verde')
pr = Player('rojo')
bubbles = []
start = True
start_t = time.time()
total_t = 0
while total_t <=90 :
	#capturar frame por frame
	ret, frame, = cap.read()
	frame = cv2.resize(frame, (0,0), fx=2, fy=2)
	frameflip = cv2.flip(frame.copy(),1)
	frameflipcpy = frameflip.copy()

	if start:
		addBubbles()
		start = False
	
	detectedg, cxg, cyg = detectGreenColor()
	detectedr, cxr, cyr = detectRedColor()
	#detectedr, cxr, cyr = detectBlueColor()

	if detectedg:
		isOn,b,i = isOnBubble(cxg, cyg)
		if isOn:
			deleteBubble(b, i)
			pg.nbubbles = pg.nbubbles + 1
			if len(bubbles)<=5:
				addBubbles()
	if detectedr:
		isOn,b,i = isOnBubble(cxr, cyr)
		if isOn:
			deleteBubble(b, i)
			pr.nbubbles = pr.nbubbles + 1
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
	total_t = time.time()-start_t 

print "Burbujas reventadas Jugador1: "+ str(pg.nbubbles)
print "Burbujas reventadas Jugador2: "+ str(pr.nbubbles)
print "Tiempo transcurrido: " + str(int(total_t)) + " seg"

cap.release()
cv2.destroyAllWindows() 