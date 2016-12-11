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

	if bubble.cont >= -1 & bubble.cont <= 0:
		x_offset = x_offset + 1
		y_offset = y_offset + 1
		bubble.cont = bubble.cont + 1
	elif bubble.cont <= 1 & bubble.cont >0:
		x_offset = x_offset - 1
		y_offset = y_offset - 1
		bubble.cont = bubble.cont - 1

	for c in range(0,2):
		frame[y_offset:y_offset+bubble.shape[0], x_offset:x_offset+bubble.shape[1], c] = bubble.img[:,:,c] * (bubble.img[:,:,c]/255.0) +  frame[y_offset:y_offset+bubble.shape[0], x_offset:x_offset+bubble.shape[1], c] * (1.0 - bubble.img[:,:,c]/255.0)

def isPosEnabled(x, y, w, h):
	offset = 5
	# if (x+w >= frame.shape[1]-offset) | (y+h >= frame.shape[0]-offset):
	# 	return False

	# for b in bubbles:
	# 	if ((x+w >= b.x-offset) & x <= b.x+b.shape[1]+offset) & ((y+h >= b.y-offset) & y <= b.y+b.shape[0]+offset):
	# 		return False

	return True
	

cap = cv2.VideoCapture(0)
bubble = cv2.imread('res/bubble.jpg')
setWhiteToBlack(bubble)

bubbles = []
flag = 0
while(True):
	#capturar frame por frame
	ret, frame, = cap.read()
	frame = cv2.resize(frame, (0,0), fx=2, fy=2)
	print "bubble ", bubble.shape
	print "frame ", frame.shape

	for b in bubbles: #se dibujan las burbujas que han sido creadas
		drawBubble(b)

	if cv2.waitKey(33) == ord('q'):
		break
	elif cv2.waitKey(33) == ord('a'): #cada vez que se crea presiona la tecla 'a' se crea una burbuja
		enabled = False
		w = bubble.shape[1]
		h = bubble.shape[0]

		# se crean posiciones aleatorias para ubicar las burbujas en el frame
		while enabled == False:
			x = random.randint(5, frame.shape[1]-w-5)
			y = random.randint(5+h, frame.shape[0]-h-5)
			enabled = isPosEnabled(x,y,w,h)

		bubbles.append(Bubble(bubble, x, y))
		

	#mostrar el frame resultante
	cv2.namedWindow("Frame", cv2.WND_PROP_FULLSCREEN)          
	cv2.setWindowProperty("Frame",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
	cv2.imshow('Frame', frame)


cap.release()
cv2.destroyAllWindows() 