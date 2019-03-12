import cv2

images = ["cameraman.jpg", "crowd2.jpg", "lenna.jpg", "tutu.jpg"]

# copiando as originais
for i in images:
	img = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
	cv2.imwrite("T01/" + i.replace(".jpg", "-raw.jpg"), img)

# redimensao
for i in images:
	img = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
	mod = cv2.resize(img, (128, 128), interpolation=cv2.INTER_LINEAR)
	mod = cv2.resize(mod, (512, 512), interpolation=cv2.INTER_LINEAR)
	cv2.imwrite("T01/" + i.replace(".jpg", "-redim.jpg"), mod)

# binarizacao
for i in images:
	img = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
	ret, mod = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
	cv2.imwrite("T01/" + i.replace(".jpg", "-binary.jpg"), mod)

# redimensao + binarizacao
for i in images:
	img = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
	mod = cv2.resize(img, (128, 128), interpolation=cv2.INTER_LINEAR)
	ret, mod = cv2.threshold(mod, 127, 255, cv2.THRESH_BINARY)
	mod = cv2.resize(mod, (512, 512), interpolation=cv2.INTER_LINEAR)
	cv2.imwrite("T01/" + i.replace(".jpg", "-redim-binary.jpg"), mod)

