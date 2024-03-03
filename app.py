from imutils.video import VideoStream
import numpy as np
import imutils
import cv2

CLASSES = ["aeroplane", "background", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")

vs = VideoStream(src=0).start()

while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	(h, w) = frame.shape[:2]
	# Resize each frame
	#resized_image = cv2.resize(frame, (300, 300))
	resized_image = frame
	
	blob = cv2.dnn.blobFromImage(resized_image, (1/127.5), (300, 300), 127.5, swapRB=True)
	
	net.setInput(blob) # net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
	# Predictions:
	predictions = net.forward()

	for i in np.arange(0, predictions.shape[2]):
	
		confidence = predictions[0, 0, i, 2]
		
		if confidence > 0.4: # 20 percetc
			
			idx = int(predictions[0, 0, i, 1])
			# Get the bounding box coordinates
			box = predictions[0, 0, i, 3:7] * np.array([w, h, w, h])
			# Example, box = [130.9669733   76.75442174 393.03834438 224.03566539]
			# Convert them to integers: 130 76 393 224
			(startX, startY, endX, endY) = box.astype("int")
			
			#if CLASSES[idx] == "person":
			if True:
				label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)

				cv2.rectangle(frame, (startX, startY), (endX, endY),
					COLORS[idx], 2)
				y = startY - 15 if startY - 15 > 15 else startY + 15
			
				cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

	# show the output frame
	cv2.imshow("Frame", frame)

	key = cv2.waitKey(1) & 0xFF

	# Press 'q' key to break the loop
	if key == ord("q"):
		break



cv2.destroyAllWindows()

vs.stop()
