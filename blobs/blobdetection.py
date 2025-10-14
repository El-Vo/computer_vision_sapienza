# Standard imports
import argparse
import os
import sys
import cv2
import numpy as np

def main():
	parser = argparse.ArgumentParser(description="Blob detection (kompatibel mit aktuellen OpenCV-Versionen)")
	parser.add_argument("-i", "--image", default="./blob.jpg", help="Pfad zum Bild (relativ zum Projektordner oder absolut)")
	parser.add_argument("--min-area", type=int, default=1, help="minimale Fläche eines Blobs")
	parser.add_argument("--max-area", type=int, default=10000000 , help="maximale Fläche eines Blobs")
	args = parser.parse_args()

	# Bildpfad auflösen:
	# 1) absolute Pfade übernehmen
	# 2) falls der angegebene Pfad relativ zum aktuellen Arbeitsverzeichnis existiert, diesen verwenden
	# 3) sonst Pfad relativ zum Skriptverzeichnis auflösen
	if os.path.isabs(args.image):
		img_path = args.image
	elif os.path.exists(args.image):
		img_path = args.image
	else:
		img_path = os.path.join(os.path.dirname(__file__), args.image)

	# Pfad normalisieren (entfernt doppelte Komponenten wie 'blobs/../' oder './')
	img_path = os.path.normpath(img_path)

	# Bild laden (Graustufen)
	im = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
	if im is None:
		print(f"Fehler: Bild nicht gefunden oder kann nicht geladen werden: {img_path}", file=sys.stderr)
		sys.exit(1)

	# Parameter für den SimpleBlobDetector setzen
	params = cv2.SimpleBlobDetector_Params()
	params.filterByArea = True
	params.minArea = args.min_area
	params.maxArea = args.max_area
	# Weitere Filteroptionen (bei Bedarf aktivieren)
	#params.filterByCircularity = True
	#params.minCircularity = 0.5
	#params.filterByConvexity = True
	#params.minConvexity = 0.5
	#params.filterByInertia = True
	#params.minInertiaRatio = 0.5

	# Erstelle Detector — kompatibel mit verschiedenen OpenCV-Versionen
	try:
		detector = cv2.SimpleBlobDetector_create(params)
	except AttributeError:
		# ältere OpenCV-Versionen nutzen den Konstruktor direkt
		detector = cv2.SimpleBlobDetector(params)

	# Blobs detektieren
	keypoints = detector.detect(im)

	# Für farbige Ausgabe in BGR konvertieren (sicher für drawKeypoints)
	im_color = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

	# Gefundene Blobs als rote Kreise zeichnen
	im_with_keypoints = cv2.drawKeypoints(im_color, keypoints, None, (0,0,255),
										 cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

	# Fenster anzeigen
	cv2.imshow("Keypoints", im_with_keypoints)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()