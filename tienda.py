import cv2
from ultralytics import YOLO

#Selección del modelo
#models = YOLO('yolov8x.pt')
#models = YOLO('yolov8l.pt')
#models = YOLO('yolov8n.pt')
models = YOLO('yolov8s.pt')

#Ruta del video
video_path = "videos/ch02_20230429171808.mp4"
#video_path = "videos/ch02_20230528125240.mp4"

#Se lee el video con libreria cv2
cap = cv2.VideoCapture(video_path)

#Contador de frames
count = 0

while cap.isOpened():
    success, frame = cap.read()

    #Edición de imagen para mejores resultados
    frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=-40)

    if success:
        #Toma foto cada 60 frames
        if count % 60 == 0:
            results = models(frame, classes=0, agnostic_nms=True)
            annotated_frame = results[0].plot()
            filename = f"runs/frame_{count}.jpg"
            cv2.imwrite(filename, annotated_frame)
        else:
            pass

        #Aumenta los frames (se sabe qué numero con el código frame.py)
        count += 12
        cv2.imshow("My tienda camera", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()
