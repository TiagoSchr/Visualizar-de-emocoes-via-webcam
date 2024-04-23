import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  #logs do TensorFlow

#modelo treinado
try:
    model = load_model('Nome do modelo')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
except Exception as e:
    print("Erro ao carregar o modelo:", e)
    exit()

#captura de vídeo
cap = cv2.VideoCapture(0)
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except Exception as e:
    print("Erro ao carregar o classificador de faces:", e)
    cap.release()
    exit()

#As emoções
emotions_pt = ('bravo', 'desgosto', 'medo', 'feliz', 'triste', 'surpresa', 'neutro')

while True:
    ret, frame = cap.read()
    if not ret:
        print("Falha ao capturar imagem da câmera.")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = roi_gray.astype('float') / 255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)

        predictions = model.predict(roi)
        max_index = int(np.argmax(predictions))
        predicted_emotion = emotions_pt[max_index]

        cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Titulo da pagina', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
