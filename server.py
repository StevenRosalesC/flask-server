from flask import Flask, request, jsonify, make_response
from io import BytesIO
import base64
import cv2
import easyocr
import numpy as np

app = Flask(__name__)


@app.route('/extract_text', methods=['POST'])
def extract_text():
    # Leer imagen de la petición
    img_data = request.files['image'].read()
    # Convertir datos en formato de imagen OpenCV
    img_array = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Realizar OCR en la imagen
    reader = easyocr.Reader(['es'])
    results = reader.readtext(img)

    # Procesar la imagen y el texto obtenido
    font = cv2.FONT_HERSHEY_SIMPLEX
    spacer = 100
    for detection in results:
        T_LEFT = tuple(detection[0][0])
        B_RIGHT = tuple(detection[0][2])
        TEXT = detection[1]
        img = cv2.rectangle(img, T_LEFT, B_RIGHT, (0, 255, 0), 3)
        img = cv2.putText(img, TEXT, (20, spacer), font,
                          0.5, (0, 255, 0), 2, cv2.LINE_AA)
        spacer += 15

    # Codificar la imagen en formato base64
    img_buffer = BytesIO()
    _, img_encoded = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')

    # Crear la respuesta en formato JSON
    response = {
        'text': [detection[1] for detection in results],
        'image_base64': img_base64
    }

    return make_response(jsonify(response), 200, {'Content-Type': 'application/json'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
