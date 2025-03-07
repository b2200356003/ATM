import numpy as np
import cv2
from flask import Flask, request, jsonify
from ultralytics import YOLO
import pytesseract
from app.atm import produce_output

button_model = None
fingertip_model = None

def create_app():
    app = Flask(__name__)

    global button_model, fingertip_model

    button_model = YOLO("/code/bestLR.pt")
    fingertip_model = YOLO("/code/finger_detector.pt")

    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

    @app.route('/predict', methods=['POST'])
    def predict():
        """
         http://127.0.0.1:3000/predict adresine {
          "image_bytes": [255, 216, 255, ...]
         }
         şeklinde resmin post isteği olarak yollanması gerkeiyor dönüt olarak bir json gelicek içindeki result parametresi
         sonucu içeriyor olacak
        """

        data = request.get_json()
        if 'image_bytes' not in data:
            return jsonify({'error': 'No image bytes provided'}), 400

        file_bytes = np.frombuffer(bytearray(data['image_bytes']), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        result = produce_output(image, button_model, fingertip_model)

        return jsonify({'result': result})

    return app

app = create_app()

if __name__ == '__main__':
    app.run(port=3000, debug=True)
