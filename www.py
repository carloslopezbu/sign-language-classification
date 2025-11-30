import os

import videoproc as vp

base_dir = os.getcwd()
video = os.path.join(base_dir, "datamining/videos/video2.mp4")
east = os.path.join(base_dir, "models/frozen_east_text_detection.pb")

# CAMBIO IMPORTANTE: Apunta al archivo .onnx
ocr_model = os.path.join(base_dir, "models/text-rec.onnx")
# Asegúrate de que en 'models/' también esté 'alphabet_94.txt'

vp.align_sign_text_sequences(video, east, ocr_model, 0.35)
