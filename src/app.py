import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from face_detector import FaceDetector
from recognizer import FaceRecognizer
from utils import ROOT, draw_results, load_config


class FaceRecognition:
    def __init__(self, config=None):
        self.config = config or load_config()
        min_size = self.config.get("min_size", [60, 60])
        self.detector = FaceDetector(
            scale=self.config.get("scale", 1.1),
            min_neighbors=self.config.get("min_neighbors", 5),
            min_size=tuple(min_size),
        )
        self.recognizer = FaceRecognizer(threshold=self.config.get("threshold", 0.72))
        self.known_dir = ROOT / self.config.get("known_dir", "images")
        self.model_path = ROOT / self.config.get("model_path", "models/known.npz")
        if self.model_path.exists():
            self.recognizer.load(self.model_path)
        else:
            self.recognizer.train_from_dir(self.known_dir)

    def process(self, frame):
        results = []
        for x, y, w, h in self.detector.detect(frame):
            face = frame[y : y + h, x : x + w]
            if face.size == 0:
                continue
            name, score = self.recognizer.identify(face)
            results.append(((x, y, w, h), name, score))
        return results

    def train(self):
        self.recognizer = FaceRecognizer(threshold=self.config.get("threshold", 0.72))
        count = self.recognizer.train_from_dir(self.known_dir)
        if count:
            self.recognizer.save(self.model_path)
        return count

    def run(self):
        camera = cv2.VideoCapture(self.config.get("camera_index", 0))
        if not camera.isOpened():
            raise RuntimeError("Não foi possível abrir a câmera.")
        try:
            while True:
                ok, frame = camera.read()
                if not ok:
                    break
                draw_results(frame, self.process(frame))
                cv2.imshow("face-recognition", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            camera.release()
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Reconhecimento facial com OpenCV")
    parser.add_argument("--image", help="Analisa uma imagem em vez da câmera")
    parser.add_argument("--output", help="Onde salvar a imagem anotada")
    parser.add_argument("--train", action="store_true", help="Gera o modelo a partir de images/<pessoa>/")
    args = parser.parse_args()

    app = FaceRecognition()
    if args.train:
        count = app.train()
        print(f"Treinado com {count} imagem(ns).")
        return

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"Não foi possível abrir {args.image}")
        results = app.process(frame)
        draw_results(frame, results)
        output = args.output or "data/resultado.jpg"
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output, frame)
        if not results:
            print("Nenhum rosto encontrado.")
        for _box, name, score in results:
            print(f"{name} {score:.2f}")
        print(output)
        return

    app.run()


if __name__ == "__main__":
    main()
