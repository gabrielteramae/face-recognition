import json
from pathlib import Path

import cv2
import numpy as np


class FaceRecognizer:
    def __init__(self, threshold=0.72):
        self.threshold = threshold
        self._matrix = {}

    def embedding(self, face_bgr):
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (100, 100))
        gray = cv2.equalizeHist(gray)
        vector = gray.astype(np.float32).ravel()
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def learn(self, name, face_bgr):
        current = self._matrix.get(name)
        vector = self.embedding(face_bgr)[None, :]
        self._matrix[name] = vector if current is None else np.vstack([current, vector])

    def train_from_dir(self, folder):
        folder = Path(folder)
        count = 0
        if not folder.exists():
            return count
        for person_dir in sorted(path for path in folder.iterdir() if path.is_dir()):
            for image_path in sorted(person_dir.iterdir()):
                if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                image = cv2.imread(str(image_path))
                if image is None:
                    continue
                self.learn(person_dir.name, image)
                count += 1
        return count

    def identify(self, face_bgr):
        if not self._matrix:
            return "Desconhecido", 0.0
        vector = self.embedding(face_bgr)
        best_name, best_score = "Desconhecido", -1.0
        for name, matrix in self._matrix.items():
            score = float((matrix @ vector).max())
            if score > best_score:
                best_name, best_score = name, score
        if best_score < self.threshold:
            return "Desconhecido", best_score
        return best_name, best_score

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        names = list(self._matrix)
        arrays = {f"f{index}": self._matrix[name] for index, name in enumerate(names)}
        np.savez(path, **arrays)
        path.with_suffix(".json").write_text(json.dumps(names), encoding="utf-8")

    def load(self, path):
        path = Path(path)
        names = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        data = np.load(path)
        self._matrix = {name: data[f"f{index}"] for index, name in enumerate(names)}
