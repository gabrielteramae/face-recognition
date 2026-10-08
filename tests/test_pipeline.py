import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app import FaceRecognition
from recognizer import FaceRecognizer


def pattern(seed, color):
    rng = np.random.default_rng(seed)
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    image[:, :] = color
    noise = rng.integers(0, 30, image.shape, dtype=np.uint8)
    return cv2.add(image, noise)


class RecognizerTest(unittest.TestCase):
    def test_reconhece_o_padrao_mais_proximo(self):
        recognizer = FaceRecognizer(threshold=0.8)
        ana = pattern(1, (180, 40, 40))
        bia = pattern(2, (40, 40, 180))
        recognizer.learn("Ana", ana)
        recognizer.learn("Bia", bia)
        name, score = recognizer.identify(ana)
        self.assertEqual(name, "Ana")
        self.assertGreater(score, 0.8)

    def test_fica_desconhecido_abaixo_do_limite(self):
        recognizer = FaceRecognizer(threshold=0.99)
        recognizer.learn("Ana", pattern(1, (180, 40, 40)))
        name, _score = recognizer.identify(pattern(9, (20, 200, 20)))
        self.assertEqual(name, "Desconhecido")

    def test_salva_e_carrega(self):
        recognizer = FaceRecognizer()
        recognizer.learn("Ana", pattern(1, (180, 40, 40)))
        path = Path("/tmp/known-test.npz")
        recognizer.save(path)
        loaded = FaceRecognizer()
        loaded.load(path)
        name, _score = loaded.identify(pattern(1, (180, 40, 40)))
        self.assertEqual(name, "Ana")

    def test_imagem_sem_rosto_nao_quebra(self):
        app = FaceRecognition()
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        self.assertEqual(app.process(blank), [])


if __name__ == "__main__":
    unittest.main()
