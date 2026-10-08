# Face Recognition — reconhecimento local com OpenCV

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.10.0.84-5C3EE8?logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.26+-013243?logo=numpy&logoColor=white)
![PyYAML](https://img.shields.io/badge/PyYAML-6+-CB171E?logo=yaml&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-python%3A3.11--slim-2496ED?logo=docker&logoColor=white)

A câmera, ou um arquivo de imagem, passa por um cascade Haar frontal. O recorte é reduzido a 100×100 em cinza, normalizado, e comparado por cosseno com o que foi gravado a partir de `images/<pessoa>/`. Abaixo do limiar 0,72 o nome é `Desconhecido`. Serve para um ensaio local. Não é identificação em escala.

| Etapa | Código | Limite |
| --- | --- | --- |
| Achar o rosto | `haarcascade_frontalface_default`, `scale` 1,1, `minNeighbors` 5, `minSize` 60×60 | só frontal; perfil e rosto pequeno passam batido |
| Dizer quem é | vetor do recorte contra a matriz de cada pessoa | no treino, `train_from_dir` embute a foto inteira, não o recorte do rosto; na hora de identificar, o vetor é só do retângulo detectado |

O pacote pinado é `opencv-python-headless`. `cv2.imshow` não vem nessa build: `python src/app.py` (câmera) quebra na janela. `--train` e `--image` não precisam de GUI.

## Stack

- Python 3.11 no `Dockerfile` (`python:3.11-slim`)
- opencv-python-headless 4.10.0.84
- NumPy >= 1.26 e PyYAML >= 6
- `config.yaml`: `camera_index` 0, `threshold` 0.72, `known_dir` `images`, `model_path` `models/known.npz`

## Estrutura

```
.
├── config.yaml
├── requirements.txt
├── Dockerfile              # instala deps e o default é --help
├── .dockerignore
├── src/
│   ├── app.py             # --train, --image ou câmera
│   ├── face_detector.py   # cascade Haar
│   ├── recognizer.py      # embedding, limiar, npz
│   └── utils.py           # YAML e retângulo com o nome
├── images/.gitkeep        # fotos em images/<pessoa>/*.{jpg,jpeg,png}
├── models/.gitkeep        # known.npz e known.json saem do --train
├── data/.gitkeep          # saída padrão data/resultado.jpg
└── tests/test_pipeline.py # unittest, sem câmera
```

`images/`, `models/` e `data/` estão vazios no git. Sem `models/known.npz`, o startup tenta treinar a partir de `images/`. Pasta vazia deixa o reconhecedor sem ninguém e todo rosto sai `Desconhecido`.

## Como rodar

```bash
git clone https://github.com/gabrielteramae/face-recognition.git
cd face-recognition
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/app.py --train
python src/app.py --image foto.jpg --output data/resultado.jpg
```

`--train` lê `images/<pessoa>/` e, se achou ao menos uma imagem, grava `models/known.npz` e `models/known.json`. `--image` imprime `nome score` (ou “Nenhum rosto encontrado.”) e salva o JPEG. Sem `--output`, o caminho é `data/resultado.jpg`.

O `Dockerfile` existe. A câmera não entra no container; o `CMD` é `python src/app.py --help`.

```bash
docker build -t face-recognition .
docker run --rm -v "$PWD/images:/app/images" -v "$PWD/foto.jpg:/app/foto.jpg" face-recognition python src/app.py --image foto.jpg
```

## Testes realizados

`tests/test_pipeline.py` usa `unittest` e imagens sintéticas (bloco de cor mais ruído), sem câmera e sem cascade nos três primeiros casos:

- o vetor mais próximo devolve o nome certo quando o score passa de 0,8
- limiar 0,99 manda um padrão diferente para `Desconhecido`
- `save` / `load` em `/tmp/known-test.npz` recupera o nome
- `FaceRecognition.process` numa imagem 480×640 preta devolve lista vazia

```bash
python -m unittest tests.test_pipeline -v
```

---

© 2026 Gabriel Teramae Chan
