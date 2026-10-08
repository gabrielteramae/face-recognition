# face-recognition

Detecção e reconhecimento de rostos com OpenCV. A câmera marca cada rosto e compara com as pessoas cadastradas em `images/`.

## Estrutura

```
src/app.py              loop da câmera e da imagem
src/face_detector.py    encontra o rosto
src/recognizer.py       compara com os cadastros
src/utils.py            config e desenho
images/<pessoa>/        fotos de treino
config.yaml             limiar, câmera e caminhos
```

## Como rodar

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Coloque fotos em `images/Gabriel/1.jpg`, `images/Gabriel/2.jpg` e assim por diante. Uma pasta por pessoa.

```bash
python src/app.py --train
python src/app.py
python src/app.py --image foto.jpg --output data/resultado.jpg
```

Na câmera, `q` fecha. O limiar fica em `config.yaml`: quanto maior, mais exigente.

O reconhecimento usa a aparência do recorte do rosto. Serve para um projeto local, não para identificação em escala.

## Docker

```bash
docker build -t face-recognition .
docker run --rm -v "$PWD/images:/app/images" face-recognition python src/app.py --image foto.jpg
```

A câmera precisa da máquina local. O container serve para analisar um arquivo.
