# Face Recognition — OpenCV

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.10-5C3EE8?logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.26-013243?logo=numpy&logoColor=white)

Reconhecimento facial local: a câmera encontra o rosto, compara com as pessoas cadastradas e mostra o nome em cima da imagem.

## Por que separar em duas etapas?

| Etapa | Módulo | Motivo |
| --- | --- | --- |
| Achar o rosto | `face_detector.py` | Recorte fixo, cascade Haar do OpenCV, sem treino |
| Dizer quem é | `recognizer.py` | Compara o recorte com as fotos de `images/<pessoa>/` |

`app.py` junta as duas: detecta, identifica e desenha o retângulo. Sem rosto cadastrado, o nome fica **Desconhecido**.

## Stack

- **OpenCV** para a câmera, o cascade Haar e o desenho
- **NumPy** para o vetor de cada rosto e a comparação
- **PyYAML** para câmera, limiar e caminhos em `config.yaml`

## Estrutura

```
src/
├── app.py              # câmera, imagem avulsa e treino
├── face_detector.py    # encontra o rosto
├── recognizer.py       # compara com os cadastros
└── utils.py            # config e desenho
images/<pessoa>/        # fotos de treino, uma pasta por pessoa
models/                 # encodings gerados pelo --train
config.yaml             # limiar, câmera e caminhos
```

## Como rodar

```bash
git clone https://github.com/gabrielteramae/face-recognition.git
cd face-recognition
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Coloque as fotos em pastas com o nome da pessoa, por exemplo `images/Gabriel/1.jpg` e `images/Gabriel/2.jpg`.

```bash
python src/app.py --train
python src/app.py
python src/app.py --image foto.jpg --output data/resultado.jpg
```

Na câmera, `q` fecha. O limiar fica em `config.yaml`: quanto maior, mais exigente o reconhecimento.

### Docker

A câmera fica na máquina local. O container serve para analisar um arquivo.

```bash
docker build -t face-recognition .
docker run --rm -v "$PWD/images:/app/images" -v "$PWD/foto.jpg:/app/foto.jpg" face-recognition python src/app.py --image foto.jpg
```

## Comandos

| Comando | O que faz |
| --- | --- |
| `python src/app.py --train` | Lê `images/<pessoa>/` e grava `models/known.npz` |
| `python src/app.py` | Abre a câmera e escreve o nome em cada rosto |
| `python src/app.py --image foto.jpg` | Analisa um arquivo e salva `data/resultado.jpg` |

## Testes

```bash
python -m unittest tests.test_pipeline -v
```

Os testes conferem o reconhecimento do padrão mais próximo, o caso abaixo do limiar, o save/load do modelo e uma imagem sem rosto. Não precisam de câmera.

O reconhecimento usa a aparência do recorte. Serve para um projeto local, não para identificação em escala.

---

© 2026 Gabriel Teramae Chan
