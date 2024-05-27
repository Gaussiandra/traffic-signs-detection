### Постанновка задачи

Для классификации и детекции дорожных знаков на изображениях, предлагается
использовать YOLOv8n модель, которая будет дообучаться на датасете с
фотографиями дорожной сцены.

### Используемые данные

За основу взят
[данный kaggle датасет](https://www.kaggle.com/datasets/pkdarabi/cardetection/data)
с фотографиями дорожной сцены, координатами и типами дорожных знаков.

<img src="https://github.com/Gaussiandra/traffic-signs-detection/assets/34653515/747129e7-879e-4407-a626-d8803cb67a75" width="600" />

У этого датасета есть особенности:

- Присутствуют синтетические фотографии из дорожного симулятора
- Знаки 15 типов: ['Green Light', 'Red Light', 'Speed Limit 10', 'Speed Limit
  100', 'Speed Limit 110', 'Speed Limit 120', 'Speed Limit 20', 'Speed Limit
  30', 'Speed Limit 40', 'Speed Limit 50', 'Speed Limit 60', 'Speed Limit 70',
  'Speed Limit 80', 'Speed Limit 90', 'Stop']
- Фиксированное разбиение на тренировочные(3530 шт), тестовые(801 шт) и
  валидационные данные(638 шт)
- Каждая картинка имеет размер 416x416 пикселей

### Подоход к моделированию

В проекте дообучается небольшая
[YOLOv8n](https://github.com/ultralytics/ultralytics) с использованием
встроенных в YOLO аугментаций. Обучение происходит с помощью фреймворка Pytorch
Lightning.

### Инференс

Схема инференса устроена таким образом:

1. Загрузка тестового изображения: извлечение изображения из тестового набора
   данных для последующей обработки.
2. Предварительная обработка изображения:
   1. Изменение размера изображения на 416x416 пикселей (или на значение,
      указанное в конфиге при обучении)
   2. Нормализация пикселей
3. Генерация предсказаний: на выходе модели получаем предсказанные координаты и
   классы дорожных знаков на изображении.

Таким образом, данный подход позволит эффективно классифицировать и
детектировать дорожные знаки на изображениях с использованием YOLOv8n модели.

### Запуск проекта

1. `git clone https://github.com/Gaussiandra/traffic-signs-detection.git`
2. `docker-compose build`
3. `docker-compose up -d`
4. `docker attach tsd_model`
5. `conda activate dev`
6. `python commands.py train detector/configs/base_config_64.yaml`
7. Следить за процессом обучения на localhost:5000

### Пример использования

Обучение:

`python commands.py train detector/configs/base_config_64.yaml`

Инференс:

`python commands.py infer detector/configs/base_config_64.yaml checkpoints/train-exp/epoch=03-val_loss=52.6284.ckpt traffic_sign.png`

### Структура собранного проекта

.\
|-- Dockerfile\
|-- README.md\
|-- checkpoints\
| -- train-exp\
|-- commands.py\
|-- data\
| |-- README.dataset.txt\
| |-- README.roboflow.txt\
| |-- data.yaml\
| |-- test\
| |-- train\
| -- valid\
|-- data.dvc\
|-- detector\
| |-- \_\_init\_\_.py\
| |-- \_\_pycache\_\_\
| |-- configs\
| |-- data.py\
| |-- infer.py\
| |-- model.py\
| -- train.py\
|-- docker-compose.yml\
|-- mlflow.dockerfile\
|-- poetry.lock\
|-- pyproject.toml\
|-- weights\
| -- yolov8n.pt
