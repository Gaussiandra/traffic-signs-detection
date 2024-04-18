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
аугментаций из
[albumentations](https://github.com/albumentations-team/albumentations).
Обучение происходит с помощью встроенного YOLO trainer.

### Инференс

Схема инференса устроена таким образом:

1. Загрузка тестового изображения: извлечение изображения из тестового набора
   данных для последующей обработки.
2. Предварительная обработка изображения:
   1. Изменение размера изображения на 416x416 пикселей
   2. Нормализация пикселей
   3. Test time augmentations, если передан флаг --tte
3. Генерация предсказаний: на выходе модели получаем предсказанные координаты и
   классы дорожных знаков на изображении.
4. Подсчёт метрик, если имеется разметка
5. Отрисовка изображений и предсказаний на них, если передан флаг --visualize

Таким образом, данный подход позволит эффективно классифицировать и
детектировать дорожные знаки на изображениях с использованием YOLOv8n модели.

### Пример использования
Обучение:

`python commands.py train --dataset_path /workspace/tsd/data/data.yaml --epochs 5`

Валидация:

`python commands.py validate --dataset_path /workspace/tsd/data/data.yaml
--checkpoint_path /workspace/tsd/runs/detect/train/weights/last.pt`

### Структура собранного проекта

├── commands.py\
├── data\
│   ├── data.yaml\
│   ├── README.dataset.txt\
│   ├── README.roboflow.txt\
│   ├── test\
│   ├── train\
│   └── valid\
├── detector\
│   ├── infer.py\
│   ├── \_\_init__.py\
│   ├── \_\_pycache__\
│   ├── train.py\
│   └── validate.py\
├── Dockerfile\
├── poetry.lock\
├── pyproject.toml\
├── README.md\
├── runs\
│   └── detect\
└── yolov8n.pt