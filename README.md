### Постановка задачи

Данный pet-project нацелен на реализацию детекции и распознавания дорожных знаков на изображениях.

### Используемые данные

За основу взят
[датасет](https://www.kaggle.com/datasets/pkdarabi/cardetection/data)
с фотографиями дорожной сцены, координатами и типами дорожных знаков.

<img src="https://github.com/Gaussiandra/traffic-signs-detection/assets/34653515/747129e7-879e-4407-a626-d8803cb67a75" width="450" />

Датасет содержит синтетические и реальные изображения, 15 типов дорожных знаков, а также разбиение на тренировочные, тестовые и валидационные данные. Каждое изображение в датасете имеет размер 416x416 пикселей.

### Описание подхода

Были использованы такие пакеты:
1. PyTorch Lightning для дообучения YOLOv8n с использованием встроенных аугментаций
2. TensorRT и PyCUDA для оптимального инференса модели в FP16
3. MLflow для логгирования экспериментов
4. Docker, conda и poetry для создания воспроизводимой среды, управления окружениями и зависимостями
5. DVC для возможности версионирования датасета
6. hydra для управления конфигами
7. pre-commit для контроля за качеством кода
8. fire для удобного создания CLIs

Todo:
1. Реализовать квантизацию модели
2. Добавить замеры влияния на скорость/качество квантизации и TRT инференса
3. Добавить визуализацию ONNX графа в netron.app
4. Добавить покрытие тестами


### Пример использования
#### Подготовка окружения
1. `git clone https://github.com/Gaussiandra/traffic-signs-detection.git`
2. `cd traffic-signs-detection/`
3. `docker-compose build`
4. `docker-compose up -d`
5. `docker attach tsd_model`
6. `conda activate dev`
7. `cd tsd/`
8. `dvc pull`

#### Обучение
`python commands.py train_model detector/configs/base_config_64.yaml`\
И затем следить за обучением на localhost:5000


#### Конвертация модели в ONNX
`python commands.py convert_to_onnx detector/configs/base_config_64.yaml checkpoints/train-exp/epoch\=00-val_loss\=62.7809.ckpt model.onnx`

#### Создание TensorRT Engine из ONNX представления
`python commands.py convert_to_trt model.onnx engine.trt`

#### Замер скорости инференса обычного PyTorch
`python commands.py benchmark_torch detector/configs/base_config_64.yaml checkpoints/train-exp/epoch\=00-val_loss\=62.7809.ckpt your_photo.png`

#### Замер скорости инференса на TensorRT
`python commands.py benchmark_trt detector/configs/base_config_64.yaml engine.trt your_photo.png`
