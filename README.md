# traffic-signs-detection

Данный проект призван классифицировать и детектировать дорожные знаки.

За основу взят [данный kaggle датасет](https://www.kaggle.com/datasets/pkdarabi/cardetection/data) с фотографиями дорожной сцены, координатами и типами дорожных знаков.

<img src="https://github.com/Gaussiandra/traffic-signs-detection/assets/34653515/747129e7-879e-4407-a626-d8803cb67a75" width="600" />

У этого датасета есть особенности:
* Присутствуют синтетические фотографии из дорожного симулятора
* Знаки 15 типов: ['Green Light', 'Red Light', 'Speed Limit 10', 'Speed Limit 100', 'Speed Limit 110', 'Speed Limit 120', 'Speed Limit 20', 'Speed Limit 30', 'Speed Limit 40', 'Speed Limit 50', 'Speed Limit 60', 'Speed Limit 70', 'Speed Limit 80', 'Speed Limit 90', 'Stop']
* Фиксированное разбиение на тренировочные(3530 шт), тестовые(801 шт) и валидационные данные(638 шт)
* Каждая картинка имеет размер 416x416 пикселей

В проекте дообучается небольшая [YOLOv8n](https://github.com/ultralytics/ultralytics) с использованием аугментаций из [albumentations](https://github.com/albumentations-team/albumentations). Обучение происходит с помощью встроенного YOLO trainer.
