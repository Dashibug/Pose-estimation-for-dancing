# 🕺 **DancePose: Real-Time Dance Similarity System**

Система для сравнения движений человека в реальном времени с эталонным танцем по видео.  
Используется YOLOv8 Pose для извлечения ключевых точек, CTR-GCN Encoder для эмбеддингов поз и metric-based similarity для оценки схожести движений.

---

## Overview

**DancePose** анализирует позы из видео и камеры в реальном времени, вычисляя степень похожести между вашим движением и эталоном.

<div align="center">
  <img src="https://github.com/user-attachments/assets/be8c5f8d-88d8-4d88-9fd3-c6f9a94559cd"/>
  <br>
  <em>Left: Your movement in real-time | Right: Reference dance</em>
</div>

---

## Архитектура системы

```text
                ┌──────────────────────────────┐
                │        Reference Video        │
                └──────────────┬───────────────┘
                               │
                        YOLOv8 Pose (17×3)
                               │
                        CTR-GCN Encoder
                               │
                         embeddings_ref
                               │
                               ▼
                ┌──────────────────────────────┐
                │          Similarity          │
                └──────────────────────────────┘
                               ▲
                               │
                         embeddings_live
                               │
                        CTR-GCN Encoder
                               │
                        YOLOv8 Pose (17×3)
                               │
                ┌──────────────────────────────┐
                │         Live Camera          │
                └──────────────────────────────┘
```

## Полный пайплайн запуска

Быстрый старт, пакетная подготовка эталонов, визуализация `.npz`, обучение metric-head (triplet), запуск realtime-сравнения.

### Требования
- Python ≥ 3.10
- PyTorch ≥ 2.1
- OpenCV
- Ultralytics (YOLOv8 Pose)

Установка:
```bash
pip install -r requirements.txt
```

## Скрипты

|             Скрипт               |                    Назначение                  |                         Пример запуска                      |
|:---------------------------------|:-----------------------------------------------|:------------------------------------------------------------|
| `scripts/extract_from_video.py` | Извлечение поз из одного видео → `.npz` | `python -m scripts.extract_from_video --video path/to.mp4 --out data/sessions/name.npz` |
| `scripts/build_ref_library.py` | Пакетная обработка видео/npz → единая библиотека эталонов | `python -m scripts.build_ref_library --src data/ref_videos --out data/ref/library.npz` |
| `scripts/preview_npz.py` | Быстрый просмотр содержимого `.npz` (форма, FPS, длина, визуализация) | `python -m scripts.preview_npz --npz data/sessions/name.npz` |
| `scripts/train_triplet.py` | Обучение metric-head / CTR-GCN-тюнинга с triplet loss | `python -m scripts.train_triplet --config default.yaml` |
| `scripts/realtime_compare.py` | Сравнение в реальном времени (камера + эталон/библиотека) | `python -m scripts.realtime_compare --ref data/ref/library.npz` |

