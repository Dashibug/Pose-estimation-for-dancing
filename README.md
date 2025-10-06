
DancePose — это система, которая сравнивает движения человека в реальном времени с эталонным танцем по видео.
Используется YOLOv8 Pose для извлечения ключевых точек,
CTR-GCN encoder для извлечения эмбеддингов поз,
и metric-based similarity (triplet loss) для вычисления похожести движений.

📸 Demo
<div align="center"> <img src="https://github.com/yourusername/DancePose/assets/demo.gif" width="700"/> </div>

Left: Reference dance | Right: Your movement in real-time

video (ref) ─▶ YOLOv8-Pose ─▶ keypoints (17×3) ─▶ CTR-GCN Encoder ─▶ embeddings_ref
camera (live) ─▶ YOLOv8-Pose ─▶ keypoints (17×3) ─▶ CTR-GCN Encoder ─▶ embeddings_live
                                               │
                                               ▼
                                    Similarity Score (exp(-‖Δemb‖))

Основные компоненты:

Модуль	Назначение
src/pose/extractor_yolo.py	Извлекает 17-точечные ключевые точки человека из видео / камеры
src/models/encoder_ctrgcn.py	Граф-сверточный энкодер (CTR-GCN) для представления позы в виде эмбеддинга
scripts/extract_reference.py	Создаёт .npz из эталонного видео
scripts/realtime_compare.py	Запускает сравнение: камера + эталонное видео бок-о-бок
src/utils/viz.py	Отрисовка скелета и визуализация сходства

Dependencies:

Python ≥ 3.10

PyTorch ≥ 2.1

OpenCV

Ultralytics (YOLOv8 Pose)

tqdm, numpy


Usage
1️⃣ Извлечь ключевые точки из эталонного видео
python -m scripts.extract_reference --video path/to/ref_dance.mp4


Результат:
data/sessions/ref_dance.npz с полем kpts формы (T,17,3).

2️⃣ Запустить сравнение в реальном времени

(камера + эталон)

python -m scripts.realtime_compare


🔹 Левое окно — эталонное видео
🔹 Правое окно — ваше изображение с наложенным скелетом
🔹 Сверху показывается коэффициент похожести (0.0–1.0)

Нажмите Q для выхода.

📦 Project Structure
DancePose/
│
├── data/
│   ├── sessions/            # .npz с ключевыми точками эталонных видео
│   └── models/              # сохранённые модели (encoder.pt)
│
├── scripts/
│   ├── extract_reference.py # извлекает позы из видео
│   ├── realtime_compare.py  # сравнение в real-time
│
├── src/
│   ├── models/
│   │   └── encoder_ctrgcn.py   # GCN encoder
│   ├── pose/
│   │   └── extractor_yolo.py   # YOLOv8 Pose extractor
│   └── utils/
│       ├── viz.py              # визуализация скелета
│       └── dataset.py          # работа с окнами кадров
│
└── README.md

🧩 Example Output
🎥 Dual-view: [left] reference   [right] realtime pose
Similarity: 0.83