from backend.app.core.config import settings
from backend.app.ml.source_training import train_v2

if settings.data_mode == "demo":
    raise RuntimeError(
        "V2.2 source-aware training is intended for the four real Kaggle datasets. "
        "Set DATA_MODE=real and run scripts/download_kaggle.py first."
    )

metrics = train_v2(settings.raw_dir)
print(metrics)
