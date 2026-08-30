from catboost import CatBoostClassifier
from app.config import MODEL_PATH

_model: CatBoostClassifier | None = None


def load_model() -> CatBoostClassifier:
    """Загружает модель один раз и кеширует в модуле."""
    global _model
    if _model is None:
        _model = CatBoostClassifier()
        _model.load_model(str(MODEL_PATH))
    return _model


def get_model() -> CatBoostClassifier:
    """Dependency для FastAPI — возвращает уже загруженную модель."""
    if _model is None:
        return load_model()
    return _model