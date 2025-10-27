import utils


def create_tables() -> bool:
    """
    Lists models and automatically creates database tables if needed.
    """
    import models

    available_models = [model_name for model_name in dir(models) if model_name[0].isupper()]

    with utils.get_db() as db:
        db.create_tables([models.__getattribute__(model_name) for model_name in available_models])

    return True
