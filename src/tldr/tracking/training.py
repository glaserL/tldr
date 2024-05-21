import mlflow


def mark_as_champion(trial):
    client = mlflow.MlflowClient()
    run_id = trial.user_attrs["run_id"]
    client.set_tag(run_id, "champion", "true")


def champion_callback(study, _):

    winner = study.user_attrs.get("winner", None)

    if study.best_value and winner != study.best_value:
        study.set_user_attr("winner", study.best_value)
