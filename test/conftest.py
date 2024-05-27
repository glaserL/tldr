from pathlib import Path

pred_run = "traveling-koi-720"
args_run = "merciful-owl-124"
gen_run = "marvelous-shark-769"

neuron_artifact_path = Path("/home/lglaser/serve/mlflow/mlartifacts/")
srl_pred_path = (
    neuron_artifact_path
    / "854113086230627037/f34dcdfd31a44dff9f3bf85b063fad1e/artifacts/output/checkpoint-30590/artifacts/checkpoint-30590"
)
srl_args_path = (
    neuron_artifact_path
    / "651566448186103026/5255d0a00d60447f8e76df2d40a78511/artifacts/output/checkpoint-31104/artifacts/checkpoint-31104"
)
