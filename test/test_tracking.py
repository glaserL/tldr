from tldr.srl.util import load_model, load_model_from_mlflow
from tldr.tracking.main import newest_srl_parser
from tldr.tracking.query import get_model

from .conftest import srl_pred_path


def test_get_model_via_mlflow():
    model_info = get_model("SRL-pred")
    model = load_model_from_mlflow(model_info)
    assert model is not None


def test_get_model_via_path():

    model = load_model(srl_pred_path)
    assert model is not None


def test_parser_creation_from_registered_models():

    srl_parser = newest_srl_parser()
    assert srl_parser is not None
    assert srl_parser.parse("I like strawberries", pred_device="cpu", args_device="cpu")
