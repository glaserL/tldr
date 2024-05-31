from tldr.data.alice_and_bob import get_raw_text
from tldr.srl.inference import SRLParser
from tldr.srl.util import load_trainer_from_path
from tldr.tracking.main import newest_srl_parser

from .conftest import srl_args_path, srl_pred_path


def test_inference():
    text = get_raw_text()

    parser = SRLParser.from_paths(srl_pred_path, srl_args_path)
    parsed = parser.parse(
        text, do_split_into_sentences=True, pred_device="cpu", args_device="cpu"
    )

    sent_ids = [row["sent_id"] for row in parsed]
    assert len(set(sent_ids)) == 7, "There should be 7 sentences in the parsed graph."

def test_champions():
    text = "I like strawberries very much!"
    parser = newest_srl_parser()
    result = parser.parse(text)
    


def test_parser_creation_from_local_path():

    parser = SRLParser.from_paths(srl_pred_path, srl_args_path)
    parsed = parser.parse(
        "I like strawberries.", pred_device="cpu", args_device="cpu"
    )
    assert len(parsed) == 1


def test_parser_creation_from_model():
    pred_trainer = load_trainer_from_path(srl_pred_path)
    args_trainer = load_trainer_from_path(srl_args_path)
    
    parser = SRLParser(pred_trainer, args_trainer)
    parsed = parser.parse(
        "I like strawberries.", pred_device="cpu", args_device="cpu"
    )
    assert len(parsed) == 1
