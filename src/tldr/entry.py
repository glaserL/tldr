# ruff: noqa: T201
from argparse import ArgumentParser

from tldr.main import TLDR
from tldr.tracking.main import SRLParserFromMLflow, newest_srl_parser

parser = ArgumentParser(description="Simple interface to backbone.")
subparsers = parser.add_subparsers(dest="command")


parser_parse = subparsers.add_parser("parse", help="Annotate a sentence with ssrl annotations")
parser_parse.add_argument(
    "text",
    type=str,
    help="Text to parse",
)
parser_parse.add_argument("-p", "--predicate_model", type=str, help="Pred model")
parser_parse.add_argument("-a", "--argument_model", type=str, help="Args model")

parser_pred = subparsers.add_parser("pred")

parser_pred.add_argument(
    "text",
    type=str,
    help="Text to parse",
)
parser_pred.add_argument("-p", "--predicate_model", type=str, help="Pred model", required=True)
parser_pred.add_argument("-a", "--argument_model", type=str, help="Args model", required=True)

parser_summarize = subparsers.add_parser("summarize", help="Summarize a text")
parser_summarize.add_argument(
    "paths",
    nargs="*",
    help="Text files to summarize",
)


# TODO: Hier Leos pdf parsing einbauen?


def run():
    args = parser.parse_args()

    if args.command == "parse":
        if args.predicate_model is None and args.argument_model is None:
            srl_parser = newest_srl_parser()
        elif args.predicate_model is not None and args.argument_model is not None:
            srl_parser = SRLParserFromMLflow.from_run_names(args.predicate_model, args.argument_model)
        else:
            parser.error("If specifying model run names, both have to be set!")
        result = srl_parser.parse(args.text)
        for parsed in result:
            print(parsed)
    elif args.command == "pred":
        srl_parser = SRLParserFromMLflow.from_run_names(args.predicate_model, args.argument_model)
        result = srl_parser.parse(args.text, tasks=["pred"])
        for parsed in result:
            print(parsed)
    elif args.command == "summarize":

        big_thingy = TLDR.default()

        for result in big_thingy.run_batch(args.paths):
            top_line = f"Document: {result['id']}"
            if "cluster_id" in result:
                top_line += f", Cluster: {result['cluster_id']}"
            print(f"{top_line}\n{result['generated_text']}")

    else:
        parser.error("Invalid command, use --help for more information")

if __name__ == "__main__":
    run()