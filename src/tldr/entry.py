# ruff: noqa: T201
from argparse import ArgumentParser

from tldr.main import TLDR
from tldr.tracking.main import newest_srl_parser

parser = ArgumentParser(description="Simple interface to backbone.")
subparsers = parser.add_subparsers(dest="command")


parser_parse = subparsers.add_parser("parse", help="Remove an item by its index.")
parser_parse.add_argument(
    "text",
    type=str,
    help="Text to parse",
)

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
        srl_parser = newest_srl_parser()
        result = srl_parser.parse(args.text)
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
