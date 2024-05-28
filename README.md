# Summary

# Usage

## Installation

Simply clone the repository and install it locally:

```bash
git clone https://github.com/glaserL/tldr && cd tldr
pip install .
```

In case you want to use the training experiments, install the `train` extra

```bash
pip install ".[train]"
```

## CLI

There is a rudimentary command line interface.

`tldr parse <TEXT>` will output the srl of the text.

`tldr summarize [PATH]` will summarize each document provided. We currently support raw text and
TODO pdf files. In case of multiple text files it will also output document clustering information.
For more information on the clustering, see TODO.

# Development

## Setup

```bash
git clone https://github.com/glaserL/tldr && cd tldr
pip install -e ".[dev, train]"
```

## INTERNAL
on neuron you'll need to use an old torch version. after installing swap torch:
`pip install torch==1.12.1+cu102 --extra-index-url https://download.pytorch.org/whl/cu102`
note it might work with `pip install -e . --extra-index-url https://download.pytorch.org/whl/cu102`

## Structure

The project is split into 5 CHECK
modules

- `srl` is concerned with training and inference of the shallow srl data
- `summarize` can summarize sentence and document graphs with a variety of techniques.
- `generate` serializes sentence graphs into natural language.
- `data` really only creates sentences graph objects from the srl parses.
- `tracking` contains utilities for training but also recovering already trained models.

Every model based component can be either created using mlflow or from local data.
