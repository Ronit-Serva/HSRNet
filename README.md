# EMNIST LeNet-5 Character Recognizer

One PyTorch implementation of the LeNet-5-style CNN described in LeCun,
Bottou, Bengio, and Haffner's 1998 *Gradient-Based Learning Applied to
Document Recognition*. It classifies the 62 case-sensitive EMNIST ByClass
labels: digits `0-9`, uppercase `A-Z`, and lowercase `a-z`.

The architecture retains trainable average-pooling/subsampling layers and
`tanh` activations. Its C3 convolution is fully connected (the standard
PyTorch form), rather than using the paper's historical sparse connection
table. Training intentionally uses modern multiclass cross-entropy logits.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Verify preprocessing

EMNIST is stored transposed by torchvision. The data loader rotates it 90°
counter-clockwise and flips it horizontally before resizing and normalising.
Inspect the labelled output before reporting results:

```bash
python -m scripts.preview_data --output runs/emnist_preview.png
```

## Train

Edit [`config.yaml`](config.yaml) to set epochs, batch size, learning rate,
device, paths, and all other training variables. Then run:

```bash
python -m src.train
```

The command downloads EMNIST ByClass into `data/` on its first run. It writes
`config.json`, `history.csv`, `last.pt`, `best.pt`, `test_metrics.json`,
`confusion_matrix.csv`, and `per_class_accuracy.json` to the output directory.

Resume a stopped run with:

Set `resume: runs/emnist_lenet5/last.pt` in `config.yaml`, then run the same
command. A different experiment file can be selected with `--config` and a
setting can still be overridden for a one-off run, for example
`--epochs 5 --device cpu`.

Use `--device cpu` or `--device cuda` to select hardware and `--no-download`
when the data has already been downloaded.

## Collaboration conventions

Keep model and training changes in `src/`, experiment invocations configurable
from the command line, and generated datasets/checkpoints under ignored
`data/` and `runs/` directories. Record the output directory and commit hash
when adding evaluation reports, plots, or comparisons.
