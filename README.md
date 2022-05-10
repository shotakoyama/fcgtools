# fcgtools

This repository gives code with Tokyo Tech NLP Group submission to the [GenChal 2022 Feedback Comment Generation Task](https://fcg.sharedtask.org/).

## How to use?

### Experiment

Experiment code is under [shotakoyama:fcgwork](https://github.com/shotakoyama/fcgwork) repository.

### Installation

```
# under the fcgtools directory
pip install -e .
```

I used `torch==1.11.0+cu113` to use `NVIDIA A100` GPU.
Please use any version of `torch` that suits your environment.

### Environment

I used environment below.

- python 3.8.13
- cuda 11.3.1
- cudnn 8.3.3


## `fcg-prepare`

This command preprocesses `*.prep_feedback_comment.public.tsv` to make inputs of `fcg-preproc`.

## `fcg-preproc`

This command makes binary dataset for `fcg-train`.

## `fcg-apply-m2`

This command applies ERRANT M2 file to `*.tsv`.

## `fcg-train`

This command trains `GPT-2` or `BART` FCG models.

## `fcg-generate`

This command generates feedback comment using trained model.

## `fcg-filter`

This command filters generated feedback comments with a simple heuristic.
If the `<<quoted word>>` or its inflections are not in the source erroneous sentence, this script converts the feedback comment to `<NO_COMMENT>`.

## `fcg-score`

This command scores F-score of the GenChal 2022 Task.

