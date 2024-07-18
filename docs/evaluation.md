> [!WARNING]
> Files in this directory may be outdated, incomplete, scratch notes, or a WIP. torchchat provides no guarantees on these files as references. Please refer to the root README for stable features and documentation.


# Evaluation Features

[shell default]: ./install_requirements.sh


Torchchat provides evaluation functionality for your language model on
a variety of tasks using the
[lm-evaluation-harness](https://github.com/facebookresearch/lm_eval)
library.

## Usage

The evaluation mode of `torchchat.py` script can be used to evaluate your language model on various tasks available in the `lm_eval` library such as "wikitext". You can specify the task(s) you want to evaluate using the `--tasks` option, and limit the evaluation using the `--limit` option. If no task is specified, it will default to evaluating on "wikitext".

**Examples**

Running wikitext for 10 iterations
```
python3 torchchat.py eval stories15M --tasks wikitext --limit 10
```

Running an exported model
```
python3 torchchat.py export stories15M --output-pte-path stories15M.pte
python3 torchchat.py eval stories15M --pte-path stories15M.pte
```

Running multiple tasks and calling eval.py directly:
```
python3 eval.py stories15M --pte-path stories15M.pte --tasks wikitext hellaswag
```

For more information and a list of tasks/metrics see [lm-evaluation-harness](https://github.com/facebookresearch/lm_eval).

[end default]: end
