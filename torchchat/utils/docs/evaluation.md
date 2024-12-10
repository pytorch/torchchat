> [!WARNING]
> Files in this directory may be outdated, incomplete, scratch notes, or a WIP. torchchat provides no guarantees on these files as references. Please refer to the root README for stable features and documentation.


# Evaluation Features

<!--

[shell default]: ./install/install_requirements.sh

[shell default]: TORCHCHAT_ROOT=${PWD} ./torchchat/utils/scripts/install_et.sh

-->

Torchchat provides evaluation functionality for your language model on
a variety of tasks using the
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
library.

## Usage

The evaluation mode of `torchchat.py` script can be used to evaluate your language model on various tasks available in the `lm_eval` library such as "wikitext". You can specify the task(s) you want to evaluate using the `--tasks` option, and limit the evaluation using the `--limit` option. If no task is specified, the task will default to evaluating on "wikitext".

## Examples

### Python

Running wikitext for 10 iterations
```
python3 torchchat.py eval stories15M --tasks wikitext --limit 10
```

Running wikitext with torch.compile for 10 iterations
```
python3 torchchat.py eval stories15M --compile --tasks wikitext --limit 10
```

Running multiple tasks and calling eval.py directly (with torch.compile):
```
python3 torchchat.py eval stories15M --compile --tasks wikitext hellaswag
```

### exported PTE

Running an exported model with ExecuTorch (as PTE)
```
python3 torchchat.py export stories15M --output-pte-path stories15M.pte
python3 torchchat.py eval stories15M --pte-path stories15M.pte
```

Running multiple tasks and calling eval.py directly (with PTE):
```
python3 torchchat.py eval stories15M --pte-path stories15M.pte --tasks wikitext hellaswag
```

### exported AOTI

Running an exported model with AOT Inductor (DSO model)
```
python3 torchchat.py export stories15M --dtype fast16 --output-dso-path stories15M.so
python3 torchchat.py eval stories15M --dtype fast16 --dso-path stories15M.so
```

Running multiple tasks and calling eval.py directly (with AOTI):
```
python3 torchchat.py eval stories15M --dso-path stories15M.so --tasks wikitext hellaswag
```

For more information and a list of tasks/metrics see [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

[end default]: end
