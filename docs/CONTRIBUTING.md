> [!WARNING]
> Files in this directory may be outdated, incomplete, scratch notes, or a WIP. torchchat provides no guarantees on these files as references. Please refer to the root README for stable features and documentation.


# Contributing to torchchat
We want to make contributing to this project as easy and transparent as possible.

&nbsp;

## Dev install
You should first [fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) the torchchat repository
and then clone your forked repository.

```git clone https://github.com/<YOUR_GITHUB_USER>/torchchat.git```

Then navigate into the newly cloned repo and install dependencies needed for development.
The following steps require that you have [Python 3.10](https://www.python.org/downloads/release/python-3100/) installed.

[skip default]: begin
```bash
# get the code
git clone https://github.com/pytorch/torchchat.git
cd torchchat

# set up a virtual environment
python3 -m venv .venv
source .venv/bin/activate
```
[skip default]: end
```
# install dependencies
./install/install_requirements.sh
```

Installations can be tested by

```bash
# ensure everything installed correctly
python3 torchchat.py --help
```

&nbsp;

## Contributing workflow
We actively welcome your pull requests.

1. Create your new branch from `main` in your forked repo, with a name describing the work you're completing e.g. `add-feature-x`.
2. If you've added code that should be tested, add tests. Ensure all tests pass. See the [testing section](#testing) for more information.
3. If you've changed APIs, [update the documentation](#updating-documentation).
4. Make sure your [code lints](#coding-style).
5. If you haven't already, complete the [Contributor License Agreement ("CLA")](#contributor-license-agreement-cla)

&nbsp;

## Testing
TODO: Fill out the details

- **Unit tests**
TODO: Fillout the details

&nbsp;

## Updating documentation
Each API and class should be clearly documented. Well-documented code is easier to review and understand/extend. All documentation is contained in the [docs directory](docs/):


Documentation is written in [RST](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html) format.


&nbsp;

## Coding Style
TODO: Define this

&nbsp;

## Best Practices

This section captures some best practices for contributing code to torchchat. Following these will make PR reviews easier.

- **Modular Blocks instead of Monolithic Classes**. Stuffing all of the logic into a single class limits readability and makes it hard to reuse logic. Think about breaking the implementation into self-contained blocks which can be used independently from a given model. For example, attention mechanisms, embedding classes, transformer layers etc.
- **Say no to Implementation Inheritance**. You really don’t need it AND it makes the code much harder to understand or refactor since the logic is spread across many files/classes. Where needed, consider using Protocols.
- **Clean Interfaces**. There’s nothing more challenging than reading through functions/constructors with ~100 parameters. Think carefully about what needs to be exposed to the user and don’t hesitate to hard-code parameters until there is a need to make them configurable.
- **Intrusive Configs**. Config objects should not intrude into the class implementation. Configs should interact with these classes through cleanly defined builder functions which convert the config into flat parameters needed to instantiate an object.
- **Limit Generalization**. Attempting to generalize code before this is needed unnecessarily complicates implementations - you are anticipating use cases you don’t know a lot about. When you actually need to generalize a component, think about whether it’s worth it to complicate a given interface to stuff in more functionality. Don’t be afraid of code duplication if it makes things easier to read.
- **Value Checks and Asserts**. Don’t check values in higher level modules - defer the checks to the modules where the values are actually used. This helps reduce the number of raise statements in code which generally hurts readability, but are critical for correctness.

&nbsp;

## Issues
We use GitHub issues to track public bugs. Please ensure your description is clear and has sufficient instructions to be able to reproduce the issue.

Meta has a [bounty program](https://www.facebook.com/whitehat/) for the safe disclosure of security bugs. In those cases, please go through the process outlined on that page and do not file a public issue.

&nbsp;

## License
By contributing to torchchat, you agree that your contributions will be licensed under the LICENSE file in the root directory of this source tree.

&nbsp;

## Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need to do this once to work on any of Meta's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

&nbsp;
