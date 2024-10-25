# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3
# Delete me after triton is updated past https://github.com/triton-lang/triton/pull/3564


from pathlib import Path

import triton


def patch_def_search_in_jit_py(jit_py: Path) -> None:
    with jit_py.open() as f:
        lines = f.readlines()
    old_match = 'self.src = self.src[self.src.find("def"):]'
    new_match = 'self.src = self.src[re.search(r"^def\s+\w+\s*\(", self.src, re.MULTILINE).start():]'
    lines.insert(4, "import re\n")
    for idx, line in enumerate(lines):
        if old_match in line:
            lines[idx] = line.replace(old_match, new_match)
    jit_py.write_text("".join(lines))


jit_py = Path(triton.__file__).parent / "runtime" / "jit.py"
patch_def_search_in_jit_py(jit_py)
