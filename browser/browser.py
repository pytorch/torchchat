# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import subprocess
import sys


def main(args):

    # Directory Containing the server file "chat_in_browser.py"
    server_dir = "browser"

    # Look for port from cmd args. Default to 5000 if not found.
    port = 5000
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--port":
            if i + 1 < len(sys.argv):
                # Extract the value and remove '--port' and the value from sys.argv
                port = sys.argv[i + 1]
                del sys.argv[i : i + 2]
                break
        else:
            i += 1

    # Construct arguments for the flask app minus 'browser' command
    # plus '--chat'
    args_plus_chat = ["'{}'".format(s) for s in sys.argv[1:] if s != "browser"] + [
        '"--chat"'
    ]
    formatted_args = ", ".join(args_plus_chat)
    command = [
        "flask",
        "--app",
        f"{server_dir}/chat_in_browser:create_app(" + formatted_args + ")",
        "run",
        "--port",
        f"{port}",
    ]
    subprocess.run(command)
