# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import subprocess

from flask import Flask, render_template, request

convo = ""
disable_input = False


def create_app(*args):
    app = Flask(__name__)

    # create a new process and set up pipes for communication
    proc = subprocess.Popen(
        ["python3", "generate.py", *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE
    )

    @app.route("/")
    def main():
        output = ""
        global disable_input

        while True:
            line = proc.stdout.readline()
            if line.decode("utf-8").startswith("What is your prompt?"):
                break
            output += line.decode("utf-8").strip() + "\n"
        return render_template(
            "chat.html",
            convo="Hello! What is your prompt?",
            disable_input=disable_input,
        )

    @app.route("/chat", methods=["GET", "POST"])
    def chat():
        # Retrieve the HTTP POST request parameter value from 'request.form' dictionary
        _prompt = request.form.get("prompt", "")
        proc.stdin.write((_prompt + "\n").encode("utf-8"))
        proc.stdin.flush()

        output = ""
        global disable_input

        while True:
            line = proc.stdout.readline()
            if line.decode("utf-8").startswith("What is your prompt?"):
                break
            if line.decode("utf-8").startswith("=========="):
                disable_input = True
                break
            output += line.decode("utf-8").strip() + "\n"

        global convo

        if _prompt:
            convo += "Your prompt:\n" + _prompt + "\n\n"
            convo += "My response:\n" + output + "\n\n"

        return render_template("chat.html", convo=convo, disable_input=disable_input)

    return app
