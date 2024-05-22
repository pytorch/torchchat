# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import subprocess
import os

from flask import Flask, render_template, request

convo = ""
disable_input = False

# Flask html Template Path
#template_dir = os.path.abspath('browser/templates')

def create_app(*args):
    app = Flask(__name__)
    # app = Flask(__name__, template_folder=template_dir)

    # create a new process and set up pipes for communication
    proc = subprocess.Popen(
        ["python3", "generate.py", *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE
    )

    @app.route("/")
    def main():
        print("Starting chat session.")
        line = b""
        output = ""
        global disable_input

        while True:
            buffer = proc.stdout.read(1)
            line += buffer
            try:
                decoded = line.decode("utf-8")
            except:
                continue

            if decoded.endswith("Do you want to enter a system prompt? Enter y for yes and anything else for no. \n"):
                print(f"| {decoded}")
                proc.stdin.write("\n".encode("utf-8"))
                proc.stdin.flush()
                line = b""
            elif line.decode("utf-8").startswith("User: "):
                print(f"| {decoded}")
                break

            if decoded.endswith("\r") or decoded.endswith("\n"):
                decoded = decoded.strip()
                print(f"| {decoded}")
                output += decoded + "\n"
                line = b""

        return render_template(
            "chat.html",
            convo="Hello! What is your prompt?",
            disable_input=disable_input,
        )

    @app.route("/chat", methods=["GET", "POST"])
    def chat():
        # Retrieve the HTTP POST request parameter value from
        # 'request.form' dictionary
        _prompt = request.form.get("prompt", "")
        proc.stdin.write((_prompt + "\n").encode("utf-8"))
        proc.stdin.flush()

        print(f"User: {_prompt}")

        line = b""
        output = ""
        global disable_input

        while True:
            buffer = proc.stdout.read(1)
            line += buffer
            try:
                decoded = line.decode("utf-8")
            except:
                continue

            if decoded.startswith("User: "):
                break
            if decoded.startswith("=========="):
                disable_input = True
                break
            if decoded.endswith("\r") or decoded.endswith("\n"):
                decoded = decoded.strip()
                print(f"| {decoded}")
                output += decoded + "\n"
                line = b""

        # Strip "Model: " from output
        model_prefix = "Model: "
        if output.startswith(model_prefix):
            output = output[len(model_prefix) :]
        else:
            print("But output is", output)

        global convo

        if _prompt:
            convo += "<H1>User</H1>\n<p> " + _prompt + " </p>\n\n"
            convo += "<H1>Model</H1>\n<p> " + output + " </p>\n\n"

        return render_template("chat.html", convo=convo, disable_input=disable_input)

    return app
