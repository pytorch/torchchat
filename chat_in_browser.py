# -*- coding: UTF-8 -*-
from flask import Flask, render_template, request
from cli import add_arguments_for_generate, arg_init, check_args
from generate import main as generate_main
import subprocess
import sys


convo = ""

def create_app(*args):
    app = Flask(__name__)

    import subprocess
    # create a new process and set up pipes for communication
    proc = subprocess.Popen(["python", "generate.py", *args],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE)

    @app.route('/')
    def main():
        output = ""
        while True:
            line = proc.stdout.readline()
            if line.decode('utf-8').startswith("What is your prompt?"):
                break
            output += line.decode('utf-8').strip() + "\n"
        return render_template('chat.html', convo="Hello! What is your prompt?")

    @app.route('/chat', methods=['POST'])
    def chat():
        # Retrieve the HTTP POST request parameter value from 'request.form' dictionary
        _prompt = request.form.get('prompt')
        proc.stdin.write((_prompt + "\n").encode('utf-8'))
        proc.stdin.flush()

        output = ""
        while True:
            line = proc.stdout.readline()
            if line.decode('utf-8').startswith("What is your prompt?"):
                break
            output += line.decode('utf-8').strip() + "\n"

        global convo

        if _prompt:
            convo += "Your prompt:\n" + _prompt + "\n\n"
            convo += "My response:\n" + output + "\n\n"

        return render_template('chat.html', convo=convo)

    return app
