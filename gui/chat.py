# -*- coding: UTF-8 -*-
"""
hello_jinja2: Get start with Jinja2 templates
"""
from flask import Flask, render_template, request
app = Flask(__name__)

convo = ""

@app.route('/')
def main():
    return render_template('chat.html', convo ="sooooo empty chat")

@app.route('/chat', methods=['POST'])
def chat():
    # Retrieve the HTTP POST request parameter value from 'request.form' dictionary
    _prompt = request.form.get('prompt')  # get(attr) returns None if attr is not present

    global convo
    
    if _prompt:
        convo = convo + "<p>and then you said: <br>'" + _prompt + "'\n</p>"
        convo = convo + "<p>and I said not much of import...\n</p>"

    return render_template('chat.html', convo=convo)

if __name__ == '__main__':
    app.run(debug=True)
