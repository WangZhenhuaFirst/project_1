from flask import Flask
from flask import render_template
from flask import request
from flask_bootstrap import Bootstrap
import Summary_func
import os

app = Flask(__name__)
bootstrap = Bootstrap(app)


@app.route('/')
def index():
    print(os.path.abspath(os.getcwd()))
    return render_template('index.html')


@app.route('/abstract', methods=['POST'])
def abstract():
    title = request.form['title']
    content = request.form['content']
    summary = Summary_func.summary_func(title, content)
    return render_template('result.html', title=title, content=content, summary=summary)


if __name__ == '__main__':
    app.run()
