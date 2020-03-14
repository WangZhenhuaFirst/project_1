from flask import Flask
from flask import render_template
from flask import request
from flask_bootstrap import Bootstrap
import Summary_func
import os
from datetime import timedelta

app = Flask(__name__)
app.config['DEBUG'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)
app.send_file_max_age_default = timedelta(seconds=1)
bootstrap = Bootstrap(app)


@app.route('/')
def index():
    print(os.path.abspath(os.getcwd()))
    return render_template('index.html')


@app.route('/abstract', methods=['POST'])
def abstract():
    title = request.form['title']
    content = request.form['content']
    summary, img_path = Summary_func.summary_func(title, content)
    return render_template('result.html', title=title, content=content, summary=summary, img_path=img_path)


if __name__ == '__main__':
    app.run()
