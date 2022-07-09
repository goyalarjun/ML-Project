from flask import Flask, url_for, render_template, request
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)


# app.config['static']

#@app.route('/')
#def welcome():
#    return render_template('index.html')


# @app.route('/output')
# def output():
#    img = url_for('static', filename="classification.png")
#    # full_filename = r"D:\Term3\Machine Learning\classification.png"
#    return render_template("output.html", user_image=img)


@app.route('/', methods=['GET', 'POST'])
def upload():
    return render_template('upload_file.html')


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        select = request.form.get('dropdown')
        f = request.files['file']
        f.save(secure_filename(f.filename))
        print(f.filename)
        if select == 'classification' and f.filename == 'adult.data':
            os.system('python ArjunMLAssignment-2.py')
            img = url_for('static', filename="classification.png")
            return render_template("output.html", user_image=img)

        elif select == 'regression' and f.filename == 'housing.csv':
            os.system('python ArjunMLAssignment-1.py')
            img = url_for('static', filename="regression.png")
            return render_template("output.html", user_image=img)

        else:
            return 'Please select type of analysis as regression with housing.csv or classification with adult.data!!'


app.run()
