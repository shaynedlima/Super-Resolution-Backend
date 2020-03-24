import flask
import srgan
from flask import send_file, request, redirect, render_template, url_for
import os
from werkzeug.utils import secure_filename
from utils import *

app = flask.Flask(__name__)
app = app_configs(app)

image_path = "./images/incomplete.jpg"

@app.route('/', methods=['GET'])
def home():
    return render_template("home.html")

@app.route("/upload_image", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        if request.files:
            if "filesize" in request.cookies:
                # if not allowed_image_filesize(request.cookies["filesize"], app):
                #     print("Filesize exceeded maximum limit")
                #     return redirect(request.url)

                image = request.files["image"]

                if image.filename == "":
                    print("No filename")
                    return redirect(request.url)

                if allowed_image(image.filename, app):
                    filename = secure_filename(image.filename)
                    file_path = os.path.join(app.config["IMAGE_UPLOADS"], filename)
                    results_path = app.config["IMAGE_RESULTS"]
                    models_path = app.config["MODELS"]
                    image.save(file_path)
                    print("Image saved")
                    print(file_path)
                    image = srgan.forward_pass(img=file_path, results = results_path, models = models_path, filename=filename)
                    # filename = json.dumps({"filename":filename})
                    print('DONE')
                else:
                    print("That file extension is not allowed")
                    return redirect(request.url)
        return redirect(url_for('display_results', filename=filename))
                    
    return render_template("upload_image.html")

@app.route('/display_results')
def display_results():
    if('filename' in request.args):
        filename = request.args['filename']
        lr_result = f"lr_{filename}"
        sr_result = f"sr_{filename}"
        print(lr_result)
        return render_template("display_results.html", lr_img = lr_result, sr_img = sr_result)
    else:
        print('Show past results')
        return render_template("past_results.html")

@app.route('/past_results')
def past_results():
    return render_template("past_results.html")

app.run()