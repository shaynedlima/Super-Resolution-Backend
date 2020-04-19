import flask
import srgan
from flask import send_file, request, redirect, render_template, url_for
import os
from werkzeug.utils import secure_filename
from utils import *
from PIL import Image

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
                # Deleting contents from the temp results folder
                results_path = app.config["IMAGE_RESULTS"]
                filelist = [ f for f in os.listdir(results_path)]
                for f in filelist:
                    os.remove(os.path.join(results_path, f))

                image = request.files["image"]
                
                if image.filename == "":
                    print("No filename")
                    return redirect(request.url)

                if allowed_image(image.filename, app):
                    filename = secure_filename(image.filename)
                    file_path = os.path.join(app.config["IMAGE_UPLOADS"], filename)
                    # results_path = app.config["IMAGE_RESULTS"]
                    models_path = app.config["MODELS"]
                    # image.save(file_path)
                    image = Image.open(image, mode="r")
                    srgan.forward_pass(img=image, results = results_path, models = models_path, filename=filename)
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
        return render_template("display_results.html", lr_img = lr_result, sr_img = sr_result)
    else:
        print('Show past results')
        past_results_path = app.config["IMAGE_RESULTS"] + '\past_results'
        files = sorted(os.listdir(past_results_path))
        lr_imgs = files[:int(len(files)/2)]
        sr_imgs = files[int(len(files)/2):]
        past_results = zip(lr_imgs, sr_imgs)
        return render_template("past_results.html", past_results= past_results)

@app.route('/past_results')
def past_results():
    return render_template("past_results.html")

if __name__ == '__main__':
    app.run(port=os.getenv('PORT', 5000))