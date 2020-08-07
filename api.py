import flask
import srgan
from flask import send_file, request, redirect, render_template, url_for, flash
import os
from werkzeug.utils import secure_filename
from utils import *
from PIL import Image
app = flask.Flask(__name__)
app = app_configs(app)

# Heroku GCP Setup
f = open("./google-credentials-heroku.json", "w")
f.write(os.environ['GOOGLE_CONFIG'])
f.close()

@app.route('/', methods=['GET'])
def home():
    return render_template("home.html")

@app.route("/upload_image", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        if request.files:
            if "filesize" in request.cookies:
                image = request.files["image"]
                
                if image.filename == "":
                    print("No filename")
                    flash("Looks like you forgot to upload an image.")
                    return redirect(request.url)

                if allowed_image(image.filename, app):
                    filename = secure_filename(image.filename)

                    # Obtain Pixelation and Removal Amounts
                    pixelateAmount = int(request.form['pixelateAmount'])
                    max_pixel_length = 100
                    pixel_length = int(float(max_pixel_length * (101 - pixelateAmount) * 0.01))

                    y_removal = float(request.form['y_removal'])
                    x_removal = float(request.form['x_removal'])
                    removalAmount = int(request.form['removalAmount'])
                    
                    models_path = app.config["MODELS"]
                    # image.save(file_path)
                    image = Image.open(image, mode="r")

                    
                    # Removing Section from image
                    shorterSide = image.height if image.width > image.height else image.width
                    # Limiting removal square to have width 50% of shortest side length of image
                    removalWidth = shorterSide * removalAmount * 0.01 * 0.5

                    image = remove_pixels(image, removalWidth, x_removal, y_removal) if removalWidth else image 
                    
                    # Pixelating Image
                    image = scale_image(image, pixel_length)

                    srgan.forward_pass(lr_img=image, bucket_name = app.config["BUCKET_NAME"], models = models_path, filename=filename)
                    # filename = json.dumps({"filename":filename})
                    print('DONE')
                else:
                    print("That file extension is not allowed")
                    flash("Make sure you upload an image.")
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
        files = sorted(list_blobs(app.config["BUCKET_NAME_PREV"]))
        lr_imgs = files[:int(len(files)/2)]
        sr_imgs = files[int(len(files)/2):]
        past_results = zip(lr_imgs, sr_imgs)
        return render_template("past_results.html", past_results= past_results)

@app.route('/past_results')
def past_results():
    return render_template("past_results.html")

    
@app.route('/about_us')
def about_us():
    return render_template("about_us.html")

if __name__ == '__main__':
    # Local Testing
    # app.secret_key = 'many random bytes'
    # For Heroku
    app.secret_key = os.environ['FLASH_SECRET_KEY']
    app.run(port=os.getenv('PORT', 5000))