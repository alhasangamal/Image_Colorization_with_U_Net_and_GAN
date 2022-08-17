import io
from PIL import Image
from Image_coloring import app
from flask import render_template, request
from Image_coloring import model
from Image_coloring.processing import predict
import matplotlib.pyplot as plt


@app.route("/", methods=["GET", "POST"])
def home_page():
    if request.method == 'POST':
        print("hey")
        f = request.files['file']
        img = Image.open(io.BytesIO(f.read()))
        img.save("./Image_coloring/static/img/input.jpg")
        pred_img = predict(model, img)
        plt.imshow(pred_img)
        plt.imsave("./Image_coloring/static/img/output.jpg", pred_img)
    return render_template("index.html")
