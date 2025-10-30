import os
from flask import Flask, render_template, request
from history import history_bp
from utils.inference import classify_meme

# Flask setup
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"

# Home page (upload form)
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template('about.html')

# Classify meme (handle file upload + prediction)
@app.route("/classify", methods=["POST"])
def classify():
    if "file" not in request.files:
        return "No file uploaded!", 400

    file = request.files["file"]
    if file.filename == "":
        return "No file selected!", 400

    # Save file
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    file.save(filepath)

    # Run classification
    predicted_class = classify_meme(filepath)

    # Render results
    return render_template("result.html",
                           prediction=predicted_class,
                           filename=file.filename)
    



if __name__ == "__main__":
    # register history blueprint
    app.register_blueprint(history_bp)
    app.run(debug=True)
