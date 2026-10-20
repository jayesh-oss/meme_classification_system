import os
import json
from flask import Flask, render_template, request
from history import history_bp, get_dashboard_stats, save_prediction_metadata
from utils.inference import classify_meme

# Flask setup
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Home page (upload form)
@app.route("/", methods=["GET"])
def index():
    stats = get_dashboard_stats()
    return render_template("index.html", 
                           total_memes=stats["total"],
                           top_class=stats["top_class"],
                           prediction=stats["last_prediction"])

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
    file.save(filepath)

    # Run classification
    result_data = classify_meme(filepath)
    
    # Save metadata for history
    save_prediction_metadata(file.filename, result_data["prediction"])

    # Render results
    return render_template("result.html",
                           prediction=result_data["prediction"],
                           scores=result_data["scores"],
                           extracted_text=result_data["extracted_text"],
                           filename=file.filename)
    

if __name__ == "__main__":
    # register history blueprint
    app.register_blueprint(history_bp)
    app.run(debug=True)
