import os
import json
import time
from collections import Counter
from flask import Blueprint, current_app, render_template, send_from_directory, url_for

history_bp = Blueprint("history", __name__)

def get_metadata_path():
    return os.path.join(current_app.root_path, "static", "uploads", "metadata.json")

def load_metadata():
    path = get_metadata_path()
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_prediction_metadata(filename, prediction):
    metadata = load_metadata()
    metadata[filename] = {
        "prediction": prediction,
        "timestamp": time.time()
    }
    path = os.path.join(current_app.root_path, "static", "uploads")
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "metadata.json"), 'w') as f:
        json.dump(metadata, f)

def get_dashboard_stats():
    metadata = load_metadata()
    if not metadata:
        return {"total": 0, "top_class": "None", "last_prediction": "None"}
    
    total = len(metadata)
    
    # Sort items by timestamp to find the latest
    sorted_items = sorted(metadata.items(), key=lambda x: x[1].get("timestamp", 0), reverse=True)
    last_prediction = sorted_items[0][1]["prediction"] if sorted_items else "None"
    
    # Find most common
    predictions = [item["prediction"] for item in metadata.values()]
    top_class = Counter(predictions).most_common(1)[0][0] if predictions else "None"
    
    return {
        "total": total,
        "top_class": top_class,
        "last_prediction": last_prediction
    }

def _get_image_files():
    uploads_dir = os.path.join(current_app.root_path, "static", "uploads")
    if not os.path.exists(uploads_dir):
        return []
    # return only common image extensions
    exts = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
    files = [f for f in os.listdir(uploads_dir)
             if os.path.isfile(os.path.join(uploads_dir, f)) and os.path.splitext(f)[1].lower() in exts]
    # sort by modified time (newest first)
    files.sort(key=lambda f: os.path.getmtime(os.path.join(uploads_dir, f)), reverse=True)
    return files


@history_bp.route("/history")
def history():
    files = _get_image_files()
    metadata = load_metadata()
    
    history_data = []
    for f in files:
        url = url_for('static', filename=f'uploads/{f}')
        meta = metadata.get(f, {})
        pred = meta.get("prediction", "Unknown")
        # format timestamp
        ts = meta.get("timestamp", os.path.getmtime(os.path.join(current_app.root_path, "static", "uploads", f)))
        date_str = time.strftime('%Y-%m-%d %H:%M', time.localtime(ts))
        
        history_data.append({
            "url": url,
            "filename": f,
            "prediction": pred,
            "date": date_str
        })
        
    return render_template('history.html', history_data=history_data)


@history_bp.route('/uploads/<path:filename>')
def uploaded_file(filename):
    uploads_dir = os.path.join(current_app.root_path, 'static', 'uploads')
    return send_from_directory(uploads_dir, filename)
