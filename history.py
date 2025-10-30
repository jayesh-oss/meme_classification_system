import os
from flask import Blueprint, current_app, render_template, send_from_directory, url_for

history_bp = Blueprint("history", __name__)


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
    # build URLs for each file using static route
    file_urls = [url_for('static', filename=f'uploads/{fname}') for fname in files]
    return render_template('history.html', files=file_urls)


@history_bp.route('/uploads/<path:filename>')
def uploaded_file(filename):
    # optional: serve files directly from static/uploads if needed
    uploads_dir = os.path.join(current_app.root_path, 'static', 'uploads')
    return send_from_directory(uploads_dir, filename)
