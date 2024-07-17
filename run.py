from flask import Flask, jsonify, request, render_template,send_from_directory
from werkzeug.utils import secure_filename
from flask_cors import CORS
from attendance import perform_inference, perform_training
import requests
import os

app = Flask(__name__)
CORS(app)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_attendance(image_link):
    perform_training()
    image = requests.get(image_link)
    students = perform_inference(image.content)
    return render_template('index.html', students=students)

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            fp = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(fp)
            filepath = f"http://localhost:3000/getimage/{filename}"
            return get_attendance(filepath)
    return "UPLOAD"

@app.route('/getimage/<fn>')
def getimage(fn):
    return send_from_directory('static/uploads', fn)

if __name__ == "__main__":
    app.run(host="localhost", port=3000, debug=True)
