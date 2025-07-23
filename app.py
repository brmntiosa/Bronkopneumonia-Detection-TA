import os
import joblib
import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from PIL import Image  # Ganti dari skimage.io
from skimage.feature import graycomatrix, graycoprops, hog

# === CONFIG ===
UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = joblib.load("model_glcm_hog_pca_final.pkl")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_glcm_hog_features(image_path):
    # Pakai Pillow agar tidak error dengan gambar HP
    image = Image.open(image_path).convert('L')  # Grayscale
    image = image.resize((128, 128))
    image_np = np.array(image)

    glcm = graycomatrix(image_np, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    glcm_props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    glcm_features = [graycoprops(glcm, prop)[0, 0] for prop in glcm_props]

    hog_features = hog(image_np,
                       orientations=9,
                       pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2),
                       visualize=False,
                       block_norm='L2-Hys')

    combined_features = np.hstack([glcm_features, hog_features])
    return combined_features.reshape(1, -1)

@app.route('/')
def beranda():
    return render_template('index.html')

@app.route('/about.html')
def tentang():
    return render_template('about.html')

@app.route('/health.html', methods=['GET', 'POST'])
def kesehatan():
    if request.method == 'POST':
        if 'image' not in request.files:
            return "No file uploaded", 400
        file = request.files['image']
        if file.filename == '':
            return "No selected file", 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(filepath)

            processed_img = extract_glcm_hog_features(filepath)
            if processed_img.shape[1] != 8106:
                return f"Jumlah fitur salah: {processed_img.shape[1]}", 500

            prediction = model.predict(processed_img)[0]
            return render_template('health.html', prediction=prediction, image_path=filepath)

    return render_template('health.html', prediction=None)

@app.route('/contact.html')
def contact():
    return render_template('contact.html')

@app.route('/artikel/<int:artikel_id>')
def artikel_detail(artikel_id):
    artikel = next((a for a in artikel_data if a['id'] == artikel_id), None)
    if artikel is None:
        return "Artikel tidak ditemukan", 404
    return render_template('artikel_detail.html', artikel=artikel)

# === DUMMY ARTIKEL ===
artikel_data = [
    {
        "id": 1,
        "judul": "Mengenal Bronkopneumonia: Gejala dan Pencegahan",
        "gambar": "img/artikel1.jpg",
        "deskripsi": "Bronkopneumonia adalah infeksi paru yang berbahaya. Kenali gejala awalnya di sini.",
        "isi": "Bronkopneumonia menyerang bronkiolus dan alveoli..."
    },
    {
        "id": 2,
        "judul": "Perbedaan Pneumonia dan Bronkopneumonia",
        "gambar": "img/artikel2.jpg",
        "deskripsi": "Pneumonia dan bronkopneumonia sering tertukar. Ini penjelasannya.",
        "isi": "Pneumonia menyebar, bronkopneumonia patchy..."
    }
]

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Railway akan set PORT otomatis
    app.run(host='0.0.0.0', port=port)
