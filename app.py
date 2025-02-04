from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
from code3 import process_image, detect_colored_regions
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit

# Ensure upload folder exists with correct permissions
upload_folder = 'static/uploads'
os.makedirs(upload_folder, exist_ok=True)
os.chmod(upload_folder, 0o777)  # Donner les permissions d'écriture

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the image
        result_image, squares_part, _, _ = process_image(filepath)
        if result_image is None:
            return jsonify({'error': 'Failed to process image'})
            
        # Save result images
        result_filename = 'result_' + filename
        squares_filename = 'squares_' + filename
        cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], result_filename), result_image)
        cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], squares_filename), squares_part)
        
        # Get biomarker results
        _, colored_regions = detect_colored_regions(squares_part)
        
        results = []
        for region in colored_regions:
            if 'test_name' in region and 'interpretation' in region:
                result = region['interpretation']
                results.append({
                    'biomarker': region['test_name'],
                    'value': result.get('value', 'N/A'),
                    'level': result.get('level', 'N/A'),
                    'description': result.get('description', 'N/A')
                })
        
        return jsonify({
            'success': True,
            'result_image': '/static/uploads/' + result_filename,
            'squares_image': '/static/uploads/' + squares_filename,
            'results': results
        })
        
    return jsonify({'error': 'Invalid file type'})

if __name__ == '__main__':
    app.run(debug=True)