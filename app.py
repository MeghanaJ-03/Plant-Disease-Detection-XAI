import os
import numpy as np
import h5py
import cv2
import matplotlib.cm as cm
from flask import Flask, request, jsonify, render_template, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from werkzeug.utils import secure_filename
import tensorflow as tf
from datetime import datetime

app = Flask(__name__)

# --- CONFIGURATION ---
MODEL_PATH = 'plant_disease_model.h5'
TARGET_SIZE = (224, 224) 
scan_history = []
# Create directories for saving images
BASE_DIR = os.path.dirname(__file__)
UPLOADS_DIR = os.path.join(BASE_DIR, 'static', 'uploads')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'static', 'outputs')
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


# --- 1. THE EXACT 38 CLASSES ---
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# --- 2. DISEASE DATABASE ---
DISEASE_INFO = {
    'Apple___Apple_scab': {'treatment': 'Apply organic copper soap or sulfur fungicide.', 'prevention': 'Rake and destroy fallen leaves in autumn. Prune for airflow.'},
    'Apple___Black_rot': {'treatment': 'Prune out dead or diseased wood. Apply Captan fungicide.', 'prevention': 'Remove mummified fruit and dead wood. Ensure rapid drying of foliage.'},
    'Apple___Cedar_apple_rust': {'treatment': 'Apply preventative fungicide sprays in early spring.', 'prevention': 'Remove nearby cedar/juniper hosts if possible. Choose rust-resistant varieties.'},
    'Cherry_(including_sour)___Powdery_mildew': {'treatment': 'Spray with Neem oil or potassium bicarbonate.', 'prevention': 'Avoid overhead watering. Prune to increase sunlight penetration.'},
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {'treatment': 'Apply foliar fungicide if disease is severe prior to tasseling.', 'prevention': 'Practice crop rotation and use resistant corn hybrids.'},
    'Corn_(maize)___Common_rust_': {'treatment': 'Apply early-season fungicide if rust pustules are spreading rapidly.', 'prevention': 'Plant rust-resistant hybrids. Avoid late planting.'},
    'Corn_(maize)___Northern_Leaf_Blight': {'treatment': 'Apply foliar fungicides during the tasseling stage.', 'prevention': 'Plow under corn debris after harvest. Rotate crops annually.'},
    'Grape___Black_rot': {'treatment': 'Apply Mancozeb or Myclobutanil fungicides early in the season.', 'prevention': 'Remove all mummified grapes from vines and ground. Prune aggressively.'},
    'Grape___Esca_(Black_Measles)': {'treatment': 'There is no chemical cure. Infected vines must be heavily pruned or removed.', 'prevention': 'Avoid large pruning wounds. Disinfect tools between cuts.'},
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {'treatment': 'Apply Bordeaux mixture or copper fungicides.', 'prevention': 'Ensure good canopy management for airflow and sunlight exposure.'},
    'Orange___Haunglongbing_(Citrus_greening)': {'treatment': 'No cure exists. Infected trees must be removed and destroyed to save others.', 'prevention': 'Control Asian citrus psyllid populations with horticultural oils.'},
    'Peach___Bacterial_spot': {'treatment': 'Spray with copper bactericides early in the growing season.', 'prevention': 'Plant resistant peach varieties. Maintain optimal tree nutrition.'},
    'Pepper,_bell___Bacterial_spot': {'treatment': 'Apply copper-based fungicides/bactericides every 7-10 days.', 'prevention': 'Use certified disease-free seeds. Avoid working in the garden when plants are wet.'},
    'Potato___Early_blight': {'treatment': 'Apply Chlorothalonil or Copper fungicides as soon as spots appear.', 'prevention': 'Rotate crops. Do not compost infected potato vines.'},
    'Potato___Late_blight': {'treatment': 'Destroy infected plants immediately. Apply protective fungicides to nearby healthy plants.', 'prevention': 'Plant certified seed potatoes. Ensure good soil drainage.'},
    'Squash___Powdery_mildew': {'treatment': 'Apply sulfur, Neem oil, or potassium bicarbonate sprays.', 'prevention': 'Space plants widely for airflow. Plant in full sun.'},
    'Strawberry___Leaf_scorch': {'treatment': 'Remove severely infected leaves. Apply copper-based fungicides.', 'prevention': 'Avoid overhead watering. Renew strawberry patches every 3-4 years.'},
    'Tomato___Bacterial_spot': {'treatment': 'Apply fixed copper sprays combined with Mancozeb.', 'prevention': 'Avoid overhead watering. Rotate crops away from peppers and tomatoes.'},
    'Tomato___Early_blight': {'treatment': 'Prune lower infected leaves. Apply copper fungicide or Bacillus subtilis.', 'prevention': 'Mulch around the base to prevent soil splashing. Stake plants well.'},
    'Tomato___Late_blight': {'treatment': 'Immediately bag and dispose of infected plants. Do not compost them.', 'prevention': 'Keep foliage dry. Space plants out to ensure maximum air circulation.'},
    'Tomato___Leaf_Mold': {'treatment': 'Apply Chlorothalonil or copper-based fungicides.', 'prevention': 'Increase ventilation in greenhouses. Prune lower leaves to improve airflow.'},
    'Tomato___Septoria_leaf_spot': {'treatment': 'Remove infected leaves. Spray with organic copper fungicides.', 'prevention': 'Use a thick layer of mulch. Water at the base of the plant only.'},
    'Tomato___Spider_mites Two-spotted_spider_mite': {'treatment': 'Spray undersides of leaves with insecticidal soap or Neem oil.', 'prevention': 'Keep plants well-watered (mites thrive in dusty, dry conditions).'},
    'Tomato___Target_Spot': {'treatment': 'Apply appropriate fungicides (like Chlorothalonil) when symptoms first appear.', 'prevention': 'Ensure excellent airflow. Avoid excessive nitrogen fertilization.'},
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {'treatment': 'No cure. Remove and destroy infected plants immediately.', 'prevention': 'Control whitefly populations using reflective mulches or insecticidal soaps.'},
    'Tomato___Tomato_mosaic_virus': {'treatment': 'No cure. Uproot and burn or discard infected plants.', 'prevention': 'Wash hands thoroughly with soap before handling plants. Disinfect gardening tools.'}
}

def get_treatment_plan(predicted_class):
    if 'healthy' in predicted_class.lower():
        return {
            'treatment': 'No treatment required. Your plant is thriving!',
            'prevention': 'Continue your current watering, sunlight, and fertilizer routines.'
        }
    return DISEASE_INFO.get(predicted_class, {
        'treatment': 'Apply a general-purpose organic fungicide/bactericide.',
        'prevention': 'Ensure good airflow, avoid overhead watering, and monitor closely.'
    })

# --- FIX FOR TEACHABLE MACHINE / MOBILENET MODELS ---
f = h5py.File(MODEL_PATH, mode="r+")
model_config_string = f.attrs.get("model_config")
if model_config_string.find('"groups": 1,') != -1:
    model_config_string = model_config_string.replace('"groups": 1,', '')
    f.attrs.modify('model_config', model_config_string)
    f.flush()
f.close()

model = load_model(MODEL_PATH, compile=False)

def get_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        try:
            # Safely check if the layer's output has 4 dimensions (Batch, Height, Width, Channels)
            if len(layer.output.shape) == 4:
                return layer.name
        except Exception:
            # If the layer doesn't have a standard shape attribute, just skip it without crashing
            continue
            
    raise ValueError("Could not find a convolutional layer.")

last_conv_layer_name = get_last_conv_layer_name(model)

def generate_gradcam(img_path, model, last_conv_layer_name, pred_index=None):
    img = image.load_img(img_path, target_size=TARGET_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # FIX 1: model.inputs is already a list, so we remove the extra brackets
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        outputs = grad_model(img_array)
        last_conv_layer_output = outputs[0]
        preds = outputs[1]
        
        # FIX 2: If the model wraps outputs in an extra list, we extract the tensor!
        if isinstance(preds, list):
            preds = preds[0]
        if isinstance(last_conv_layer_output, list):
            last_conv_layer_output = last_conv_layer_output[0]

        if pred_index is None:
            pred_index = tf.argmax(preds[0])
            
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Prevent dividing by zero just in case the heatmap is perfectly flat
    max_heat = tf.math.reduce_max(heatmap)
    if max_heat == 0:
        max_heat = 1e-10
        
    heatmap = tf.maximum(heatmap, 0) / max_heat
    heatmap = heatmap.numpy()

    # Overlay heatmap on original image
    img = cv2.imread(img_path)
    img = cv2.resize(img, TARGET_SIZE)
    
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    superimposed_img = heatmap * 0.4 + img
    
    output_filename = 'gradcam_' + os.path.basename(img_path)
    output_path = os.path.join(OUTPUTS_DIR, output_filename)
    cv2.imwrite(output_path, superimposed_img)
    
    return output_filename

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=TARGET_SIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0  
    
    preds = model.predict(x)
    predicted_index = np.argmax(preds)
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(np.max(preds)) * 100 
    
    return predicted_class, round(confidence, 2), predicted_index


# --- ROUTES ---
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/history_page', methods=['GET'])
def history_page():
    return render_template('history.html')

@app.route('/analysis_page', methods=['GET'])
def analysis_page():
    return render_template('analysis.html')

@app.route('/history', methods=['GET'])
def get_history():
    return jsonify(scan_history)

@app.route('/latest_analysis', methods=['GET'])
def get_latest_analysis():
    if not scan_history:
        return jsonify({'error': 'No recent scans found.'})
    # Return the most recent scan details
    return jsonify(scan_history[0])

@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
        
    f = request.files['file']
    if f.filename == '':
        return jsonify({'error': 'No selected file'})
        
    filename = secure_filename(f.filename)
    file_path = os.path.join(UPLOADS_DIR, filename)
    f.save(file_path)

    try:
        predicted_class, confidence, pred_index = model_predict(file_path, model)
        info = get_treatment_plan(predicted_class)
        
        # Generate Grad-CAM Heatmap
        gradcam_filename = generate_gradcam(file_path, model, last_conv_layer_name, pred_index)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        scan_record = {
            'disease': predicted_class,
            'confidence': confidence,
            'time': timestamp,
            'original_image': f'/static/uploads/{filename}',
            'gradcam_image': f'/static/outputs/{gradcam_filename}',
            'treatment': info.get('treatment'),
            'prevention': info.get('prevention')
        }
        
        scan_history.insert(0, scan_record)
        if len(scan_history) > 5:
            scan_history.pop()

        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence,
            'treatment': info.get('treatment'),
            'prevention': info.get('prevention')
        })
    except Exception as e:
        print("\n" + "="*40)
        print("🔥 CRITICAL ERROR:", str(e))
        print("="*40 + "\n")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)