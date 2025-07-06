import zipfile
import os
from fastapi import FastAPI, UploadFile, File 
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from keras.models import load_model
from keras.utils import load_img, img_to_array
from io import BytesIO
from keras.applications.imagenet_utils import preprocess_input
from collections import Counter

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths to the model zip files
zip_paths = {
    'cnnfromscratch': 'cnnfromscratch_model.zip',
    'inceptionv3': 'inceptionV3_model.zip',
    'mobilenetv2': 'mobilenetv2_model.zip',
    'resnet50': 'resnet50_model.zip',
    'vgg19': 'VGG19_model.zip',
    'alexnet': 'alexnet_model.zip',
    'ensemble_learning': 'ensemble_learning_model.zip',
    'tomato_non_tomato': 'tomato_non_tomato_model.zip'
}

# Extract directories
extract_dirs = {key: key + '_model' for key in zip_paths}

# Unzip function
def unzipModel(zip_path, extract_dir):
    if not os.path.isdir(extract_dir):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

# Unzip models
for key in zip_paths:
    unzipModel(zip_paths[key], extract_dirs[key])

# Load models
models = {key: load_model(os.path.join(extract_dirs[key], key + '_model')) for key in zip_paths}
class_names = ['Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
               'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

@app.get('/')
def welcome():
    return {
        'success': True,
        'message': 'server of "multimodel tomato disease classification using 10 classes" is up and running successfully.'
    }

@app.post('/predict')
async def predict_disease(fileUploadedByUser: UploadFile = File(...)):
    contents = await fileUploadedByUser.read()
    imageOfUser = load_img(BytesIO(contents), target_size=(224, 224, 3))
    image_to_arr = img_to_array(imageOfUser)
    image_to_arr_preprocessed = preprocess_input(image_to_arr)
    image_to_arr_preprocessed_expand_dims = np.expand_dims(image_to_arr_preprocessed, axis=0)

    # Predict if the image is tomato or not
    is_tomato_prediction = models['tomato_non_tomato'].predict(image_to_arr_preprocessed_expand_dims)[0]
    is_tomato = np.argmax(is_tomato_prediction) == 0

    if not is_tomato:
        return {
            'success': True,
            'message': 'The uploaded image is not a tomato leaf.'
        }

    # Predictions by different models
    predictions = {}
    confidences = {}
    for model_name, model in models.items():
        if model_name != 'tomato_non_tomato':
            prediction = model.predict(image_to_arr_preprocessed_expand_dims)[0]
            predicted_class_index = np.argmax(prediction)
            predictions[model_name] = class_names[predicted_class_index]
            confidences[model_name] = np.max(prediction) * 100

    all_predicted_results = list(predictions.values())
    all_confidence_results = list(confidences.values())

    disease_that_is_occurring_for_max_time = Counter(all_predicted_results).most_common(1)[0][0]
    confidence_of_most_common_prediction_by_the_models = [confidence for prediction, confidence in zip(
        all_predicted_results, all_confidence_results) if prediction == disease_that_is_occurring_for_max_time]
    name_of_the_models_with_common_prediction = [name_of_the_model for prediction, name_of_the_model in zip(
        all_predicted_results, list(predictions.keys())) if prediction == disease_that_is_occurring_for_max_time]

    name_of_the_model_and_its_corresponding_confidence = [f"{name_of_the_model}: {confidence_of_the_model:.2f}%" for name_of_the_model, confidence_of_the_model in zip(
        name_of_the_models_with_common_prediction, confidence_of_most_common_prediction_by_the_models)]


    max_confidence_among_the_common_predicted_disease = np.max(
        confidence_of_most_common_prediction_by_the_models)

    about_the_disease = ''
    solution_of_the_disease = ''

    disease_details = {
        'Tomato___Bacterial_spot': {
            'about': 'Bacterial Spot of Tomato is a disease caused by a group of bacteria...',
            'solution': 'Unfortunately, once a plant is infected, it cannot be cured...'
        },
        'Tomato___Early_blight': {
            'about': 'Tomato Early Blight is a common disease affecting tomato plants...',
            'solution': 'To manage Early Blight, it is crucial to practice preventive measures...'
        },
        'Tomato___Late_blight': {
            'about': 'Tomato Late Blight is a disease caused by the oomycete pathogen...',
            'solution': 'There is no cure for Tomato Late Blight once a plant is infected...'
        },
        'Tomato___Leaf_Mold': {
            'about': 'Tomato Leaf Mold is a foliar disease caused by the fungus...',
            'solution': 'To manage Tomato Leaf Mold, it\'s important to use certified disease-free seed...'
        },
        'Tomato___Septoria_leaf_spot': {
            'about': 'Tomato Septoria Leaf Spot is a fungal disease caused by Septoria lycopersici...',
            'solution': 'While there is no cure for the disease once the plant is infected...'
        },
        'Tomato___Spider_mites Two-spotted_spider_mite': {
            'about': 'The Two-Spotted Spider Mite, Tetranychus urticae, is a common pest of tomato plants...',
            'solution': 'Management of spider mites includes both natural and chemical methods...'
        },
        'Tomato___Target_Spot': {
            'about': 'Tomato Target Spot is a disease caused by the fungal pathogen...',
            'solution': 'Managing Tomato Target Spot involves cultural practices and fungicide applications...'
        },
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
            'about': 'Tomato Yellow Leaf Curl Virus (TYLCV) is a DNA virus...',
            'solution': 'Unfortunately, there is no cure for plants infected with TYLCV...'
        },
        'Tomato___Tomato_mosaic_virus': {
            'about': 'Tomato Mosaic Virus (ToMV) is a plant pathogenic virus...',
            'solution': 'Unfortunately, there is no cure for plants infected with ToMV...'
        },
        'Tomato___healthy': {
            'about': 'The leaf is healthy as any kind of disease is not detected in the leaf.',
            'solution': 'No solution is required as the leaf is healthy.'
        }
    }

    if disease_that_is_occurring_for_max_time in disease_details:
        about_the_disease = disease_details[disease_that_is_occurring_for_max_time]['about']
        solution_of_the_disease = disease_details[disease_that_is_occurring_for_max_time]['solution']

    return {
        'success': True,
        'is_tomato': is_tomato,
        'prediction_results_of_all_model': [
            {
                'name_of_the_model': model_name,
                'predicted_result': predictions[model_name],
                'confidence': f'{confidences[model_name]:.2f}%',
                'message': f'Status of the leaf: {predictions[model_name]} with a confidence of {confidences[model_name]:.2f}%'
            } for model_name in predictions
        ],
        'final_predicted_result_of_the_leaf': {
            'predicted_disease': disease_that_is_occurring_for_max_time,
            'max_confidence_among_the_common_predicted_disease': max_confidence_among_the_common_predicted_disease,
            'about_the_disease': about_the_disease,
            'solution_of_the_disease': solution_of_the_disease,
            'name_of_the_model_and_its_corresponding_confidence': name_of_the_model_and_its_corresponding_confidence
        }
    }
