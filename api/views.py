from django.shortcuts import render
from django.conf import settings
from django.http import JsonResponse
from keras.models import load_model
import keras.utils as image
import os
from django.core.files.storage import FileSystemStorage
import numpy as np
import pickle
from sklearn import preprocessing
import json
from django.http import JsonResponse
import pandas as pd


pest_img_height, pest_img_width = 180, 180
crop_des_height, crop_des_width = 40, 40

model_dir = settings.MODELS_ROOT
pest_detection_model_path = os.path.join(
    model_dir, str('pest_detection_model.h5'))
crop_disease_model_path = os.path.join(
    model_dir, str('crop_disease_model.h5')
)

file_path_crop1 = os.path.join(
    model_dir, str('crops_recomendation_model1.pickle'))
file_path_crop2 = os.path.join(
    model_dir, str('crops_recomendation_model2.pickle'))
file_path_fertilizer = os.path.join(
    model_dir, str('fertilizer_recomendation_model.pickle'))

pest_model = load_model(pest_detection_model_path)
crop_disease_model = load_model(crop_disease_model_path)
pest_class_names = ['aphids', 'armyworm', 'beetle', 'bollworm',
                    'grasshopper', 'mites', 'mosquito', 'sawfly', 'stem_borer']
crop_disease_class_names = ['Corn_(maize)___healthy',
                            'Corn___Cercospora_leaf-spot Gray_Leaf_Spot',
                            'Corn___Common_Rust',
                            'Corn___Leaf_Blight',
                            'Cotton__bacterial_blight',
                            'Cotton__curl_virus',
                            'Cotton__fussarium_wilt',
                            'Cotton__healthy',
                            'Rice__Bacterial leaf blight',
                            'Rice__Brown spot',
                            'Rice__Leaf smut',
                            'Rice___Healthy',
                            'Rice___Hispa',
                            'Rice___Leaf_Blast',
                            'Sugarcane__Bacterial Blight',
                            'Sugarcane__Healthy',
                            'Sugarcane__RedRot',
                            'Sugarcane__RedRust',
                            'Wheat__Healthy',
                            'Wheat___Brown_Rust',
                            'Wheat___Yellow_Rust',
                            'Wheat__septoria',
                            'Wheat__stripe_rust']


def predict_image(img):
    prediction = pest_model.predict(img)[0]
    pred = {pest_class_names[i]: float(prediction[i]) for i in range(9)}
    max_value = max(pred, key=pred.get)
    return max_value


def pest_model_prediction(request):
    fileObj = request.FILES['upload']
    fs = FileSystemStorage()
    filePathName = fs.save(fileObj.name, fileObj)
    deletePathName = filePathName
    filePathName = fs.url(filePathName)
    pred_img = '.'+filePathName
    img = image.load_img(pred_img, target_size=(
        pest_img_height, pest_img_width))
    x = image.img_to_array(img)
    x = x/255
    x = x.reshape(-1, pest_img_height, pest_img_width, 3)
    result = predict_image(x)
    fs.delete(deletePathName)
    return JsonResponse({"pred": result})


def crop_predict_image(img):
    prediction = crop_disease_model.predict(img)[0]
    pred = {crop_disease_class_names[i]: float(
        prediction[i]) for i in range(23)}
    max_value = max(pred, key=pred.get)
    return max_value


def crop_model_pred(request):
    fileObj = request.FILES['upload']
    fs = FileSystemStorage()
    filePathName = fs.save(fileObj.name, fileObj)
    deletePathName = filePathName

    filePathName = fs.url(filePathName)
    crop_disease_img = '.'+filePathName
    img = image.load_img(crop_disease_img, target_size=(
        crop_des_height, crop_des_width))
    x = image.img_to_array(img)
    x = x/255
    x = x.reshape(-1, crop_des_height, crop_des_width, 3)
    result = crop_predict_image(x)
    fs.delete(deletePathName)
    return JsonResponse({"pred": result})


crop1_pickle_in = open(
    file_path_crop1, 'rb')
crop2_pickle_in = open(
    file_path_crop2, 'rb')
fertilizer_pickle_in = open(
    file_path_fertilizer, 'rb')


crops_recomendation_model1 = pickle.load(crop1_pickle_in)
crops_recomendation_model2 = pickle.load(crop2_pickle_in)
fertilizer_recomendation_model = pickle.load(fertilizer_pickle_in)


crops_with_soil_df1 = pd.read_csv(
    "https://raw.githubusercontent.com/Nazif-Malhi/Farmstead_Models/main/ML%20Models%20Farmstead/Dataset/Crops/Crop_with_soil%20(i).csv")
crops_with_soil_df2 = pd.read_csv(
    "https://raw.githubusercontent.com/Nazif-Malhi/Farmstead_Models/main/ML%20Models%20Farmstead/Dataset/Crops/Crop_with_soil%20(ii).csv")
fertilizer_df = pd.read_csv(
    "https://raw.githubusercontent.com/Nazif-Malhi/Farmstead_Models/main/ML%20Models%20Farmstead/Dataset/Fertilizer/Fertilizer%20Prediction.csv")


le1 = preprocessing.LabelEncoder()
crops_with_soil_df1['soil'] = le1.fit_transform(crops_with_soil_df1['soil'])

le2 = preprocessing.LabelEncoder()
crops_with_soil_df2['soil'] = le2.fit_transform(crops_with_soil_df2['soil'])

le_soil = preprocessing.LabelEncoder()
le_crop = preprocessing.LabelEncoder()
fertilizer_df['Soil Type'] = le_soil.fit_transform(fertilizer_df['Soil Type'])
fertilizer_df['Crop Type'] = le_crop.fit_transform(fertilizer_df['Crop Type'])


def crop_simple_recomendation_prediction(request):
    data = json.loads(request.body)
    convertedLabel = le1.transform([data['soil_type']])
    prepare_data = np.array(
        [[convertedLabel, float(data['temp']), float(data['humi']), float(data['ph']), float(data['rain'])]])
    prediction_crop1 = crops_recomendation_model1.predict(prepare_data)
    return JsonResponse({"prediction_simple": prediction_crop1[0]})


def crop_advance_recomendation_prediction(request):
    data = json.loads(request.body)
    convertedLabel = le2.transform([data['soil_type']])
    prepare_data = np.array(
        [[float(data['nitrogen']), float(data['phosphorus']), float(data['potassium']), convertedLabel, float(data['temp']), float(data['humi']), float(data['ph']), float(data['rain'])]])
    prediction_crop2 = crops_recomendation_model2.predict(prepare_data)
    return JsonResponse({"prediction_advance": prediction_crop2[0]})


def fertilizer_recomendation_prediction(request):
    data = json.loads(request.body)
    convertedLabelSoil = le_soil.transform([data['soil']])
    convertedLabelCrop = le_crop.transform([data['crop']])
    data = np.array([[float(data['temp']), float(data['humi']), float(data['moisture']), convertedLabelSoil,
                    convertedLabelCrop, float(data['nitrogen']), float(data['phosphorus']), float(data['potassium'])]])
    prediction_fertilizer = fertilizer_recomendation_model.predict(data)
    return JsonResponse({"prediction_fertilizer": prediction_fertilizer[0]})
