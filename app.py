# Importing essential libraries and modules

from flask import Flask, render_template, request, Markup
import numpy as np
#import pandas as pd
import os
import requests
import config
import pickle
import io
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# ==============================================================================================

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------





#disease_dic= ["Eye Spot","Healthy Leaf","Red Leaf Spot","Redrot","Ring Spot"]



from model_predict2  import pred_leaf_disease
from model_predict2  import pred_leaf_disease,save_explainable_output

# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------


app = Flask(__name__)

# render home page
plant_data=[
  {
    "plant_name": "Apple___Apple_scab",
    "disease_name": "Apple Scab",
    "cure": "Apply fungicides like Captan or Mancozeb. Remove infected leaves and fruits.",
    "precaution": "Ensure proper air circulation, avoid overhead watering, and use resistant varieties.",
    "chemicals_used": "Captan, Mancozeb, Copper-based fungicides"
  },
  {
    "plant_name": "Apple___Black_rot",
    "disease_name": "Black Rot",
    "cure": "Prune infected branches and remove mummified fruits. Apply fungicides like Thiophanate-methyl.",
    "precaution": "Maintain proper orchard sanitation and ensure adequate tree nutrition.",
    "chemicals_used": "Thiophanate-methyl, Copper sprays"
  },
  {
    "plant_name": "Apple___healthy",
    "disease_name": "Healthy Plant",
    "cure": "No disease present, maintain proper care and nutrition.",
    "precaution": "Regular monitoring and preventive fungicide application if needed.",
    "chemicals_used": "None"
  },
  {
    "plant_name": "Corn_(maize)___Common_rust_",
    "disease_name": "Common Rust",
    "cure": "Use resistant varieties and apply fungicides like Propiconazole.",
    "precaution": "Rotate crops and avoid overhead irrigation.",
    "chemicals_used": "Propiconazole, Triazole fungicides"
  },
  {
    "plant_name": "Corn_(maize)___Northern_Leaf_Blight",
    "disease_name": "Northern Leaf Blight",
    "cure": "Apply fungicides like Azoxystrobin. Use resistant hybrids.",
    "precaution": "Crop rotation and residue management.",
    "chemicals_used": "Azoxystrobin, Strobilurin fungicides"
  },
  {
    "plant_name": "Corn_(maize)___healthy",
    "disease_name": "Healthy Plant",
    "cure": "No disease present, maintain proper care and nutrition.",
    "precaution": "Regular field inspections and preventive measures.",
    "chemicals_used": "None"
  },
  {
    "plant_name": "Grape___Black_rot",
    "disease_name": "Black Rot",
    "cure": "Apply fungicides like Myclobutanil and remove infected vines.",
    "precaution": "Ensure good air circulation and proper pruning.",
    "chemicals_used": "Myclobutanil, Mancozeb, Captan"
  },
  {
    "plant_name": "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "disease_name": "Isariopsis Leaf Spot",
    "cure": "Use copper-based fungicides and remove infected leaves.",
    "precaution": "Avoid excessive moisture and ensure good drainage.",
    "chemicals_used": "Copper hydroxide, Mancozeb"
  },
  {
    "plant_name": "Grape___healthy",
    "disease_name": "Healthy Plant",
    "cure": "No disease present, maintain proper care and nutrition.",
    "precaution": "Regular inspections and good vineyard management.",
    "chemicals_used": "None"
  },
  {
    "plant_name": "Peach___Bacterial_spot",
    "disease_name": "Bacterial Spot",
    "cure": "Apply copper-based sprays and prune infected areas.",
    "precaution": "Avoid overhead watering and use resistant varieties.",
    "chemicals_used": "Copper hydroxide, Streptomycin"
  },
  {
    "plant_name": "Peach___healthy",
    "disease_name": "Healthy Plant",
    "cure": "No disease present, maintain proper care and nutrition.",
    "precaution": "Regular monitoring and good orchard management.",
    "chemicals_used": "None"
  },
  {
    "plant_name": "Pepper,_bell___Bacterial_spot",
    "disease_name": "Bacterial Spot",
    "cure": "Use copper-based sprays and remove infected leaves.",
    "precaution": "Ensure proper spacing and good air circulation.",
    "chemicals_used": "Copper sulfate, Streptomycin"
  },
  {
    "plant_name": "Pepper,_bell___healthy",
    "disease_name": "Healthy Plant",
    "cure": "No disease present, maintain proper care and nutrition.",
    "precaution": "Regular field inspections and good crop management.",
    "chemicals_used": "None"
  },
  {
    "plant_name": "Potato___Early_blight",
    "disease_name": "Early Blight",
    "cure": "Apply fungicides like Chlorothalonil and remove infected leaves.",
    "precaution": "Use disease-free seeds and practice crop rotation.",
    "chemicals_used": "Chlorothalonil, Mancozeb"
  },
  {
    "plant_name": "Potato___Late_blight",
    "disease_name": "Late Blight",
    "cure": "Use fungicides like Metalaxyl and remove affected plants.",
    "precaution": "Avoid excessive moisture and plant-resistant varieties.",
    "chemicals_used": "Metalaxyl, Copper fungicides"
  },
  {
    "plant_name": "Potato___healthy",
    "disease_name": "Healthy Plant",
    "cure": "No disease present, maintain proper care and nutrition.",
    "precaution": "Regular inspections and good field management.",
    "chemicals_used": "None"
  },
  {
    "plant_name": "Tomato___Early_blight",
    "disease_name": "Early Blight",
    "cure": "Apply fungicides like Chlorothalonil and prune lower leaves.",
    "precaution": "Ensure proper plant spacing and avoid wet leaves.",
    "chemicals_used": "Chlorothalonil, Mancozeb"
  },
  {
    "plant_name": "Tomato___Late_blight",
    "disease_name": "Late Blight",
    "cure": "Use fungicides like Metalaxyl and remove infected plants.",
    "precaution": "Avoid overhead irrigation and use resistant varieties.",
    "chemicals_used": "Metalaxyl, Copper fungicides"
  },
  {
    "plant_name": "Tomato___healthy",
    "disease_name": "Healthy Plant",
    "cure": "No disease present, maintain proper care and nutrition.",
    "precaution": "Regular field monitoring and preventive measures.",
    "chemicals_used": "None"
  }

]





@ app.route('/')
def home():
    title = 'Plant Leaf Disease Detection using Ensemble Learning and Explainable AI'
    return render_template('index.html', title=title)  

# render crop recommendation form page

@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Plant Leaf Disease Detection using Ensemble Learning and Explainable AI'

    if request.method == 'POST':
        file = request.files.get('file')

        if not file:
            return render_template('rust.html', title=title)

        # Process the uploaded file
        img = Image.open(file)
        img.save('output.png')
        

        # Make the prediction
        prediction = pred_leaf_disease("output.png")
        save_explainable_output("output.png")
        #prediction = str(disease_dic[prediction])

        print("Prediction result:", prediction)
         
        for plant in plant_data:
                    if plant['plant_name'] == prediction:
                        details = plant
                        break
        #    error_message = f"No details found for {plant_name}."

        return render_template('rust-result.html',details=details)



    # Default page rendering
    return render_template('rust.html', title=title)





# render disease prediction result page


# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=True)
