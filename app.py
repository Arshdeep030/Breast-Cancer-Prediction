import numpy as np
import pickle
import gradio as gr

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))

# Define prediction function
def predict(features):
    # Input: comma-separated string
    features = [float(x) for x in features.split(',')]
    np_features = np.array(features).reshape(1, -1)
    
    prediction = model.predict(np_features)[0]
    return "Cancerous" if prediction == 1 else "Not Cancerous"

# Create Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Enter 30 feature values (comma-separated)"),
    outputs="text",
    title="Breast Cancer Prediction",
    description="Enter the 30 comma-separated numerical features to predict if the tumor is cancerous or not."
)

iface.launch()
