from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('model_pipeline_final_GST.joblib')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect form data
    if request.method == 'POST':
        Column0= float(request.form['Column0'])
        Column1= int(request.form['Column1'])
        Column2= float(request.form['Column2'])
        Column3= float(request.form['Column3'])
        Column4= float(request.form['Column4'])
        Column5= float(request.form['Column5'])
        Column6= float(request.form['Column6'])
        Column7= float(request.form['Column7'])
        Column8= float(request.form['Column8'])
        Column9= float(request.form['Column9'])
        Column10= int(request.form['Column10'])
        Column11= int(request.form['Column11'])
        Column12= int(request.form['Column12'])
        Column13= int(request.form['Column13'])
        Column14= float(request.form['Column14'])
        Column15= float(request.form['Column15'])
        Column16= float(request.form['Column16'])
        Column17= int(request.form['Column17'])
        Column18= float(request.form['Column18'])
        Column19= int(request.form['Column19'])
        Column20= int(request.form['Column20'])
        Column21= int(request.form['Column21'])

    # Convert form data to DataFrame for prediction
        features = pd.DataFrame({
            'Column0':[Column0], 
            'Column1':[Column1], 
            'Column2':[Column2], 
            'Column3':[Column3], 
            'Column4':[Column4], 
            'Column5':[Column5], 
            'Column6':[Column6], 
            'Column7':[Column7], 
            'Column8':[Column8], 
            'Column9':[Column9], 
            'Column10':[Column10], 
            'Column11':[Column11], 
            'Column12':[Column12], 
            'Column13':[Column13], 
            'Column14':[Column14], 
            'Column15':[Column15], 
            'Column16':[Column16], 
            'Column17':[Column17], 
            'Column18':[Column18], 
            'Column19':[Column19], 
            'Column20':[Column20], 
            'Column21':[Column21], 
        })
    
    # Make prediction
        prediction = model.predict(features)
        prediction=prediction[0]
        if prediction == 0:
            prediction="Poor (0)"
        elif prediction == 1:
            prediction = "Good (1)"
        else:
            prediction="Standard (0-1)"
    
        return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
