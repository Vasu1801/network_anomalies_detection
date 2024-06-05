from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import numpy as np 

app = Flask(__name__)

encoder_files = ['encoder1.pkl', 'encoder2.pkl', 'encoder3.pkl', 'encoder4.pkl']
loaded_encoders = []
for file_name in encoder_files:
    with open(file_name, 'rb') as f:
        loaded_encoders.append(pickle.load(f))

def preprocess_columns(df):
    df = df.drop(['Unnamed: 0','land','urgent','numfailedlogins','numoutboundcmds','numroot', 'srvserrorrate', 'dsthostserrorrate', 'dsthostsrvserrorrate', 'dsthostserrorrate', 'dsthostsrvserrorrate', 'srvrerrorrate', 'dsthostsrvrerrorrate', 'dsthostsrvrerrorrate', 'dsthostsrvserrorrate','attack'], axis=1)
    df['protocoltype'] = loaded_encoders[0].transform(df['protocoltype'])
    df['service'] = loaded_encoders[1].transform(df['service'])
    df['flag'] = loaded_encoders[2].transform(df['flag'])
    with open('standard_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    df_scaled = scaler.transform(df)  # Apply scaling
    df_scaled = pd.DataFrame(df_scaled, columns=df.columns)  # Convert scaled array back to DataFrame
    df_result = df_scaled.head(20)
    return df_result

def preprocess_input(protocoltype, service, flag, data2):
    protocoltype = loaded_encoders[0].transform([protocoltype])
    service = loaded_encoders[1].transform([service])
    flag = loaded_encoders[2].transform([flag])
    data = np.array(data2)
    with open('standard_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    data= np.insert(data, 1, protocoltype)
    data = np.insert(data, 2, service)
    data = np.insert(data, 3, flag)
    data_scaled = scaler.transform(data.reshape(1, -1))
    return data_scaled
    
@app.route('/')
def hello_world():
    return render_template("Home.html")

@app.route('/manual-input')
def manual_input():
    return render_template('Network anomaly detection.html')

@app.route('/upload-csv')
def upload_csv():
    return render_template('csv.html')

@app.route('/type', methods=['GET','POST'])
def types():
        duration = int(request.form.get('duration', 0))
        protocoltype = str(request.form.get('protocoltype', 0))
        service = str(request.form.get('service', 0))
        flag = str(request.form.get('flag', 0))
        srcbytes = int(request.form.get('srcbytes', 0))
        dstbytes = int(request.form.get('dstbytes', 0))
        wrongfragment = int(request.form.get('wrongfragment', 0))
        hot = int(request.form.get('hot', 0))
        loggedin = int(request.form.get('loggedin', 0))
        numcompromised = int(request.form.get('numcompromised', 0))
        rootshell = int(request.form.get('rootshell', 0))
        suattempted = int(request.form.get('suattempted', 0))
        numfilecreations = int(request.form.get('numfilecreations', 0))
        numshells = int(request.form.get('numshells', 0))
        numaccessfiles = int(request.form.get('numaccessfiles', 0))
        ishotlogin = int(request.form.get('ishotlogin', 0))
        isguestlogin = int(request.form.get('isguestlogin', 0))
        count = int(request.form.get('count', 0))
        srvcount = int(request.form.get('srvcount', 0))
        serrorrate = float(request.form.get('serrorrate', 0))
        rerrorrate = float(request.form.get('rerrorrate', 0))
        samesrvrate = float(request.form.get('samesrvrate', 0))
        diffsrvrate = float(request.form.get('diffsrvrate', 0))
        srvdiffhostrate = float(request.form.get('srvdiffhostrate', 0))
        dsthostcount = int(request.form.get('dsthostcount', 0))
        dsthostsrvcount = int(request.form.get('dsthostsrvcount', 0))
        dsthostsamesrvrate = float(request.form.get('dsthostsamesrvrate', 0))
        dsthostdiffsrvrate = float(request.form.get('dsthostdiffsrvrate', 0))
        dsthostsamesrcportrate = float(request.form.get('dsthostsamesrcportrate', 0))
        dsthostsrvdiffhostrate = float(request.form.get('dsthostsrvdiffhostrate', 0))
        dsthostrerrorrate = float(request.form.get('dsthostrerrorrate', 0))
        lastflag = int(request.form.get('lastflag', 0))
        data2 = [duration, srcbytes, dstbytes, wrongfragment, hot, loggedin, numcompromised, rootshell, suattempted, numfilecreations, numshells, numaccessfiles, ishotlogin, isguestlogin, count, srvcount, serrorrate, rerrorrate, samesrvrate, diffsrvrate, srvdiffhostrate, dsthostcount, dsthostsrvcount, dsthostsamesrvrate, dsthostdiffsrvrate, dsthostsamesrcportrate, dsthostsrvdiffhostrate, dsthostrerrorrate, lastflag]
        data = preprocess_input(protocoltype, service, flag, data2)
        
        # Load the SVM model
        with open('SVM_model.pkl', 'rb') as f:
            svm_model = pickle.load(f)
        
        # Make predictions
        predictions = svm_model.predict(data.reshape(1, -1))
        
        # Inverse transform predictions to get original labels
        predicted_labels = loaded_encoders[3].inverse_transform(predictions)
        
        # Render a template passing predicted labels to display
        return render_template('pred.html', variable=predicted_labels)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and file.filename.endswith('.csv'):
        try:
            df = pd.read_csv(file)
            df1 = preprocess_columns(df)
            with open('SVM_model.pkl', 'rb') as f:
                svm_model = pickle.load(f)
            predictions = svm_model.predict(df1)
            
            predicted_labels = loaded_encoders[3].inverse_transform(predictions)
            
            df=df.head(5)  # Only take the first five rows
            df = df.drop(['Unnamed: 0','land','urgent','numfailedlogins','numoutboundcmds','numroot', 'srvserrorrate', 'dsthostserrorrate', 'dsthostsrvserrorrate', 'dsthostserrorrate', 'dsthostsrvserrorrate', 'srvrerrorrate', 'dsthostsrvrerrorrate', 'dsthostsrvrerrorrate', 'dsthostsrvserrorrate','attack'], axis=1)
            df['attack'] = predicted_labels[:5]  # Only take the predicted labels for the first five rows
            df.to_csv('predicted_data.csv', index=False)
            
            return render_template('preprocessed_columns.html', data=df)
        except pd.errors.ParserError as pe:
            return jsonify({'error': 'Error parsing CSV file: {}'.format(str(pe))})
        except Exception as e:
            return jsonify({'error': 'Error processing CSV file: {}'.format(str(e))})
    else:
        return jsonify({'error': 'Invalid file format. Please upload a CSV file'})

@app.route('/predict', methods=['POST'])
def predict():
    request_data = request.json
    print(request_data)
    try:
        model = pickle.load(open('SVM_model.pkl', 'rb'))
        arr = request_data['values']
        x = model.predict([arr])
        return {'prediction': x[0]}
        
    except Exception as e:
        print(e)
        return {'prediction': 1}

if __name__ == '__main__':
    app.run(host='0.0.0.0')