from importlib import reload
from fastapi import FastAPI, HTTPException
from tensorflow.keras.models import load_model
from pydantic import BaseModel
from clean_text import CleanText
from encode_text import EncodeText
import uvicorn
import numpy as np
import pickle

app = FastAPI()

class StockIn(BaseModel):
    ticket_text: str

class StockOut(BaseModel):
    result: str

#encode text class
encoder = EncodeText()

encoder.load_encoder_variables('./encoder_variables.json')

#clean text class
ct = CleanText()

#labels
label_binarizer_path = './label_binarizer.pkl'
with open(label_binarizer_path, 'rb') as handle:
    lb = pickle.load(handle)

labels = list(lb.classes_)

model = load_model('./classification_model.h5')

@app.get('/')
def read_root():
    return {'message':'Welcome to the API'}

@app.post('/support_ticket_classification/result', response_model=StockOut, status_code=200)
def get_prediction(payload: StockIn):
    ticket_text = payload.ticket_text
    
    #tt = clean text
    tt = ct.prepare_text(ticket_text)
    
    #tt = encode text
    tt = encoder.encode_text(tt, test_data=True)
    print([tt])
    result = model.predict([tt])[0]
    print(result)
    res = np.argmax(result)
    
    pred_label = labels[res]
    
        
    if not pred_label:
        raise HTTPException(status_code = 400, detail = 'Model not found.')

    response_object = {"result":pred_label}
    return response_object

if __name__=='__main__':
    uvicorn.run("main:app", host='0.0.0.0', port=5000, reload=True)