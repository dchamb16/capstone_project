from fastapi import FastAPI, HTTPException
from tensorflow.keras.models import load_model
from pydantic import BaseModel
from clean_text import CleanText
from encode_text import EncodeText

app = FastAPI()

class StockIn(BaseModel):
    ticket_text: str

class StockOut(BaseModel):
    result: float

# encode text class
encoder = EncodeText()

# clean text class
ct = CleanText()

model = load_model('./classification_model.h5')

@app.post('/support_ticket_classification/result', response_model=StockOut, status_code=200)
def get_prediction(payload: StockIn):
    ticket_text = payload.ticket_text

    #tt = clean text
    tt = ct.prepare_text(ticket_text)
    #tt = encode text
    tt = encoder.encode_text([tt], test_data=True)

    result = model.predict(tt)

    if not result:
        raise HTTPException(status_code = 400, detail = 'Model not found.')

    response_object = {"result":result}
    return response_object
