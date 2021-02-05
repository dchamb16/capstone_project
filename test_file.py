from zendesk import fetchZenDesk
from clean_text import CleanText
import json

with open('./.creds.json') as f:
    creds = json.load(f)

usr = creds['usr']
token = creds['pwd']
url = creds['url']

ticket_type_id = 360026814711

zd = fetchZenDesk(usr, token, url)

df = zd.get_messages_and_outcome(15, ticket_type_id)

df['issue'].value_counts()
df.tail()

df.to_pickle('./data.pkl')


import pandas as pd

df = pd.read_pickle('./data.pkl')

df.head()

ct = CleanText()
df['clean_text'] = df['ticket_text'].apply(lambda x: ct.prepare_text(x))

df['clean_text'][12]

'this is a followup to your previous request 149791'
