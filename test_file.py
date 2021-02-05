from zendesk import fetchZenDesk
import json
import pandas as pd

with open('./.creds.json') as f:
    creds = json.load(f)

usr = creds['usr']
token = creds['pwd']
url = creds['url']

ticket_type_id = 360026814711

zd = fetchZenDesk(usr, token, url)

df = zd.get_messages_and_outcome(150, ticket_type_id)

df['issue'].value_counts()
df.tail()

df.to_pickle('./data.pkl')



df = pd.read_pickle('./data.pkl')
