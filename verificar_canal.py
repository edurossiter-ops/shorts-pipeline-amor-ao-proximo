import json
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

token = json.load(open('youtube_token.json'))
creds = Credentials(
    token=token.get('token'),
    refresh_token=token.get('refresh_token'),
    token_uri=token.get('token_uri'),
    client_id=token.get('client_id'),
    client_secret=token.get('client_secret'),
)
youtube = build('youtube', 'v3', credentials=creds)
resp = youtube.channels().list(part='snippet', mine=True).execute()
canal = resp['items'][0]['snippet']['title']
canal_id = resp['items'][0]['id']
print(f'Canal: {canal}')
print(f'ID: {canal_id}')
