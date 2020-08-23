[![AnalyticsDojo](https://github.com/rpi-techfundamentals/spring2019-materials/blob/master/fig/final-logo.png?raw=1)](http://rpi.analyticsdojo.com)
<center><h1> Sentiment Analysis</h1></center>
<center><h3><a href = 'http://rpi.analyticsdojo.com'>rpi.analyticsdojo.com</a></h3></center>

### Sentiment Analysis
- The [pricing](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/text-analytics/).
- Helpful [example](https://www.johanahlen.info/en/2017/04/text-analytics-and-sentiment-analysis-with-microsoft-cognitive-services/).
- [Microsoft quickstart documentation](https://docs.microsoft.com/en-us/azure/cognitive-services/text-analytics/quickstarts/python)


#This imports some required packages.
#import utils #Often I'll develop functions in a notebook then move to utils. 
import pandas as pd
import urllib.request
import json
import pprint

### Keys are Needed for the API. 
You will find the appropriate key on the slack channel. 


#I typically store my config values. 
azure_text_endpoint= 'https://eastus.api.cognitive.microsoft.com/text/analytics/v2.0'
azure_text_key1= 'key posted on slack'


def azure_request(command, endpoint, key, postdata):
    #Set URI
    uri=endpoint+"/"+command
    #Set header
    headers = {}
    headers['Ocp-Apim-Subscription-Key'] = key
    headers['Content-Type'] = 'application/json'
    headers['Accept'] = 'application/json'
    #Make request
    request = urllib.request.Request(uri, postdata, headers)
    response = urllib.request.urlopen(request)
    return json.loads(response.read().decode('utf-8'))

def df_to_postdata(df):
    return json.dumps({'documents': json.loads(df.to_json(orient='records')) }).encode('utf-8')


#First lets test with sample data from the examples. 
#https://docs.microsoft.com/en-us/azure/cognitive-services/text-analytics/quickstarts/python
test_data = json.dumps({ 'documents': [
    { 'id': '1', 'language': 'en', 'text': 'I really enjoy the new XBox One S. It has a clean look, it has 4K/HDR resolution and it is affordable.' },
    { 'id': '2', 'language': 'es', 'text': 'Este ha sido un dia terrible, llegué tarde al trabajo debido a un accidente automobilistico.' }
]}).encode('utf-8')
pprint.pprint(test_data)

test_result=azure_request('sentiment', azure_text_endpoint, azure_text_key1, test_data)
test_result