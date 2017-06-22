# Pedro Lanzagorta

from oauth2client.service_account import ServiceAccountCredentials
from httplib2 import Http
from apiclient.discovery import build
import gevent.monkey
import simplejson as json



class GooglePredictionEngine:

    def __init__(self,PROJECT_NAME,MODEL_NAME,LOGS=False):
        self.PROJECT_NAME=PROJECT_NAME
        self.MODEL_NAME = MODEL_NAME
        self.LOGS = LOGS
        
    def authenticate(self,scopes,keyfile_path):
        credentials = ServiceAccountCredentials.from_json_keyfile_name(keyfile_path, scopes=scopes)
        http_auth = credentials.authorize(Http())
        service = build('prediction', 'v1.6',http=http_auth)
        self.pred_api = service.trainedmodels()
        
    def getModel(self):
        response = self.pred_api.get(project=self.PROJECT_NAME, id=self.MODEL_NAME).execute()
        self.printResponse(response)
        return response

    def createModel(self,storage):
        response = self.pred_api.insert(project=self.PROJECT_NAME, body={'storageDataLocation': storage ,'id':self.MODEL_NAME}).execute()
        self.printResponse(response)
        return response
        
    def analyzeModel(self):
        response = self.pred_api.analyze(project=self.PROJECT_NAME, id=self.MODEL_NAME).execute()
        self.printResponse(response)
        return response
        
    def deleteModel(self):
        response = self.pred_api.delete(project=self.PROJECT_NAME,id=self.MODEL_NAME).execute()
        self.printResponse(response)
        return response

        
    def predict(self,val):
        val = [val]
        body = {'input' : {'csvInstance': val}}
        response = self.pred_api.predict(project=self.PROJECT_NAME, id=self.MODEL_NAME, body=body).execute()
        self.printResponse(response)
        return response
        
    def predictMany(self,values):
        jobs = [gevent.spawn(self.predict, value) for value in values]
        gevent.wait(jobs)
        predictions = {}
        i = 0
        for job in jobs:
            output = job.value['outputValue']
            predictions[values[i]]= output
            i += 1
        if self.LOGS == True:
            print(predictions)
        return predictions
            
    def printResponse(self,response):
        if self.LOGS == True:
            print (json.dumps(response, sort_keys=True, indent=4))
    

