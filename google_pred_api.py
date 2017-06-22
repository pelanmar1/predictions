# Pedro Lanzagorta

from oauth2client.service_account import ServiceAccountCredentials
from httplib2 import Http
from apiclient.discovery import build
import gevent.monkey
import simplejson as json



class GooglePredictionEngine:
    # This is a simple class to make Google API calls easier
    def __init__(self,PROJECT_NAME,MODEL_NAME,LOGS=False):
        self.PROJECT_NAME=PROJECT_NAME
        self.MODEL_NAME = MODEL_NAME
        self.LOGS = LOGS
        
    # By defining an array with scope permissions and a path to the keyfile provided by google OAUTH2 authenication
    # is performed
    def authenticate(self,scopes,keyfile_path):
        credentials = ServiceAccountCredentials.from_json_keyfile_name(keyfile_path, scopes=scopes)
        http_auth = credentials.authorize(Http())
        service = build('prediction', 'v1.6',http=http_auth)
        self.pred_api = service.trainedmodels()
        
    def getModel(self):
        # Gives information about the current loaded Prediction Model
        response = self.pred_api.get(project=self.PROJECT_NAME, id=self.MODEL_NAME).execute()
        self.printResponse(response)
        return response

    def createModel(self,storage):
        # By providing a storage path (google_data_bucket_name/file_name) a new model is trained for predictions
        response = self.pred_api.insert(project=self.PROJECT_NAME, body={'storageDataLocation': storage ,'id':self.MODEL_NAME}).execute()
        self.printResponse(response)
        return response
        
    def analyzeModel(self):
        # Shows information about the current loaded Prediction Model after trained
        response = self.pred_api.analyze(project=self.PROJECT_NAME, id=self.MODEL_NAME).execute()
        self.printResponse(response)
        return response
        
    def deleteModel(self):
        response = self.pred_api.delete(project=self.PROJECT_NAME,id=self.MODEL_NAME).execute()
        self.printResponse(response)
        return response

        
    def predict(self,val):
        # By providing a single feature value (day_number), an output estimator is returned
        val = [val]
        body = {'input' : {'csvInstance': val}}
        response = self.pred_api.predict(project=self.PROJECT_NAME, id=self.MODEL_NAME, body=body).execute()
        self.printResponse(response)
        return response
        
    def predictMany(self,values):
        # My providing a list of feture values (day_number), multiple output estimators are returned
        # an Asynchronous job is made so we can wait until all REST calls to the API are performed
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
        # You can choose whether you want the methods to print to the terminal
        if self.LOGS == True:
            print (json.dumps(response, sort_keys=True, indent=4))
    

