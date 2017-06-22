# Pedro Lanzagorta
import pandas as pd
from fbprophet import Prophet
import numpy as np
from google_pred_api import GooglePredictionEngine
import matplotlib.pyplot as plt




def main():    
    keyfile_path = './google_api_test_key_d958387b183b.json'
    scopes = ['https://www.googleapis.com/auth/prediction', 'https://www.googleapis.com/auth/devstorage.full_control', \
              'https://www.googleapis.com/auth/cloud-platform']
    PROJECT_NAME='primal-outrider-115714'
    MODEL_NAME='test_timeseries'
    
    p = GooglePredictionEngine(PROJECT_NAME,MODEL_NAME)
    p.authenticate(scopes,keyfile_path)
    
    # MODEL SETUP
    #p.LOGS=True
    #p.createModel('quickstart-1497980730/test_training_data.csv')
    #p.analyzeModel()
    #p.getModel()
    
    data = createTrainingDataGPAPI()
    last_day_num = max(data.shape)
    
    periods = 30 #in days
    
    values = [x for x in range(last_day_num+1,last_day_num+periods)]
    
    predictions = p.predictMany(values)
    columns = ['y','ds']
    pred_df = pd.DataFrame(index=values,columns=columns)
    for day, pred in predictions.items():
        pred_df.loc[day]=[pred,day]
    #print(pred_df)
    data=data.astype(float)
    plt.plot(data['ds'],data['y'],'o')
    plt.holdOn = True
    plt.plot(pred_df['ds'],pred_df['y'],'or')
    
    








def createTrainingDataGPAPI():
    input_data_fn = "/Users/t-pelanz/Documents/MS/ltdashboard/pipeline/data/step1/mail_device_ts.csv" 
    input_targets_fn = "/Users/t-pelanz/Documents/MS/ltdashboard/pipeline/targets/mail_targets.csv" 
    targets_df = pd.read_csv(input_targets_fn, header=0, index_col=0, delimiter=",")
    df = pd.read_csv(input_data_fn, header=0, index_col=0, delimiter=",", parse_dates=True)
    
    metrics = ["mad_mobile"]
    metric_key = metrics[0]
    
    temp_df = pd.DataFrame({'ds': df[metric_key].index, 'y': df[metric_key].values}).dropna()
    temp_df=temp_df.reset_index(drop=True)
    temp_df['ds'] = (temp_df.index+1)
    temp_df = temp_df[['y','ds']]
    
    temp_df.to_csv('test_training_data.csv', sep=',',header=None,index=False)
    return temp_df



    
    
    

def prophetAnalysis():
    input_data_fn = "/Users/t-pelanz/Documents/MS/ltdashboard/pipeline/data/step1/mail_device_ts.csv" 
    input_targets_fn = "/Users/t-pelanz/Documents/MS/ltdashboard/pipeline/targets/mail_targets.csv" 
    targets_df = pd.read_csv(input_targets_fn, header=0, index_col=0, delimiter=",")
    df = pd.read_csv(input_data_fn, header=0, index_col=0, delimiter=",", parse_dates=True)
           
    metrics = ["mad_mobile"]
    
    for metric in metrics:
        metric = findStringInArray(df.columns,metric,'Metric name')
        target_metric = findStringInArray(targets_df.index,metric,'Target Metric Name')
        if target_metric != None:
            goal_target_value = targets_df.loc[target_metric]["target"]
            print (goal_target_value)
        else:
            goal_target_value = -1
        if metric!= None:
            analyzeData(df,metric,2)
    
    
    



def analyzeData(df,metric_key,code=1):
    functions = {
           1 : prophetAnalysis1,
           2 : prophetAnalysis2,
           } 
    
    
    functions[code](df,metric_key)












def prophetAnalysis1(df,metric_key):
    # Create a dataframe with just the x and y values
    prophet_df = pd.DataFrame({'ds': df[metric_key].index.values, 'y': df[metric_key].values}).dropna()
    prophet_df['y']= np.log(prophet_df['y'])
    m = Prophet()
    m.fit(prophet_df)
    
    future = m.make_future_dataframe(periods=10)
    forecast = m.predict(future)
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    m.plot(forecast)
    #m.plot_components(forecast)
    
def prophetAnalysis2(df,metric_key):
    # Create a dataframe with just the x and y values
    prophet_df = pd.DataFrame({'ds': df[metric_key].index.values, 'y': df[metric_key].values}).dropna()
    
    prophet_df['y']= np.log(prophet_df['y'])
    
    m = Prophet(growth='logistic')
    m.fit(prophet_df)
    
    df['cap'] = 100000000
    
    future = m.make_future_dataframe(periods=10)
    future['cap'] = 100000000
    forecast = m.predict(future)
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    m.plot(forecast)
    #m.plot_components(forecast)
    






def findStringInArray(metrics,metric_key,alert_title=None):
    alert_title= alert_title.upper()+' - '
    similar = []
    prompt = '> '
    for met in metrics:
        dist = damerauLevenshteinDistance(metric_key,met)
        if dist ==0:
            return metric_key
        elif 0<dist<=2:
            similar.append(met)
    if(len(similar)>0):
        msg = alert_title+'Did you mean? \n'
        for index,wrd in enumerate(similar):
            msg = msg + '('+str(index)+')'+wrd+'\n'
        msg = msg + 'Press anything else to cancel'
        print (msg)
        decision= input(prompt)
        if decision.isdigit() and int(decision)<len(similar):
            return str(similar[int(decision)])
        else:
            return None
    else:
        return None
    
            
def damerauLevenshteinDistance(wrd1,wrd2):
    wrd1 = wrd1.upper()
    wrd2 = wrd2.upper()
    # Setup matrix
    n = len(wrd1)
    m = len(wrd2)
    mat = [[]]
    mat = [[i for i in range(m+1)] for i in range(n+1)]
    for i in range(n+1):
        for j in range(m+1):
            if i == 0:
                mat[i][j] = j
            elif j == 0:
                mat[i][j] = i
            else:
                mat[i][j] = 0
    
    # Calculate costs

    for i in range(1,n+1):
        for j in range(1,m+1):
            c = 1
            if wrd1[i-1] == wrd2[j-1]:
                c = 0
            m1 = mat[i-1][j]+1
            m2 = mat[i][j-1]+1
            m3 = mat[i-1][j-1]+c
            if (1<i<n and 1<j<m) and (wrd1[i] == wrd2[j-1]) and (wrd1[i-1] == wrd2[j]):
                m4 = mat[i-2][j-2]+1
                mat[i][j] = min(m1,m2,m3,m4)
            else:
                mat[i][j] = min(m1,m2,m3)
    
    return mat[n][m]

    
if __name__ == "__main__":
    main()



