from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
import numpy as np

# df = pd.read_csv( "final.csv" )
# Create your views here.
def index (request):
    # return HttpResponse("hello")
    return render(request,'agrost/index.html')
    
def result (request):
     # Crop = request.POST.get('Crop')
    # State = request.POST.get('State')
    # Season = request.POST.get('Season')
    # Rainfall = request.POST.get('Rainfall')
    import pandas as pd
    import numpy as np
    df = pd.read_csv("final.csv")
    import warnings
    warnings.filterwarnings('ignore')
    #df.drop('Column1',axis=1)
    df['yield_cat'] = [0]*74975

    for i in range((74796)):
        if df['Production'][i]<=1:
            df['yield_cat'][i] = 0
        elif df['Production'][i]<=3:
            df['yield_cat'][i] = 1
        elif df['Production'][i]<=10:
            df['yield_cat'][i] = 2      
        elif df['Production'][i]<=50:
            df['yield_cat'][i] = 3
        else:
            df['yield_cat'][i] = 4

    df.columns
    df.dropna()
    df.isnull().sum()
    df[df["Yield "] == 'Infinity']

    df.replace([np.inf, -np.inf], np.nan,inplace=True)
    df.replace(to_replace =np.nan, value = 0,inplace=True) 
    df[df["Yield "] == 'Infinity']["Yield "] = 0
    df.drop([47178,48701],inplace=True)
    df[df["Yield "] == 'Infinity']
    a = list(set(df['State']))
    b = [i for i in range(len(a))]
    c = list(set(df['Season']))
    d = [i for i in range(len(c))]
    e = list(set(df['Crop']))
    f = [i for i in range(len(e))]
    df.State = df.State.replace(to_replace=a, value=b)
    df.Season = df.Season.replace(to_replace=c, value=d)
    df.Crop = df.Crop.replace(to_replace=e, value=f)
            

    from sklearn.ensemble import RandomForestClassifier
    import numpy
    import matplotlib.pyplot as plot
    import pandas
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
            
            
    Crop = request.POST.get('Crop')
    State = request.POST.get('State')
    Season = request.POST.get('Season')
    Rainfall = request.POST.get('Rainfall')
    Rainfall = float(Rainfall)
            
    if State in a:
        State = a.index(State)
    if Season in c:
        Season = c.index(Season)
    if Crop in e:
        Crop = e.index(Crop)
                
    if Season in set(df['Season']) and Crop in set(df['Crop']) and State in set(df['State']):
        if Crop in set(df[df['State']==State]['Crop']) and Crop in set(df[df['Season']==Season]['Crop']):
            x = np.array(df[['Rainfall','Season','Crop','State']])
            y = np.array(df['yield_cat']).reshape(-1,1)
            x = x.astype(int)
            y = y.astype(int)
            # Splitting dataset into train and test set 
            X_train, X_test, y_train, y_test = train_test_split( x, y, test_size = 1/3, random_state = 0 ) 
            #Create a Gaussian Classifier
            clf=RandomForestClassifier(n_estimators=100)

            #Train the model using the training sets y_pred=clf.predict(X_test)
            clf.fit(X_train,y_train)

            y_pred=clf.predict(X_test)
            x_new = np.array([[Rainfall,Season,Crop,State]])
            y_new=clf.predict(x_new)
            cat_ = y_new[0]
            print(x_new)
            if cat_ == 0:
                out = "Predicted Yield can be less than 1"
            elif cat_ == 1:
                out = "Predicted Yield can be less than 3"
            elif cat_ == 2:
                out = "Predicted Yield can be less than 10"
            elif cat_ == 3:
                out = "Predicted Yield can be less than 50"
            else:
                out = "Predicted Yield can be greater than 50"

            from sklearn import metrics
            # Model Accuracy, how often is the classifier correct?
            acc = metrics.accuracy_score(y_test, y_pred)
            print("Accuracy:",acc)
            dictionary = {'outputs':out,'accus':acc}
        else:
            out = 'Crop is not cultivated in given season or state'
            dictionary = {'outputs':out,'accus':0}
    else:
        out = 'Enter valid input'
        dictionary = {'outputs':out,'accus':0}
        
    return render(request,'agrost/Result.html',dictionary)