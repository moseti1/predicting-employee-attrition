# load the require packages 
import pandas  as pd 
import matplotlib.pyplot as plt
import seaborn as sns 

import dash
from dash import html
from dash import dcc
import plotly.graph_objects as go
import plotly.express as px



from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score




import attrition

from attrition import (load_data, bar_plot, salary_bar,roc_curve, attrition_prop,hist,dummify_category, logreg,modelRf,feature_importance_percentage,feature_importances_graph,heatmap_svc,random_forest_graph,cls_report,random_cls,svc,test_pred,full_data)

#load data 
#Existing employee data in sheet1 into data1
data1 = pd.read_excel('data/Hash-Analytic-Python-Analytics-Problem-case-study-1.xlsx',sheet_name=1)
#employees who left into data2 
data2 = pd.read_excel('data/Hash-Analytic-Python-Analytics-Problem-case-study-1.xlsx',sheet_name=2)

#introduce column 'Attrition' to indicate attrition if the  employees left or stayed
data1.insert(1, 'Attrition', 0)

data2.insert(1, 'Attrition', 1)  


full_data_vars = full_data.columns.values.tolist()
y=['Attrition']
X=[i for i in full_data_vars if i not in y]


# merge the data
full_data = pd.concat([data1,data2])







# plots 

#hist_plots = hist(full_data)

#turnover_bar = bar_plot(full_data)

#salary_bar = salary_bar(full_data)

# split data
#X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)


app = dash.Dash()   #initialising dash app

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

# build the dashboard
app.layout = html.Div(children=[ 
                       
    
    html.H1(children='A Web Application For Employee Atrrition.', style={
        'textAlign': 'center',
        'color': colors['text']
    })]
                      
                      
)
                       


if __name__ == '__main__': 
    app.run_server(debug=True)