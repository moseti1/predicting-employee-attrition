
#import required packages
import pandas  as pd 
import matplotlib.pyplot as plt
import seaborn as sns 



from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score





# suppress warnings 
import warnings
warnings.filterwarnings('ignore')





#filenames 
file = 'data/Hash-Analytic-Python-Analytics-Problem-case-study-1.xlsx'



# read data from excel files. 
def load_data(file,sheet_name):
    data = pd.read_excel(file,sheet_name)
    
    return data

#load data 
#Existing employee data in sheet1 into data1
data1 = pd.read_excel('data/Hash-Analytic-Python-Analytics-Problem-case-study-1.xlsx',sheet_name=1)
#employees who left into data2 
data2 = pd.read_excel('data/Hash-Analytic-Python-Analytics-Problem-case-study-1.xlsx',sheet_name=2)

#introduce column 'Attrition' to indicate attrition if the  employees left or stayed
data1.insert(1, 'Attrition', 0)

data2.insert(1, 'Attrition', 1)    




#combine the two dataframes 
def merge_data(data1,data2):
    full_data = pd.concat([data1,data2])
    
    return full_data


full_data = pd.concat([data1,data2])

#peak at the last few records of data
full_data.tail(5)


# # **Task 2** 
# # Data cleaning 
# The first step to clean data is to check for some null values. We us `.isnull().any()` to check and note an value that might be missing in all our columns. In the case of our data we have none that is missing. So our data i okay for now.




#check for missing values 
full_data.isnull().any()




#check the number who left and those who did not
full_data["Attrition"].value_counts()


# # **Task 3** 
# # Data Visualization. 
# 
# Inorder to gain meaningful information about our data we need to explore it further. By this we can make some statistical visualizations using the data to compare various aspects of the data using statistics. 
# 
# Some visualization you can make out of data include some but not limited to Column Chart
#   
#    - Bar Graph                 
#    - Pie Chart
#    - Stacked Bar Graph          
#    - Waterfall Chart
#    - Stacked Column Chart      
#    - Bubble Chart
#    - Area Chart                
#    - Scatter Plot Chart
#    - Dual Axis Chart           
#    - Bullet Graph
#    - Line Graph                 
#    - Funnel Chart
#    - Mekko Chart                
#    - Heat Map
# 








#do visualitions
def bar_plot(full_data):
    pd.crosstab(full_data.dept,full_data.Attrition).plot(kind='bar')
    plt.title('Turn over frequency for Department')
    plt.xlabel('Department')
    plt.ylabel('Frequency of Turnover')
    #plt.show()
    #plt.savefig('img/department_bar_chart')
    





#Bar chart for employee salary level and the frequency of turnover
def salary_bar(full_data):
    
    table=pd.crosstab(full_data.salary, full_data.Attrition)
    table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
    plt.title('Stacked Bar Chart of Salary Level vs Turnover')
    plt.xlabel('Salary Level')
    plt.ylabel('Proportion of Employees')
    plt.show()
    





#Proportion of employees who left by department
attrition_prop = pd.crosstab(full_data.dept,full_data.Attrition)





#Histogram of numerical variables
def hist(full_data):
    num_bins = 10 
    full_data.hist(bins=num_bins,figsize=(20,25))
    #plt.savefig("img/hr_histogram_plots")
    plt.show()
    







#create dummy variables for categorical variables 
categorical_variables = ['dept','salary']

def dummify_category(full_data,categorical_variables):
    for var in categorical_variables:
        cat_list = 'var' + '_' + var
        cat_list = pd.get_dummies(full_data[var],prefix=var)
        full_data1 = full_data.join(cat_list)
        full_data = full_data
    





full_data.drop(full_data.columns[[9,10]],axis=1,inplace=True)





full_data.columns.values


# # **Task 4**
# 
# # Feature Selection
# 
# 
# We select the most significant features using RFE. The Recursive Feature Elimination (RFE) works by recursively removing variables and building a model on those variables that remain. It uses the model accuracy to identify which variables (and combination of variables) contribute the most to predicting the target attribute. The Attrition variable has two choices 0 or 1 so we will use logistic regression model to make the predictions, it is our target attribute



full_data_vars = full_data.columns.values.tolist()
y=['Attrition']
X=[i for i in full_data_vars if i not in y]








#modelLg = LogisticRegression()
#rfe = RFE(modelLg,10)
#rfe = rfe.fit(full_data[X],full_data[y])
    
    




cols = [ 'satisfaction_level',
 'last_evaluation',
 'number_project',
 'average_montly_hours',
 'time_spend_company',
 'Work_accident',
 'promotion_last_5years']
X = full_data[cols]
y = full_data['Attrition']


# # **Task 5**
# 
# # Model Predicting and  Fitting.




#split the data into training and test samples 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)



#Logistic regression model 
logreg = LogisticRegression()
logreg= logreg.fit(X_train, y_train)




# print('Logistic regression accuracy: {:.3f}'.format(accuracy_score(y_test, logreg.predict(X_test))))




from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
#create a model and train 
modelRf = RandomForestClassifier()
modelRf.fit(X_train,y_train)




#predict the results for the test
test_pred = modelRf.predict(X_test)

#test the accuracy 
#print('Random Forest Accuracy: {:.3f}'.format(accuracy_score(y_test, modelRf.predict(X_test))))

def score_accuracy(model,X_test,y_test):
    accuracy_score(y_test,model.predict(X_test))


# create a support vector machine

from sklearn.svm import SVC

svc = SVC()


svc = svc.fit(X_train, y_train)







def random_cls(X_train,y_train):
    kfold = model_selection.KFold(n_splits=10, random_state=7)
    modelCV = RandomForestClassifier()
    scoring = 'accuracy'
    results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
    print("10-fold cross validation average accuracy: %.3f" % (results.mean()))






def cls_report(model,X_test,y_test):
    print(classification_report(y_test, modelRf.predict(X_test)))




# random forests graph
y_pred = modelRf.predict(X_test)


from sklearn.metrics import confusion_matrix
import seaborn as sns

def random_forest_graph(X_test,y_pred,y_test):
    forest_cm = metrics.confusion_matrix(y_pred, y_test, [1,0])
    sns.heatmap(forest_cm, annot=True, fmt='.2f',xticklabels = ["Left", "Stayed"] , yticklabels = ["Left", "Stayed"] )
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.title('Random Forest')
    #plt.savefig('img/random_forest')
    plt.show()




#cls_report= classification_report(y_test, svc.predict(X_test))




def heatmap_svc():
    svc_y_pred = svc.predict(X_test)
    svc_cm = metrics.confusion_matrix(svc_y_pred, y_test, [1,0])
    sns.heatmap(svc_cm, annot=True, fmt='.2f',xticklabels = ["Left", "Stayed"] , yticklabels = ["Left","Stayed"])
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.title('Support Vector Machine')
    #plt.savefig('img/support_vector_machine')
    plt.show()





from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


def roc_curve(X_test,y_test,model):
    logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
    fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
    rf_roc_auc = roc_auc_score(y_test, modelRf.predict(X_test))
    rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, modelRf.predict_proba(X_test)[:,1])
    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot(rf_fpr, rf_tpr, label='Random Forest (area = %0.2f)' % rf_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    #plt.savefig('img/ROC_CURVE')
    plt.show()




# Feature importances 
def feature_importances_graph(model,X):
    feat_importances = pd.Series(model.feature_importances_,index=X.columns)
    feat_importances= feat_importances.nlargest(20)
    feat_importances.plot(kind='barh')
    #plt.savefig('img/feat_importances_barh')
    plt.show()



# feature importances graph


def feature_importance_percentage(model):
    feature_labels = np.array(['satisfaction_level', 'last_evaluation','number_project', 'average_montly_hours', 'time_spend_company',
           'Work_accident', 'promotion_last_5years'])
    importance = model.feature_importances_
    feature_indexes_by_importance = importance.argsort()
    for index in feature_indexes_by_importance:
        print('{}-{:.2f}%'.format(feature_labels[index], (importance[index] *100.0)))






