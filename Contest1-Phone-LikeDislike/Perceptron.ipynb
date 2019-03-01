

#Import into DF
df_phone_train=pd.DataFrame(train_new)
df_phone_test=pd.DataFrame(test_new)


X_train = df_phone_train

#Convert Rating to Binary
X_train['Rating_Binary'] = X_train['Rating'].map(lambda x: 0 if x < 4 else 1)

#Convert RAM to same unit - GB
X_train['RAM'] = X_train['RAM'].map(lambda x: 0.5 if x > 500 else x)


#Create Y_train
Y_train = X_train['Rating_Binary']


#Setup X_train removing unwanted columns
X_train = df_phone_train.drop('Rating',axis=1)
X_train = X_train.drop('Rating_Binary',axis=1)
X_train = X_train.drop('PhoneId',axis=1)


#Change Column Name to avoid Syntax error where referring to these columns
X_train=X_train.rename(columns = {'Brand_10.or':'Brand_10'})
X_train=X_train.rename(columns = {'SIM Slot(s)_Dual SIM, GSM+GSM':'GSMGSM'})
X_train=X_train.rename(columns = {'Brand_Xiaomi Poco':'POCO'})
X_train=X_train.rename(columns = {'SIM Slot(s)_Dual SIM, GSM+CDMA':'GSMCDMA'})
X_train=X_train.rename(columns = {'Num_cores_Tru-Octa':'Num_cores_Tru_Octa'})


#Prepare X_test - repeate same steps as X_train
X_test = df_phone_test.drop('PhoneId',axis=1)
X_test['RAM'] = X_test['RAM'].map(lambda x: 0.5 if x > 500 else x)

X_test=X_test.rename(columns = {'Brand_10.or':'Brand_10'})
X_test=X_test.rename(columns = {'SIM Slot(s)_Dual SIM, GSM+GSM':'GSMGSM'})
X_test=X_test.rename(columns = {'Brand_Xiaomi Poco':'POCO'})
X_test=X_test.rename(columns = {'SIM Slot(s)_Dual SIM, GSM+CDMA':'GSMCDMA'})
X_test=X_test.rename(columns = {'Num_cores_Tru-Octa':'Num_cores_Tru_Octa'})




#The Columns below had a inverse relationship with being LIKED. Higher the value, Higher the chance of being DISLIKED. 
#So swappping the values for the model to consider this inverse relationship

X_train['Brand_Blackberry'] = X_train['Brand_Blackberry'].map({1:0,0:1})    
X_train['Brand_HTC'] = X_train['Brand_HTC'].map({1:0,0:1})    
X_train['Brand_Micromax'] = X_train['Brand_Micromax'].map({1:0,0:1})    
X_train['Brand_Mobiistar'] = X_train['Brand_Mobiistar'].map({1:0,0:1})    
X_train['Brand_Panasonic'] = X_train['Brand_Panasonic'].map({1:0,0:1})    
X_train['Brand_Yu'] = X_train['Brand_Yu'].map({1:0,0:1})    
X_train['Sim1_3G'] = X_train['Sim1_3G'].map({1:0,0:1})  

X_train['Brand_LG'] = X_train['Brand_LG'].map({1:0,0:1}) 
X_train['Brand_Sony'] = X_train['Brand_Sony'].map({1:0,0:1}) 
X_train['Num_cores_Dual'] = X_train['Num_cores_Dual'].map({1:0,0:1}) 
X_train['Num_cores_Quad'] = X_train['Num_cores_Quad'].map({1:0,0:1}) 
X_train['GSMGSM'] = X_train['GSMGSM'].map({1:0,0:1}) 
X_train['SIM 2_2G'] = X_train['SIM 2_2G'].map({1:0,0:1}) 
X_train['SIM 2_3G'] = X_train['SIM 2_3G'].map({1:0,0:1}) 

X_train['Brand_Gionee'] = X_train['Brand_Gionee'].map({1:0,0:1}) 
X_train['Brand_Infinix'] = X_train['Brand_Infinix'].map({1:0,0:1}) 
X_train['Brand_Lenovo'] = X_train['Brand_Lenovo'].map({1:0,0:1}) 
X_train['Brand_Nokia'] = X_train['Brand_Nokia'].map({1:0,0:1}) 
X_train['Brand_Gionee'] = X_train['Brand_Gionee'].map({1:0,0:1}) 
X_train['os_name_Android'] = X_train['os_name_Android'].map({1:0,0:1}) 
    
X_train['Brand_Lava'] = X_train['Brand_Lava'].map({1:0,0:1}) 
X_train['Brand_Moto'] = X_train['Brand_Moto'].map({1:0,0:1}) 


#For the columns below, the Mean Rating when LIKED was 0 - which means that if any one of these columns was 1 for a particular phone, 
#the phone was NEVER LIKED.

negative_properties = []
negative_properties = ['Brand_10','Brand_Coolpad','Brand_InFocus','Brand_Intex','Brand_Jivi','Brand_Karbonn','Brand_Lephone',  \
                       'Brand_Lyf','Brand_Nubia','Brand_Razer','Brand_Reliance','Brand_VOTO','Brand_iVooMi','os_name_Blackberry', \
                       'os_name_KAI','os_name_Tizen','Num_cores_Deca']


#For the columns below, the Mean Rating when DISLIKED was 0 - which means that if any one of these columns was 1 for a particular phone, 
#the phone was ALWAYS LIKED.

positive_properties = []
positive_properties = ['Brand_Apple','Brand_Comio','Brand_Google','Brand_Huawei','Brand_LeEco','Brand_Meizu','Brand_Motorola','Brand_OPPO', \
                       'Brand_OnePlus','Brand_Realme','Brand_Ulefone','POCO','os_name_iOS','Num_cores_Tru_Octa','GSMCDMA' ]



#Set the initila value of w
#Initiliase a HIGH Positive value for Positive Properties
#Initiliase a LOW Negative value for Positive Properties
#This will ensure that the model will predict correctly for the Positive and Negative properties
w_baised = np.ones(X_test.shape[1])
for column_name in negative_properties:
    neg_idx = X_train.columns.get_loc(column_name)
    w_baised[neg_idx] = -10000

for column_name in positive_properties:
    pos_idx = X_train.columns.get_loc(column_name)
    w_baised[pos_idx] = 10000


#Normalize the Train Data
min_max_scaler = MinMaxScaler(feature_range=(0, 1))
X_train_minmax  = min_max_scaler.fit_transform(X_train)            
X_train = X_train_minmax


X_train_binary= X_train
Y_train_binary= Y_train
Y_train_binary= Y_train_binary.values



#Repeate the processing for X_test

X_test['Brand_Blackberry'] = X_test['Brand_Blackberry'].map({1:0,0:1})    
X_test['Brand_HTC'] = X_test['Brand_HTC'].map({1:0,0:1})    
X_test['Brand_Micromax'] = X_test['Brand_Micromax'].map({1:0,0:1})    
X_test['Brand_Mobiistar'] = X_test['Brand_Mobiistar'].map({1:0,0:1})    
X_test['Brand_Panasonic'] = X_test['Brand_Panasonic'].map({1:0,0:1})    
X_test['Brand_Yu'] = X_test['Brand_Yu'].map({1:0,0:1})    
X_test['Sim1_3G'] = X_test['Sim1_3G'].map({1:0,0:1})  

X_test['Brand_LG'] = X_test['Brand_LG'].map({1:0,0:1}) 
X_test['Brand_Sony'] = X_test['Brand_Sony'].map({1:0,0:1}) 
X_test['Num_cores_Dual'] = X_test['Num_cores_Dual'].map({1:0,0:1}) 
X_test['Num_cores_Quad'] = X_test['Num_cores_Quad'].map({1:0,0:1}) 
X_test['GSMGSM'] = X_test['GSMGSM'].map({1:0,0:1}) 
X_test['SIM 2_2G'] = X_test['SIM 2_2G'].map({1:0,0:1}) 
X_test['SIM 2_3G'] = X_test['SIM 2_3G'].map({1:0,0:1}) 


X_test['Brand_Gionee'] = X_test['Brand_Gionee'].map({1:0,0:1}) 
X_test['Brand_Infinix'] = X_test['Brand_Infinix'].map({1:0,0:1}) 
X_test['Brand_Lenovo'] = X_test['Brand_Lenovo'].map({1:0,0:1}) 
X_test['Brand_Nokia'] = X_test['Brand_Nokia'].map({1:0,0:1}) 
X_test['Brand_Gionee'] = X_test['Brand_Gionee'].map({1:0,0:1}) 
X_test['os_name_Android'] = X_test['os_name_Android'].map({1:0,0:1}) 
    
X_test['Brand_Lava'] = X_test['Brand_Lava'].map({1:0,0:1}) 
X_test['Brand_Moto'] = X_test['Brand_Moto'].map({1:0,0:1}) 


 
min_max_scaler = MinMaxScaler(feature_range=(0, 1))
X_test_minmax  = min_max_scaler.fit_transform(X_test)            
X_test = X_test_minmax

X_test_binary= X_test


class Perceptron():
    def __init__(self):
        self.b = None
        self.w = None
    
    def model(self,x):
        return(int(np.dot(x,self.w) >= self.b))
               
        
    def predict(self,X):
        y_pred = []
        a = 0

        for itemx in X:
            temp_pred = self.model(itemx)
            y_pred.append(temp_pred)            
        return(np.array(y_pred))    
        
    
            
    def fit(self,X,Y,epochs,learn_rate,initialw):
        self.b = 0
        
        self.w = initialw
        print(initialw)
        
        accuracy = {}
        ckpt_w = ckpt_b = max_acc = 0
        for e in range(epochs):
                   
            for itemx,itemy in zip(X,Y):
                temp_pred = self.model(itemx)
                
                if temp_pred == 1 and itemy == 0:
                    self.w = self.w - learn_rate * itemx
                    self.b = self.b + learn_rate * 1
                elif temp_pred == 0 and itemy == 1:
                    self.w = self.w + learn_rate * itemx 
                    self.b - self.b - learn_rate * 1   
            accuracy[e] = accuracy_score(self.predict(X),Y)
               
            if accuracy[e] > max_acc:
                max_acc = accuracy[e]
                ckpt_w = self.w
                ckpt_b = self.b
                
        print('max acc :', max_acc)  
        print('ckptw :', ckpt_w)  
        print('ckptb :', ckpt_b)    
        
        self.w = ckpt_w
        self.b = ckpt_b
        
        plt.plot(accuracy.values())
        plt.show()
        
    
      
    
Perceptron_I = Perceptron()

Perceptron_I.fit(X_train_binary,Y_train_binary,100,.005,w_baised)
Perceptron_I.predict(X_train_binary)


#Run model for Test Data
y_test_pred = Perceptron_I.predict(X_test_binary)
print(y_test_pred)
nonzeorocnt = np.count_nonzero(y_test_pred)
print ('Zero Count :', 119 - nonzeorocnt)

submission = pd.DataFrame({'PhoneId':test_new['PhoneId']})
submission['Class'] = y_test_pred

submission = submission[['PhoneId', 'Class']]

pd.options.display.max_rows = None
print(submission)
submission.to_csv("submission.csv", index=False)



