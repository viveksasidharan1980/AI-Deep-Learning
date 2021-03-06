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
X_train['Brand_10'] = X_train['Brand_10.or']
X_train['GSMGSM'] = X_train['SIM Slot(s)_Dual SIM, GSM+GSM']
X_train['POCO'] = X_train['Brand_Xiaomi Poco']
X_train['GSMCDMA'] = X_train['SIM Slot(s)_Dual SIM, GSM+CDMA']
X_train['Num_cores_Tru_Octa'] = X_train['Num_cores_Tru-Octa']


#Drop the Columns copied above
X_train = X_train.drop('Brand_10.or',axis=1)
X_train = X_train.drop('SIM Slot(s)_Dual SIM, GSM+GSM',axis=1)
X_train = X_train.drop('Brand_Xiaomi Poco',axis=1)
X_train = X_train.drop('SIM Slot(s)_Dual SIM, GSM+CDMA',axis=1)
X_train = X_train.drop('Num_cores_Tru-Octa',axis=1)


#Prepare X_test 

#Setup X_test removing unwanted columns
X_test = df_phone_test.drop('PhoneId',axis=1)

#Convert RAM to same unit - GB
X_test['RAM'] = X_test['RAM'].map(lambda x: 0.5 if x > 500 else x)


#Change Column Name to avoid Syntax error where referring to these columns
X_test['Brand_10'] = X_test['Brand_10.or']
X_test['GSMGSM'] = X_test['SIM Slot(s)_Dual SIM, GSM+GSM']
X_test['POCO'] = X_test['Brand_Xiaomi Poco']
X_test['GSMCDMA'] = X_test['SIM Slot(s)_Dual SIM, GSM+CDMA']
X_test['Num_cores_Tru_Octa'] = X_test['Num_cores_Tru-Octa']


#Drop the Columns copied above
X_test = X_test.drop('Brand_10.or',axis=1)
X_test = X_test.drop('SIM Slot(s)_Dual SIM, GSM+GSM',axis=1)
X_test = X_test.drop('Brand_Xiaomi Poco',axis=1)
X_test = X_test.drop('SIM Slot(s)_Dual SIM, GSM+CDMA',axis=1)
X_test = X_test.drop('Num_cores_Tru-Octa',axis=1)


#For the columns in IF statement below, the Mean Rating when LIKED was 0 - which means that if any one of these columns was 1 for a particular phone, 
#the phone was NEVER LIKED.
#So to ensure that the Model works correctly for this scenario, the  entire row corresponding to the phone was set to 0.

for b in range(X_train.shape[0]):
  if (X_train.loc[[b]].Brand_Coolpad == 1).bool() or (X_train.loc[[b]].Brand_InFocus == 1).bool() or (X_train.loc[[b]].Brand_Intex == 1).bool()or (X_train.loc[[b]].Brand_Jivi == 1).bool()or (X_train.loc[[b]].Brand_Karbonn == 1).bool()or (X_train.loc[[b]].Brand_Lephone == 1).bool()or (X_train.loc[[b]].Brand_Lyf == 1).bool()or (X_train.loc[[b]].Brand_Nubia == 1).bool()or (X_train.loc[[b]].Brand_Razer == 1).bool()or (X_train.loc[[b]].Brand_Reliance == 1).bool()or (X_train.loc[[b]].Brand_VOTO == 1).bool()or (X_train.loc[[b]].Brand_iVooMi == 1).bool() or (X_train.loc[[b]].Num_cores_Deca == 1).bool() or (X_train.loc[[b]].os_name_Blackberry == 1).bool() or (X_train.loc[[b]].os_name_KAI == 1).bool() or (X_train.loc[[b]].os_name_Tizen == 1).bool() or (X_train.loc[[b]].Brand_10 == 1).bool(): 
    temp_x = X_train.loc[[b]] 
    X_train.loc[[b]] = np.zeros_like(temp_x)
   

#For the columns in IF statement below, the Mean Rating when DISLIKED was 0 - which means that if any one of these columns was 1 for a particular phone, 
#the phone was ALWAYS LIKED.
#So to ensure that the Model works correctly for this scenario, the  entire row corresponding to the phone was set to 1.

for b in range(X_train.shape[0]):
  if (X_train.loc[[b]].Brand_Apple == 1).bool() or (X_train.loc[[b]].Brand_Comio == 1).bool() or (X_train.loc[[b]].Brand_Google == 1).bool()or (X_train.loc[[b]].Brand_Huawei == 1).bool()or (X_train.loc[[b]].Brand_LeEco == 1).bool()or (X_train.loc[[b]].Brand_Meizu == 1).bool()or (X_train.loc[[b]].Brand_Motorola == 1).bool()or (X_train.loc[[b]].Brand_OPPO  == 1).bool()or (X_train.loc[[b]].Brand_OnePlus == 1).bool()or (X_train.loc[[b]].Brand_Realme == 1).bool()or (X_train.loc[[b]].Brand_Ulefone == 1).bool()or (X_train.loc[[b]].POCO == 1).bool() or (X_train.loc[[b]].os_name_iOS == 1).bool() or (X_train.loc[[b]].Num_cores_Tru_Octa == 1).bool() or (X_train.loc[[b]].GSMCDMA == 1).bool() :
    temp_x = X_train.loc[[b]] 
    X_train.loc[[b]] = np.ones_like(temp_x)

    
#Binarise Train Data
X_train_binary = X_train.apply(pd.cut,bins=2,labels=(0,1))
Y_train_binary = Y_train
   


#The Columns below had a inverse relationship with being LIKED. Higher the value, Higher the chance of being DISLIKED. 
#So swappping the values for the model to consider this inverse relationship

X_train_binary['Brand_Blackberry'] = X_train_binary['Brand_Blackberry'].map({1:0,0:1})    
X_train_binary['Brand_HTC'] = X_train_binary['Brand_HTC'].map({1:0,0:1})    
X_train_binary['Brand_Micromax'] = X_train_binary['Brand_Micromax'].map({1:0,0:1})    
X_train_binary['Brand_Mobiistar'] = X_train_binary['Brand_Mobiistar'].map({1:0,0:1})    
X_train_binary['Brand_Yu'] = X_train_binary['Brand_Yu'].map({1:0,0:1})    
X_train_binary['Sim1_3G'] = X_train_binary['Sim1_3G'].map({1:0,0:1})  

X_train_binary['SIM 2_2G'] = X_train_binary['SIM 2_2G'].map({1:0,0:1}) 
X_train_binary['SIM 2_3G'] = X_train_binary['SIM 2_3G'].map({1:0,0:1}) 
X_train_binary['Brand_LG'] = X_train_binary['Brand_LG'].map({1:0,0:1}) 
X_train_binary['Brand_Sony'] = X_train_binary['Brand_Sony'].map({1:0,0:1}) 
X_train_binary['Num_cores_Dual'] = X_train_binary['Num_cores_Dual'].map({1:0,0:1}) 
X_train_binary['Num_cores_Quad'] = X_train_binary['Num_cores_Quad'].map({1:0,0:1}) 
X_train_binary['GSMGSM'] = X_train_binary['GSMGSM'].map({1:0,0:1}) 
X_train_binary['Brand_Panasonic'] = X_train_binary['Brand_Panasonic'].map({1:0,0:1}) 

X_train_binary['Brand_Gionee'] = X_train_binary['Brand_Gionee'].map({1:0,0:1}) 
X_train_binary['Brand_Infinix'] = X_train_binary['Brand_Infinix'].map({1:0,0:1}) 
X_train_binary['Brand_Lenovo'] = X_train_binary['Brand_Lenovo'].map({1:0,0:1}) 
X_train_binary['Brand_Nokia'] = X_train_binary['Brand_Nokia'].map({1:0,0:1}) 
X_train_binary['os_name_Android'] = X_train_binary['os_name_Android'].map({1:0,0:1}) 

X_train_binary['Brand_Lava'] = X_train_binary['Brand_Lava'].map({1:0,0:1}) 
X_train_binary['Brand_Moto'] = X_train_binary['Brand_Moto'].map({1:0,0:1}) 


#Repeating the same Data Pre-processing for X_test

for b in range(X_test.shape[0]):
  if (X_test.loc[[b]].Brand_Coolpad == 1).bool() or (X_test.loc[[b]].Brand_InFocus == 1).bool() or (X_test.loc[[b]].Brand_Intex == 1).bool()or (X_test.loc[[b]].Brand_Jivi == 1).bool()or (X_test.loc[[b]].Brand_Karbonn == 1).bool()or (X_test.loc[[b]].Brand_Lephone == 1).bool()or (X_test.loc[[b]].Brand_Lyf == 1).bool()or (X_test.loc[[b]].Brand_Nubia == 1).bool()or (X_test.loc[[b]].Brand_Razer == 1).bool()or (X_test.loc[[b]].Brand_Reliance == 1).bool()or (X_test.loc[[b]].Brand_VOTO == 1).bool()or (X_test.loc[[b]].Brand_iVooMi == 1).bool() or (X_test.loc[[b]].Num_cores_Deca == 1).bool() or (X_test.loc[[b]].os_name_Blackberry == 1).bool() or (X_test.loc[[b]].os_name_KAI == 1).bool() or (X_test.loc[[b]].os_name_Tizen == 1).bool() or (X_test.loc[[b]].Brand_10 == 1).bool(): 
    temp_x = X_test.loc[[b]] 
    X_test.loc[[b]] = np.zeros_like(temp_x)
    #print(X_test.loc[[b]])

for b in range(X_test.shape[0]):
  if (X_test.loc[[b]].Brand_Apple == 1).bool() or (X_test.loc[[b]].Brand_Comio == 1).bool() or (X_test.loc[[b]].Brand_Google == 1).bool()or (X_test.loc[[b]].Brand_Huawei == 1).bool()or (X_test.loc[[b]].Brand_LeEco == 1).bool()or (X_test.loc[[b]].Brand_Meizu == 1).bool()or (X_test.loc[[b]].Brand_Motorola == 1).bool()or (X_test.loc[[b]].Brand_OPPO  == 1).bool()or (X_test.loc[[b]].Brand_OnePlus == 1).bool()or (X_test.loc[[b]].Brand_Realme == 1).bool()or (X_test.loc[[b]].Brand_Ulefone == 1).bool()or (X_test.loc[[b]].POCO == 1).bool() or (X_test.loc[[b]].os_name_iOS == 1).bool() or (X_test.loc[[b]].Num_cores_Tru_Octa == 1).bool() or (X_test.loc[[b]].GSMCDMA == 1).bool() :
    temp_x = X_test.loc[[b]] 
    X_test.loc[[b]] = np.ones_like(temp_x)

#Binarise Test Data  
X_test_binary = X_test.apply(pd.cut,bins=2,labels=(0,1))  

    
X_test_binary['Brand_Blackberry'] = X_test_binary['Brand_Blackberry'].map({1:0,0:1})    
X_test_binary['Brand_HTC'] = X_test_binary['Brand_HTC'].map({1:0,0:1})    
X_test_binary['Brand_Micromax'] = X_test_binary['Brand_Micromax'].map({1:0,0:1})    
X_test_binary['Brand_Mobiistar'] = X_test_binary['Brand_Mobiistar'].map({1:0,0:1})    
X_test_binary['Brand_Yu'] = X_test_binary['Brand_Yu'].map({1:0,0:1})    
X_test_binary['Sim1_3G'] = X_test_binary['Sim1_3G'].map({1:0,0:1})  

X_test_binary['SIM 2_2G'] = X_test_binary['SIM 2_2G'].map({1:0,0:1}) 
X_test_binary['SIM 2_3G'] = X_test_binary['SIM 2_3G'].map({1:0,0:1}) 
X_test_binary['Brand_LG'] = X_test_binary['Brand_LG'].map({1:0,0:1}) 
X_test_binary['Brand_Sony'] = X_test_binary['Brand_Sony'].map({1:0,0:1}) 
X_test_binary['Num_cores_Dual'] = X_test_binary['Num_cores_Dual'].map({1:0,0:1}) 
X_test_binary['Num_cores_Quad'] = X_test_binary['Num_cores_Quad'].map({1:0,0:1}) 
X_test_binary['GSMGSM'] = X_test_binary['GSMGSM'].map({1:0,0:1}) 
X_test_binary['Brand_Panasonic'] = X_test_binary['Brand_Panasonic'].map({1:0,0:1}) 

X_test_binary['Brand_Gionee'] = X_test_binary['Brand_Gionee'].map({1:0,0:1}) 
X_test_binary['Brand_Infinix'] = X_test_binary['Brand_Infinix'].map({1:0,0:1}) 
X_test_binary['Brand_Lenovo'] = X_test_binary['Brand_Lenovo'].map({1:0,0:1}) 
X_test_binary['Brand_Nokia'] = X_test_binary['Brand_Nokia'].map({1:0,0:1}) 
X_test_binary['os_name_Android'] = X_test_binary['os_name_Android'].map({1:0,0:1}) 

X_test_binary['Brand_Lava'] = X_test_binary['Brand_Lava'].map({1:0,0:1}) 
X_test_binary['Brand_Moto'] = X_test_binary['Brand_Moto'].map({1:0,0:1}) 

X_train_binary= X_train_binary.values
Y_train_binary= Y_train_binary.values

X_test_binary= X_test_binary.values

pd.options.display.max_rows = pd.options.display.max_columns = None


class MPNeuron:
    def __init__(self):
        self.b = None
    
    def model(self,x):
        return int((sum(x) >= self.b))

    def predict(self,X,b):
        self.b = b
        a = 0
        y_pred=[]
        for itemx in X:
            a = a + 1
            temp_pred = self.model(itemx)
            y_pred.append(temp_pred)
        return(np.array(y_pred))    
                    
    def fit(self,X,Y):
        max_acc = mac_acc_b = 0
        accuracy = {}
        for b in range(X.shape[1]+1):
          self.b = b
          y_pred = self.predict(X,self.b)
          accuracy[b] = accuracy_score(y_pred,Y)
          if accuracy[b] >= max_acc:
                max_acc = accuracy[b]
                max_acc_b = b
          print("Accuracy and b value : ", accuracy[b],b)
          
        print('Maximum Accuracy Achieved:',max_acc,max_acc_b)      
        return(max_acc_b)
            
           
MPN_Ins = MPNeuron()


max_acc_b = MPN_Ins.fit(X_train_binary,Y_train_binary)

print('Max Acc ',max_acc_b)


#Run model for Test data
y_test_pred = MPN_Ins.predict(X_test_binary,max_acc_b)
submission = pd.DataFrame({'PhoneId':test_new['PhoneId']})
submission['Class'] = y_test_pred

submission = submission[['PhoneId', 'Class']]

pd.options.display.max_rows = None
print(submission)
submission.to_csv("submission.csv", index=False)

