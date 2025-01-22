import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_score, GridSearchCV, RepeatedStratifiedKFold

#Made for Processing the acs income easier
#value: indicates column
#ranges: indicates dictionary of values
'''group_dict = {}
def assign_group(value):
  """Assigns a group label to a value based on the ranges dictionary."""
  if value not in group_dict:
    for group_name, group_range in ranges.items():
      if group_range[0] <= value <= group_range[1]:
        group_dict[value] = group_name
        break
  return group_dict.get(value)'''

def preprocess_german(df, preprocess):
    df['status'] = df['status'].map({'A11': 0, 'A12': 1, 'A13': 2, 'A14': 3}).astype(int)
    df['credit_hist'] = df['credit_hist'].map({'A34': 0, 'A33': 1, 'A32': 2, 'A31': 3, 'A30': 4}).astype(int)

    df['savings'] = df['savings'].map({'A61': 0, 'A62': 1, 'A63': 2, 'A64': 3, 'A65': 4}).astype(int)
    df['employment'] = df['employment'].map({'A71': 0, 'A72': 1, 'A73': 2, 'A74': 3, 'A75': 4}).astype(int)    
    df['gender'] = df['personal_status'].map({'A91': 1, 'A92': 0, 'A93': 1, 'A94': 1, 'A95': 0}).astype(int)
    df['debtors'] = df['debtors'].map({'A101': 0, 'A102': 1, 'A103': 2}).astype(int)
    df['property'] = df['property'].map({'A121': 3, 'A122': 2, 'A123': 1, 'A124': 0}).astype(int)        
    df['install_plans'] = df['install_plans'].map({'A141': 1, 'A142': 1, 'A143': 0}).astype(int)
    if preprocess:
        df = pd.concat([df, pd.get_dummies(df['purpose'], prefix='purpose')],axis=1)
        df = pd.concat([df, pd.get_dummies(df['housing'], prefix='housing')],axis=1)
        df.loc[(df['credit_amt'] <= 2000), 'credit_amt'] = 0
        df.loc[(df['credit_amt'] > 2000) & (df['credit_amt'] <= 5000), 'credit_amt'] = 1
        df.loc[(df['credit_amt'] > 5000), 'credit_amt'] = 2    
        df.loc[(df['duration'] <= 12), 'duration'] = 0
        df.loc[(df['duration'] > 12) & (df['duration'] <= 24), 'duration'] = 1
        df.loc[(df['duration'] > 24) & (df['duration'] <= 36), 'duration'] = 2
        df.loc[(df['duration'] > 36), 'duration'] = 3
        df['age'] = df['age'].apply(lambda x : 1 if x >= 45 else 0) # 1 if old, 0 if young
    df['job'] = df['job'].map({'A171': 0, 'A172': 1, 'A173': 2, 'A174': 3}).astype(int)    
    df['telephone'] = df['telephone'].map({'A191': 0, 'A192': 1}).astype(int)
    df['foreign_worker'] = df['foreign_worker'].map({'A201': 1, 'A202': 0}).astype(int)

    return df


def process_adult(df):
    # replace missing values (?) to nan and then drop the columns
    df['country'] = df['country'].replace(' ?',np.nan)
    df['workclass'] = df['workclass'].replace(' ?',np.nan)
    df['occupation'] = df['occupation'].replace(' ?',np.nan)
    # dropping the NaN rows now
    df.dropna(how='any', inplace=True)
    df['income'] = df['income'].map({' <=50K': 0, ' >50K': 1})
    df['age'] = df['age'].apply(lambda x : 1 if x >= 45 else 0) # 1 if old, 0 if young
    df['workclass'] = df['workclass'].map({' Never-worked': 0, ' Without-pay': 1, ' State-gov': 2, ' Local-gov': 3, ' Federal-gov': 4, ' Self-emp-inc': 5, ' Self-emp-not-inc': 6, ' Private': 7})
    df['education'] = df['education'].map({' Preschool': 0, ' 1st-4th': 1, ' 5th-6th': 2, ' 7th-8th': 3, ' 9th': 4, ' 10th': 5, ' 11th': 6, ' 12th': 7, ' HS-grad':8, ' Some-college': 9, ' Bachelors': 10, ' Prof-school': 11, ' Assoc-acdm': 12, ' Assoc-voc': 13, ' Masters': 14, ' Doctorate': 15}).astype(int)
    df['marital'] = df['marital'].map({' Married-civ-spouse': 2, ' Divorced': 1, ' Never-married': 0, ' Separated': 1, ' Widowed': 1, ' Married-spouse-absent': 2, ' Married-AF-spouse': 2}).astype(int)
    df['relationship'] = df['relationship'].map({' Wife': 1 , ' Own-child': 0 , ' Husband': 1, ' Not-in-family': 0, ' Other-relative': 0, ' Unmarried': 0}).astype(int)
    df['race'] = df['race'].map({' White': 1, ' Asian-Pac-Islander': 0, ' Amer-Indian-Eskimo': 0, ' Other': 0, ' Black': 0}).astype(int)
    df['gender'] = df['gender'].map({ ' Male': 1, ' Female': 0})
    # process hours
    df.loc[(df['hours'] <= 40), 'hours'] = 0
    df.loc[(df['hours'] > 40), 'hours'] = 1
    df = df.drop(columns=['fnlwgt', 'education.num', 'occupation', 'country', 'capgain', 'caploss'])
    df = df.reset_index(drop=True)
    return df



def preprocess_compas(df):
    df['age_cat'] = df['age_cat'].map({'Less than 25': 0, '25 - 45': 1, 'Greater than 45': 2}).astype(int)    
    df['score_text'] = df['score_text'].map({'Low': 0, 'Medium': 1, 'High': 2}).astype(int)    
    df['race'] = df['race'].map({'Other': 0, 'African-American': 0, 'Hispanic': 0, 'Native American': 0, 'Asian': 0, 'Caucasian': 1}).astype(int)
    df['sex'] = df['sex'].map({'Male': 1, 'Female': 0}).astype(int)    
    
    df.loc[(df['priors_count'] <= 5), 'priors_count'] = 0
    df.loc[(df['priors_count'] > 5) & (df['priors_count'] <= 15), 'priors_count'] = 1
    df.loc[(df['priors_count'] > 15), 'priors_count'] = 2
    
    df.loc[(df['juv_fel_count'] == 0), 'juv_fel_count'] = 0
    df.loc[(df['juv_fel_count'] == 1), 'juv_fel_count'] = 1
    df.loc[(df['juv_fel_count'] > 1), 'juv_fel_count'] = 2
    
    df.loc[(df['juv_misd_count'] == 0), 'juv_misd_count'] = 0
    df.loc[(df['juv_misd_count'] == 1), 'juv_misd_count'] = 1
    df.loc[(df['juv_misd_count'] > 1), 'juv_misd_count'] = 2
    
    df.loc[(df['juv_other_count'] == 0), 'juv_other_count'] = 0
    df.loc[(df['juv_other_count'] == 1), 'juv_other_count'] = 1
    df.loc[(df['juv_other_count'] > 1), 'juv_other_count'] = 2
    return df

def preprocess_acsincome(df):
    #Alter the Class of Worker column
    #df['COW'] = df['COW'].map({1:'private_business', 2:'non-profit', 3:'local_gov', 4:'state_gov', 5:'federal_gov', 6:'SE_no_business', 7:'SE_business', 8:'no_pay_work', 9:'unemployed'})
    #Alter the Marital status column
    df['MAR'] = df['MAR'].map({1:'married', 2:'widowed', 3:'divorced', 4:'seperated', 5:'never_married'})
    #Alter the sex column
    df['SEX'] = df['SEX'].map({1:'male', 2:'female'})
    
     # Define the ranges dictionary for SCHL
    ranges = {'GS': (1, 17), 'BD': (18, 21), 'HE': (22, 24)}
    # Function to map the integer values to strings based on the ranges
    def map_values(value):
        for key, (start, end) in ranges.items():
            if start <= value <= end:
                return key
        return None 
    # Apply the function to the 'SCHL' column
    df['SCHL'] = df['SCHL'].apply(map_values)

    #Alter the Occupation column
    ranges = {'business': (10,960),'stem': (1005,1560),'life': (1600,3550),'sales': (3601,5940),
              'infrastructure': (6005,9760),'military': (9800,9920)
              }
    
    df['OCCP'] = df['OCCP'].apply(map_values)
    #Alter the Class of Worker column
    ranges = {'private_business':(1,1), 'non-profit':(2,2), 'gov':(3,5), 'self_employed':(6,7), 'no_income':(8,9)}
    df['COW'] = df['COW'].apply(map_values)
    #Alter the Relationship codes
    ranges = {'family': (0,10),'non-family': (11,17)}
    df['RELP'] = df['RELP'].apply(map_values)
    #Alter the race column
    ranges = {'white': (1,1),'poc': (2,9)}
    df['RAC1P'] = df['RAC1P'].apply(map_values)
    #Alter the income column
    ranges = {0:(0,50000),1:(50001,2000000)}
    df['PINCP'] = df['PINCP'].apply(map_values)
    #Alter the age column
    df['AGEP'] = df['AGEP'].apply(lambda x : 'old' if x >= 45 else 'young')

    #Get dummy columns
    #df = pd.concat([df, pd.get_dummies(df['housing'], prefix='housing')],axis=1)
    #Dummies for COW column
    df = pd.concat([df, pd.get_dummies(df['COW'],prefix='COW')],axis=1)
    #Dummies for MAR column
    df = pd.concat([df,pd.get_dummies(df['MAR'],prefix='MAR')],axis=1)
    #Dummies for SEX column
    df = pd.concat([df,pd.get_dummies(df['SEX'],prefix='SEX')],axis=1)
    #Dummies for SCHL column
    df = pd.concat([df,pd.get_dummies(df['SCHL'],prefix='SCHL')],axis=1)
    #Dummies for OCCP Column
    df = pd.concat([df,pd.get_dummies(df['OCCP'],prefix='OCCP')],axis=1)
    #Dummies for RELP column
    df = pd.concat([df,pd.get_dummies(df['RELP'],prefix='RELP')],axis=1)
    #Dummies for RAC1P column
    df = pd.concat([df,pd.get_dummies(df['RAC1P'],prefix='RAC1P')],axis=1)
    #Dummies for AGEP
    df = pd.concat([df,pd.get_dummies(df['AGEP'],prefix='AGEP')],axis=1)

    return df


def load_german(preprocess=True):
    cols = ['status', 'duration', 'credit_hist', 'purpose', 'credit_amt', 'savings', 'employment',\
            'install_rate', 'personal_status', 'debtors', 'residence', 'property', 'age', 'install_plans',\
            'housing', 'num_credits', 'job', 'num_liable', 'telephone', 'foreign_worker', 'credit']
    df = pd.read_table('german.data', names=cols, sep=" ", index_col=False)
    df['credit'] = df['credit'].replace(2, 0) #1 = Good, 2= Bad credit risk
    y = df['credit']
    df = preprocess_german(df, preprocess)
    if preprocess:
        df = df.drop(columns=['purpose', 'personal_status', 'housing', 'credit'])
    else:
        df = df.drop(columns=['personal_status', 'credit'])
    
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=1)
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    return X_train, X_test, y_train, y_test


def load_adult(sample=False):
    cols = ['age', 'workclass', 'fnlwgt', 'education', 'education.num', 'marital', 'occupation',\
            'relationship', 'race', 'gender', 'capgain', 'caploss', 'hours', 'country', 'income']
    if sample:
        df_train = pd.read_csv('adult-sample-train-10pc', names=cols, sep=",")
        df_test = pd.read_csv('adult-sample-test-10pc', names=cols, sep=",")
    else:
        df_train = pd.read_csv('adult.data', names=cols, sep=",")
        df_test = pd.read_csv('adult.test', names=cols, sep=",")

    df_train = process_adult(df_train)
    df_test = process_adult(df_test)

    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    X_train = df_train.drop(columns='income')
    y_train = df_train['income']

    X_test = df_test.drop(columns='income')
    y_test = df_test['income']
    return X_train, X_test, y_train, y_test

def load_acsincome(sample_size):
    cols = ['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP', 'SEX', 'RAC1P', 'ST', 'PINCP']
    df_income = pd.read_csv('acs_income.csv' ,index_col=None, sep=',')
    #Grabs datapoints that are from Florida
    df_cali = df_income[df_income['ST']==6.0]
    #print('1')
    df_process = preprocess_acsincome(df_cali)
    df_process = df_process.sample(n=sample_size)
    df_process=df_process.drop(columns=['COW','MAR','SEX','SCHL','OCCP','RELP','RAC1P','AGEP'])
    y=df_process['PINCP']
    df_process=df_process.drop(columns='PINCP')
    #df_final = df_process.sample(n=sample_size)
    X_train, X_test, y_train, y_test = train_test_split(df_process, y, test_size=0.2, random_state=1)

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    return X_train, X_test, y_train, y_test


def load_compas():
    df = pd.read_csv('compas-scores-two-years.csv')
    df = df[['event', 'is_violent_recid', 'is_recid', 'priors_count', 'juv_other_count',\
             'juv_misd_count', 'juv_fel_count', 'race', 'age_cat', 'sex','score_text']]
    df = preprocess_compas(df)

    y = df['is_recid']
    # y = df['is_violent_recid']
    df = df.drop(columns=['is_recid', 'is_violent_recid'])

    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=1)
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    return X_train, X_test, y_train, y_test


def load_traffic():
    df = pd.read_csv('traffic_violations_cleaned.csv')
    y = df['search_outcome']
    df = df.drop(columns=['search_outcome'])
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=1)
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    return X_train, X_test, y_train, y_test


def load_sqf():
    df_train = pd.read_csv('sqf_train.csv')
    y_train = df_train['frisked']
    X_train = df_train.drop(columns=['frisked'])
    
    df_test = pd.read_csv('sqf_test.csv')
    y_test = df_test['frisked']
    X_test = df_test.drop(columns=['frisked'])
    
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    return X_train, X_test, y_train, y_test


def load(dataset, preprocess=True, row_num=10000, attr_num=30, sample=False):
    if dataset == 'compas':
        return load_compas()
    elif dataset == 'adult':
        return load_adult(sample=sample)
    elif dataset == 'german':
        return load_german(preprocess)
    elif dataset == 'traffic':
        return load_traffic()
    elif dataset == 'sqf':
        return load_sqf()
    elif dataset == 'random':
        return generate_random_dataset(row_num, attr_num)
    elif dataset == 'acsincome':
        row_num=20000
        return load_acsincome(row_num)
    else:
        raise NotImplementedError
        
        
def generate_random_dataset(row_num, attr_num):
    cols_ls = list()
    for attr_idx in range(attr_num):
        col = np.random.binomial(n=1, p=0.5, size=(row_num, 1))
        cols_ls.append(col)
    X_mat = np.concatenate(cols_ls, axis=1)
    noise = np.random.binomial(n=2, p=0.03, size=(row_num, 1))
    random_coef = np.random.random(attr_num)
    
    X = pd.DataFrame(X_mat, columns=[f'A{attr_idx}' for attr_idx in range(attr_num)])
    y = pd.Series(np.where((np.dot(X_mat, random_coef.reshape(-1, 1))+noise)>attr_num*0.25, 1, 0).ravel(), name='foo')
    X['AA'] = np.random.binomial(n=1, p=0.5, size=(row_num, 1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    return X_train, X_test, y_train, y_test
