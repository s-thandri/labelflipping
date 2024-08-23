import pandas as pd
class StatisticalParity:
    def __init__(self, test_data, predictors, prediction_df, sensitive_attribute, sensitive_attribute_cutoff, concat_col):
        #DF of the x_test info (or whatever it is at the time)
        self.test_data = test_data
        #list of the independent variables
        self.predictors = predictors
        #Labels in DF form
        self.prediction_df = prediction_df
        #Dependent Attribute
        self.sensitive_attribute = sensitive_attribute
        #Cut off value
        self.sensitive_attribute_cutoff = sensitive_attribute_cutoff
        #Name of the column I am creating the DF on
        self.concat_col = concat_col
    
    def calc_statiscal_parity(self):
        #Function used to calculate the statistical parity
        test_demo_df = pd.DataFrame(self.test_data, columns = self.predictors)
        predicted_df = pd.DataFrame(self.prediction_df, columns = [self.concat_col])
        concat_df = pd.concat([test_demo_df,predicted_df], axis=1)

        #Get the two groups of people total
        total_unpriv = (len(concat_df[concat_df[self.sensitive_attribute]<self.sensitive_attribute_cutoff]))
        total_priv = (len(concat_df[concat_df[self.sensitive_attribute]>=self.sensitive_attribute_cutoff]))

         #Number of people accepted
        total_credit_unpriv = len(concat_df[(concat_df[self.concat_col] == 1) & (concat_df[self.sensitive_attribute] < self.sensitive_attribute_cutoff)])
        total_credit_priv = len(concat_df[(concat_df[self.concat_col] == 1) & (concat_df[self.sensitive_attribute] >= self.sensitive_attribute_cutoff)])

        #Percentage of approved people
        p_unpriv = total_credit_unpriv/total_unpriv
        p_priv = total_credit_priv/total_priv


        #Calculate the parity
        parity = p_priv - p_unpriv


        return parity
    

