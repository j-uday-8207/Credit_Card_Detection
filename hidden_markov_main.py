import numpy as np

from clustering import KMeansClustering
from hidden_markov_model import HMM
import random
from config import *
import pandas as pd 
# import numpy as np

def get_F1Score(TP,FP,TN,FN):
    # return 1,1,1
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    F1_score = (2*precision*recall)/(precision+recall)
    # print(f"precision is {precision}")
    # print(f"recall is {recall}")
    # print(f"F1 Score is {(2*precision*recall)/(precision + recall)}")
    return precision,recall ,F1_score
    
def get_metrics(df):
    # Read the CSV file into a DataFrame
    # df = pd.read_csv(input_file)
    
    # Initialize counters for TP, FP, TN, FN
    TP = FP = TN = FN = 0
    
    # Iterate over each row in the DataFrame
    random_number = random.randint(0, len(observations)-CHUNK_SIZE)
    base_obs_seq = observations[random_number:random_number+CHUNK_SIZE]
    # i = 0 
    for index, row in df.iterrows():
        actual_class = row['is_fraud']
        predicted_class = 0
        # i+=1
        # if(i==10):
        #     break  
        new_transaction = k.predict(int(row['amt']))
        new_observation = np.append(base_obs_seq[1:], [new_transaction])
        if h.detect_fraud(base_obs_seq, new_observation, THRESHOLD):
            predicted_class=1
        else :
            base_obs_seq = new_observation
            # print('Fraud')
       
        if actual_class == 1 and predicted_class == 1:
            TP += 1
        elif actual_class == 0 and predicted_class == 1:
            FP += 1
        elif actual_class == 0 and predicted_class == 0:
            TN += 1
        elif actual_class == 1 and predicted_class == 0:
            FN += 1

    return TP, FP, TN, FN

# Example usage:

# Example usage:
# get_input('input.csv', 'output.txt')
# Example usage:
# get_input('input.txt', 'output.txt')

# Example usage:
# get_input('input.txt', 'output.txt')

# def get_input():
#     while True:
#         new_transaction = input('Please add your new transaction : ')
#         if int(new_transaction) == TERMINATE:
#             print("You have successfully terminated the process")
#             break
#         new_transaction = k.predict(int(new_transaction))
#         new_observation = np.append(observations[1:], [new_transaction])

#         if h.detect_fraud(observations, new_observation, THRESHOLD):
#             print('Fraud')
#         else:
#             print('Normal')




if __name__ == '__main__':
    # d = Driver('./Data/train_kaggle.txt')
    
    df = pd.read_csv("fraudTrain.csv")
    df['amt'] = df['amt'].round().astype(int)
    df = df[df['is_fraud'] == 0]
    columns_to_keep = ['cc_num', 'amt','is_fraud']
    df_filtered = df[columns_to_keep]
    grouped = df_filtered.groupby('cc_num')
    # Creating a dictionary of DataFrames, where keys are cc_num
    
    dfs = {group: grouped.get_group(group) for group in grouped.groups}
    df_test = pd.read_csv("fraudTest.csv")
    df_test['amt'] = df_test['amt'].round().astype(int)
    # df_test = df_test[df_test['is_fraud'] == 0]
    columns_to_keep = ['cc_num', 'amt','is_fraud']
    df_test_filtered = df_test[columns_to_keep]
    grouped = df_test_filtered.groupby('cc_num')
    # Creating a dictionary of DataFrames, where keys are the unique values in column1
    df_tests = {group: grouped.get_group(group) for group in grouped.groups}
    p = 0
    
    i= 0
    possible_thresholds = [0.3,0.5,0.7,0.9]
    for curr_threshold in possible_thresholds:
        TP_final = 0 
        FP_final = 0 
        TN_final = 0 
        FN_final = 0 
        THRESHOLD = curr_threshold
        print(THRESHOLD)
        for card_number, df in dfs.items():
            
        # Extract the transaction amounts column as a numpy array
            if(card_number not in df_tests):
                continue
            # p+=1
            if(df_tests[card_number]['is_fraud'].sum()<=14):
                continue
            # i+=1 
            # if(i==10):
            #     break 
            # if(p==3):
            #     break
            transaction_amounts = df['amt'].values
            h = HMM(n_states=STATES, n_possible_observations=CLUSTERS)
            k = KMeansClustering()
            observations = k.run(transaction_amounts)
            h.train_model(observations=list(observations), steps=STEPS)
            TP, FP, TN, FN = get_metrics(df_tests[card_number])
            print(f"{TP} {FP} {TN} {FN} \n")
            TP_final += TP
            FP_final += FP
            TN_final += TN
            FN_final += FN
            # with open('out.txt', 'a') as f:
            #     print('Filename:', "result.txt", file=f)
            
        print(f"TP : {TP_final} FP: {FP_final} TN: {TN_final} FN: {FN_final}")
        precision,recall,F1_score = get_F1Score(TP_final,FP_final,TN_final,FN_final)
        print(f"threshold was {THRESHOLD}  : F1_score is {F1_score}  , precision is {precision} , recall is {recall}")
    # get_F1Score(65,335,14033,216)
    # You can then use this numpy array for training your model
    # For example, let's print the card number and the transaction amounts for demonstration
        # print("Card Number:", card_number)
        # print("Transaction Amounts:", transaction_amounts)
        # print()
    # df1 = pd.read_csv("/kaggle/input/fraud-detection/fraudTrain.csv")

    # get_input('./Data/test_kaggle.txt','./Data/prediction.txt')
    # print("True Positives:", TP)
    # print("False Positives:", FP)
    # print("True Negatives:", TN)
    # print("False Negatives:", FN)
