# Name: Pratik Jasani
# Module 4 - runs module 3's KNN algorithm

# import libraries
import streamlit as st
import random
from typing import List, Dict, Tuple, Callable

# function used to read the file and parse the data
def parse_data(file_name: str) -> List[List]:
    data = []
    file = open(file_name, "r")
    for line in file:
        datum = [float(value) for value in line.rstrip().split(",")]
        data.append(datum)
    random.shuffle(data)
    return data


# Implementation of KNN algorithm
def build_knn(database):
    def knn(k, query, debug = False):
        euc_distance = []
        
        # calculate euclidean distance
        for obs in database:
            euc_distance += [(sum([(query[i] - obs[i])**2 for i in range(len(query)-1)])**0.5, obs[-1])]
            
        euc_distance.sort(key = lambda obs: obs[0])
        
        if debug == True:
            print("Query:", query)
            for i in range(len(euc_distance[:k])):
                print(f"\t Nearest observation {i}: y = {euc_distance[i][-1]}") 
            print(f"Predicted y = {round(sum([obs[1] for obs in euc_distance[:k]])/k, 4)}")
            print("\n")
            
        return round(sum([obs[1] for obs in euc_distance[:k]])/k, 4)
    return knn

# read the file and parse the data
data = parse_data("concrete_compressive_strength.csv")

# split features in to seperate lists and save them in the dictionary
features_split = [[obs[i] for obs in data] for i in range(len(data[0])-1)]
features_dict = {"cement": features_split[0],
                 "slag" : features_split[1],
                 "ash" :  features_split[2],
                 "water": features_split[3],
                 "superplasticizer": features_split[4],
                 "coarse aggregate": features_split[5],
                 "fine aggregate": features_split[6],
                 "age" : features_split[7]}

# for each feature find the max and the min
max_min_dict = {}
for key in features_dict.keys():
    max_min_dict[key] = [int(max(features_dict[key])), int(min(features_dict[key]))]

# initialize the KNN model
knn = build_knn(data)

# print he headers
st.markdown('<h1><center> Module 4: K-Nearest Neighbours</center></h1>', unsafe_allow_html = True)
st.markdown('<h3> Concrete Compressive Strength Prediction</h3><br>', unsafe_allow_html = True)

# slider for K for KNN 
k = st.slider("Choose the value for K in K-Nearest Neighbours", 1,21, 3)

# slider for each of the features
st.markdown("<h4>Choose a value for the below features</h4>", unsafe_allow_html = True)

cement = st.slider("Cement:", max_min_dict["cement"][1],max_min_dict["cement"][0], int(max_min_dict["cement"][0]*.10) + max_min_dict["cement"][1])
slag = st.slider("Slag:", max_min_dict["slag"][1],max_min_dict["slag"][0], int(max_min_dict["slag"][0]*.10) + max_min_dict["slag"][1])
ash = st.slider("Ash:", max_min_dict["ash"][1],max_min_dict["ash"][0], int(max_min_dict["ash"][0]*.10) + max_min_dict["ash"][1])
water = st.slider("Water:", max_min_dict["water"][1],max_min_dict["water"][0], int(max_min_dict["water"][0]*.10) + max_min_dict["water"][1])
superplasticizer = st.slider("Superplasticizer:", max_min_dict["superplasticizer"][1],max_min_dict["superplasticizer"][0], int(max_min_dict["superplasticizer"][0]*.10) + max_min_dict["superplasticizer"][1])
coarse_aggregate = st.slider("Coarse aggregate:", max_min_dict["coarse aggregate"][1],max_min_dict["coarse aggregate"][0], int(max_min_dict["coarse aggregate"][0]*.10) + max_min_dict["coarse aggregate"][1])
fine_aggregate = st.slider("Fine aggregate:", max_min_dict["fine aggregate"][1],max_min_dict["fine aggregate"][0], int(max_min_dict["fine aggregate"][0]*.10) + max_min_dict["fine aggregate"][1])
age = st.slider("Age:", max_min_dict["age"][1],max_min_dict["age"][0], int(max_min_dict["age"][0]*.10) + max_min_dict["age"][1])

# predict for user's chosen quer
query = [cement, slag, ash, water, superplasticizer, coarse_aggregate, fine_aggregate, age, 0]
result = knn(k, query)

# print the table that show information about the chosen query
st.write(f"""<br> The Information below was selected:<center>
<TABLE>
   <TR>
      <TD></TD>
      <TD>Cement</TD>
      <TD>Slag</TD>
      <TD>Ash</TD>
      <TD>Water</TD>
      <TD>Superplasticizer</TD>
      <TD>Coarse Aggregate</TD>
      <TD>Fine Aggregate</TD>
      <TD>Age</TD>
   </TR>
    <TR>
      <TD>Query</TD>
      <TD>{cement}</TD>
      <TD>{slag}</TD>
      <TD>{ash}</TD>
      <TD>{water}</TD>
      <TD>{superplasticizer}</TD>
      <TD>{coarse_aggregate}</TD>
      <TD>{fine_aggregate}</TD>
      <TD>{age}</TD>
   </TR>
</TABLE></center><br><br>""", unsafe_allow_html = True)

# prin the prediction
st.write("Predicted Concrete Compressive Strength:", result)
