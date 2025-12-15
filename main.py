import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing, tree, model_selection
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree

###Wczytywanie i przygotowywanie danych
data = pd.read_csv("train.csv", sep = ",") #Wczytuje dane za pomocą modułu pandas
print(data)