import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing, tree, model_selection, ensemble
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree

###Wczytywanie i przygotowywanie danych
data = pd.read_csv("train.csv", sep = ",") #Wczytuje dane za pomocą modułu pandas
original_data=data
print(data.head()) #nagłówek danych
print(data.dtypes) #typy danych w kolumnach
res = data.describe().iloc[:,:5] #statystyki danych
out = res.rename( ##zmiana nazwy statystyk
    {"count":" liczba",
    "mean": "średnia",
    "std": "odch.stand."}, axis="index") 
print(out)#wypisanie odstawowych danych

print(data.isnull().sum()) #wyświetlenie pustych rekordów

labelencoder = preprocessing.LabelEncoder() #przekształcenie danych kategorycznych w numeryczne
labelencoder.fit(data["Sex"])
data["Sex"] = labelencoder.transform(data["Sex"])

y = data['Survived'] # zmienna objaśniona
data= data.drop(['PassengerId', 'Name', 'SibSp', 'Parch','Ticket','Fare','Cabin','Embarked','Survived'], axis=1) #wyrzucamy niepotrzebne dane
data = data.fillna(data.mean()) # uzupełnia pustę komórki średnią
print(data.isnull().sum())  #wyświetlenie pustych rekordów
print("Dane po zmianach: \n", data.describe())

X_train, X_test, Y_train, Y_real = model_selection.train_test_split(data,y, train_size=0.8) # przygotowywanie danych do trenowania modelu

model = tree.DecisionTreeClassifier(max_depth=3) # Tworzenie modelu drzewa
model.fit(X_train, Y_train) # dopasowanie danych do modelu

plt.figure(figsize=(8,8)) # stworzenie obszaru wykresu
graph_tree = plot_tree (model, feature_names=['Pclass','Sex','Age'],
                            class_names=['Survived','Not Survived'],
                            filled=True, rounded=True,fontsize=10) # rysowanie wykresu drzewa
plt.show()

tree_pred=model.predict(X_test)

acc = accuracy_score(Y_real, tree_pred)
print(acc)

model_rf = ensemble.RandomForestClassifier(n_estimators=200)
model_rf.fit(X_train, Y_train)
Y_pred_rf=model_rf.predict(X_test)
result_rf = pd.DataFrame({"Przezyli w rzeczywistoci": Y_real, "Przezyli w klasyfikacji": Y_pred_rf})
print(result_rf)

accuracy_rf = model_rf.score(X_test, Y_real)
print (accuracy_rf)