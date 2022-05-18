import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from gensim.parsing.preprocessing import remove_stopwords
from sklearn.decomposition import NMF,TruncatedSVD

# Find files
pathList = [] # os.listdir('Journals/')
for file in os.listdir('Journals/'):
    if file.endswith(".txt"):
        pathList.append(file)
docDFList = {}

for path in pathList:
    docDFList[path] = pd.DataFrame([{""}])
    docDFList[path] = docDFList[path].drop(0)

# Load and pre process datas

for path in docDFList:
    df = open('Journals/' + path)
    lines = df.readlines()
    df.close()
    for index, line in enumerate(lines):

        lines[index] = line.strip() # sup des retours chariots etc..
        lines[index] = lines[index].lower()
        lines[index] = re.sub(r'[0-9]+', '', lines[index]) # suppression des chiffres
        lines[index] = re.sub(r'[^\w\s]','',lines[index]) # ponctuation
        lines[index] = re.sub(r'[^a-z][a-z]{1}[^a-z]','',lines[index]) # signle letter in a line
        lines[index] = re.sub(r'^[a-z]{1}[^a-z]','',lines[index]) # signle letter start of line
        lines[index] = re.sub(r'[^a-z][a-z]{1}$','',lines[index]) # signle letter end of line
        lines[index] = remove_stopwords(lines[index])

        df = pd.DataFrame(lines[index].split()) # lower case + séparation ligne mot
        docDFList[path] = pd.concat([docDFList[path],df],ignore_index=True)
    docDFList[path].columns = ['words']
    docDFList[path].dropna()
    # print(path)
    # print(docDFList[path])

# Matrix factorization

factMatDF = pd.DataFrame([{""}])
factMatDF = factMatDF.drop(0)
for path in docDFList:
    wordCountDoc = pd.DataFrame(docDFList[path].words.str.split(expand=True).stack().value_counts())
    wordCountDoc.columns = [path]
    wordCountDoc.to_csv('Res/'+ path +'Count.csv')
    if int(factMatDF.size) > 0 :
        factMatDF = factMatDF.join(wordCountDoc, sort=True)
    else :
        factMatDF = wordCountDoc
factMatDF = factMatDF.fillna(0)
print("Factorized matrix :")
print(factMatDF)
factMatNP = factMatDF.to_numpy()
# factMatDF.to_csv('Res/factMatDF.csv')

model = NMF(n_components=2, init='random', random_state=0)
modelLSA  = TruncatedSVD(n_components=2, n_iter=7, random_state=42)

W = model.fit_transform(factMatNP)
H = model.components_
WH= W.dot(H)
# print("Matrice U/W : \n",W)
# print("\nMatrice V/H : \n",H)

U = modelLSA.fit_transform(factMatNP)
V = model.components_
UsV = modelLSA.inverse_transform(U)

print("WH")
print(WH)

print("UsV")
print(UsV)


#MSE Calcul

print("MSE NMF : ",((factMatNP-WH)**2).mean())
print("MSE LSA : ",((factMatNP-UsV)**2).mean())

# Scatter plots of Topic relation

plt.scatter(H[0], H[1], s=1.5)
plt.title("Topic 0 Topic 1 : docs (NMF)")
plt.xlabel("Topic 0")
plt.ylabel("Topic 1")
plt.show()

plt.scatter(*zip(*W), s=1.5)
plt.title("Topic 0 Topic 1 : words (NMF)")
plt.xlabel("Topic 0")
plt.ylabel("Topic 1")
plt.show()

plt.scatter(V[0], V[1], s=1.5)
plt.title("Topic 0 Topic 1 : docs (LSA)")
plt.xlabel("Topic 0")
plt.ylabel("Topic 1")
plt.show()

plt.scatter(*zip(*U), s=1.5)
plt.title("Topic 0 Topic 1 : words (LSA)")
plt.xlabel("Topic 0")
plt.ylabel("Topic 1")
plt.show()