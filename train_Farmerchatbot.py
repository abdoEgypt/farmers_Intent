from sklearn.metrics import accuracy_score
import xlrd     
#Ref https://www.javatpoint.com/python-read-excel-file      
# Define the location of the file     
loc = ("dataset.xls") 

book = xlrd.open_workbook("dataset.xls")
print("The number of worksheets is {0}".format(book.nsheets))
#print("Worksheet name(s): {0}".format(book.sheet_names()))
sh = book.sheet_by_index(0)
print("{0} {1} {2}".format(sh.name, sh.nrows, sh.ncols))
print("test   --------------------------")

#############################################
import nltk
words = []
labels = []
docs_x = []
docs_y = [] 

for rx in range(sh.nrows):
    if(rx>0):
        wrds = nltk.word_tokenize(sh.row(rx)[0].value)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(sh.row(rx)[1].value)

        if sh.row(rx)[1].value not in labels:
            labels.append(sh.row(rx)[1].value)

####################################################################
from nltk.stem.lancaster import LancasterStemmer
#https://www.nltk.org/_modules/nltk/stem/arlstem.html
#/Arabic stemmer
from nltk.stem.arlstem import ARLSTem
stemmer = ARLSTem()
#stemmer = LancasterStemmer()

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)
print("========================show intents")
print(labels)
training = []
output = []

out_empty = [0 for _ in range(len(labels))]
############################################################
import numpy
for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)


    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)


training = numpy.array(training)
output = numpy.array(output)

############################################################
import pickle  
with open("dataFarmer.pickle", "wb") as f:
    pickle.dump((words, labels, training, output), f)
############################################################################
import tensorflow as tf
import tflearn 

Input_Layer = tflearn.input_data(shape=[None, len(training[0])])
Layer_1 = tflearn.fully_connected(Input_Layer, 256)
Layer_2 = tflearn.fully_connected(Layer_1, 256)
Output_Layer = tflearn.fully_connected(Layer_2, len(output[0]), activation = "softmax")
net = tflearn.regression(Output_Layer)

model = tflearn.DNN(net)
################################################
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(training, output, test_size=0.33, random_state=80)
################################################

#try:
#    model.load("model.tflearn")
#except:
#################visualize######################################
loss=[]
accuracy=[]
class MonitorCallback(tflearn.callbacks.Callback):
    def __init__(self, api):
        self.my_monitor_api = api

#def on_batch_end(training_state, snapshot, log={}):    
    def on_epoch_end(self, training_state, train_index=0):
        try:
            accuracy.append( round(training_state.acc_value,3) )
            loss.append( str(training_state.loss_value) )
        except Exception as e:
            print(str(e))

#######################################################

monitorCallback = MonitorCallback(tflearn)
model.fit(X_train, y_train, n_epoch=100, show_metric=True,callbacks=monitorCallback)



model.save("model.tflearn")

##########################################################################33
# Get training and test loss histories


####################################################################
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)
#####################################################################
from time import sleep
import random
def chat():
    print("Hi, How can i help you ?")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        if results[results_index] > 0.8:
            #for tg in data["intents"]:
            #    if tg['tag'] == tag:
            #       responses = tg['responses']
            # sleep(3)
            #Bot = random.choice(responses)
           # print(Bot)
            print(tag)
        else:
            print("I don't understand!")


############################ Evaluate ###############################################################


TheAccuracy=model.evaluate(X_test,y_test)
print("TheAccuracy: ",TheAccuracy[0])       

#################print(accuracy)
#https://stackoverflow.com/questions/71726739/cnn-accuracy-plotting
###########################################visualize##########################
import matplotlib.pyplot as plt
import numpy as np
plt.plot( accuracy )

#plt.plot([1,2,8,5,6,7])
#epochs = range(1,4001)
#plt.plot(epochs, accuracy, 'g', label='Training loss')
#plt.plot( loss )
plt.xlabel("Epoch")
#plt.set_xlim(0, 1)
plt.ylabel("Accuracy")
plt.title("Relationshipt beween number of epoch and Accuracy")
plt.show()
#plt.close()


# plot


###########################################visualize##########################
chat()