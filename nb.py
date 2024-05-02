from sklearn.model_selection import StratifiedKFold
import math
def classify_nb(X_train, y_train, X_test):
  X = [[], []] #0:'no', 1:'yes'; (dict)attributes[index]: mean, sd (standard deviation) 
  X_class_yes = []
  X_class_no = []
  y_test = []
  no_attributes = 0

  #Load Training Dataset
  no_attributes = len(X_train[0])
  for i in range(no_attributes):
    X[0].append({'mean': 0.0, 'sd': 0.0})
    X[1].append({'mean': 0.0, 'sd': 0.0})
  for i in range(len(X_train)):
    if y_train[i] == 'yes':
      X_class_yes.append(X_train[i])
      for j in range(no_attributes):
        X[1][j]['mean'] += X_train[i][j]
    elif y_train[i] == 'no':
      X_class_no.append(X_train[i])
      for j in range(no_attributes):
          X[0][j]['mean'] += X_train[i][j]
  for i in range(no_attributes):
    X[0][i]['mean'] /= float(len(X_class_no))
    X[1][i]['mean'] /= float(len(X_class_yes))

  #Pre-Computation of sd (Standard Deviation) for each attribute..
  for i in range(no_attributes):
    #class = 'yes'
    sd = 0.0
    mean = X[1][i]['mean']
    for example in X_class_yes:
      sd += (example[i]-mean)**2
    sd = (sd/float(len(X_class_yes)-1))**0.5
    X[1][i]['sd'] = sd
    #class = 'no'
    sd = 0.0
    mean = X[0][i]['mean']
    for example in X_class_no:
      sd += (example[i]-mean)**2
    sd = (sd/float(len(X_class_no)-1))**0.5
    X[0][i]['sd'] = sd
    
  #Prediction / Classification
  no_total_samples = float(len(X_class_no)+len(X_class_yes))
  for test in X_test:
    prediction = ''
    probabilities = [float(len(X_class_no))/no_total_samples, float(len(X_class_yes))/no_total_samples] #'no', 'yes'
    for i in range(no_attributes):
      probabilities[0] *= (1/(X[0][i]['sd']*(2*math.pi)**0.5))*math.exp(-((test[i]-X[0][i]['mean'])**2/(2*(X[0][i]['sd'])**2)))
      probabilities[1] *= (1/(X[1][i]['sd']*(2*math.pi)**0.5))*math.exp(-((test[i]-X[1][i]['mean'])**2/(2*(X[1][i]['sd'])**2)))
    if probabilities[0] > probabilities[1]:
      prediction = 'no'
    else:
      prediction = 'yes'
    y_test.append(prediction)

  return y_test

def nb(filename):
  X = []
  y = []

  #Load Training Dataset
  f = open(filename, 'r')
  f.readline()
  lines = f.read().split('\n')
  no_attributes = len(lines[0].split(",")) - 1
  for i in range(len(lines)):
    splitted = lines[i].split(",")
    if len(splitted) > 1:
      X.append([float(i) for i in splitted[:no_attributes]])
      y.append(splitted[no_attributes])

  kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
  ave_acc = 0.0
  for train_ix, test_ix in kfold.split(X, y):
    X_train, X_test = [X[i] for i in train_ix], [X[i] for i in test_ix]
    y_train, y_test = [y[i] for i in train_ix], [y[i] for i in test_ix]

    acc = 0.0
    predictions = classify_nb(X_train, y_train, X_test)
    for i in range(len(predictions)):
      if predictions[i] == y_test[i]:
        acc += 1
    #print(acc/float(len(X_test)))
    ave_acc += acc/float(len(X_test))

  return (ave_acc/10)*100

print(nb('pima.csv'))
print(nb('occupancy.csv'))
