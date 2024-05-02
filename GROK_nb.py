import math
def classify_nb(training_filename, testing_filename):
  X_train = [[], []] #0:'no', 1:'yes'; (dict)attributes[index]: mean, sd (standard deviation) 
  X_class_yes = []
  X_class_no = []
  X_test = []
  y_test = []
  no_attributes = 0

  #Load Training Dataset
  f = open(training_filename, 'r')
  lines = f.read().split('\n')
  no_attributes = len(lines[0].split(",")) - 1
  for i in range(no_attributes):
    X_train[0].append({'mean': 0.0, 'sd': 0.0})
    X_train[1].append({'mean': 0.0, 'sd': 0.0})
  for i in range(len(lines)):
    splitted = lines[i].split(",")
    splitted2 = [float(i) for i in splitted[:no_attributes]]
    if splitted[no_attributes] == 'yes':
      X_class_yes.append(splitted2)
      for i in range(no_attributes):
        X_train[1][i]['mean'] += splitted2[i]
    elif splitted[no_attributes] == 'no':
      X_class_no.append(splitted2)
      for i in range(no_attributes):
          X_train[0][i]['mean'] += splitted2[i]
  for i in range(no_attributes):
    X_train[0][i]['mean'] /= float(len(X_class_no))
    X_train[1][i]['mean'] /= float(len(X_class_yes))

  #Load Testing Dataset
  f = open(testing_filename, 'r')
  lines = f.read().split('\n')
  for i in range(len(lines)):
    splitted = lines[i].split(",")
    X_test.append([float(i) for i in splitted])

  #Pre-Computation of sd (Standard Deviation) for each attribute..
  for i in range(no_attributes):
    #class = 'yes'
    sd = 0.0
    mean = X_train[1][i]['mean']
    for example in X_class_yes:
      sd += (example[i]-mean)**2
    sd = (sd/float(len(X_class_yes)-1))**0.5
    X_train[1][i]['sd'] = sd
    #class = 'no'
    sd = 0.0
    mean = X_train[0][i]['mean']
    for example in X_class_no:
      sd += (example[i]-mean)**2
    sd = (sd/float(len(X_class_no)-1))**0.5
    X_train[0][i]['sd'] = sd
    
  #Prediction / Classification
  no_total_samples = float(len(X_class_no)+len(X_class_yes))
  for test in X_test:
    prediction = ''
    probabilities = [float(len(X_class_no))/no_total_samples, float(len(X_class_yes))/no_total_samples] #'no', 'yes'
    for i in range(no_attributes):
      probabilities[0] *= (1/(X_train[0][i]['sd']*(2*math.pi)**0.5))*math.exp(-((test[i]-X_train[0][i]['mean'])**2/(2*(X_train[0][i]['sd'])**2)))
      probabilities[1] *= (1/(X_train[1][i]['sd']*(2*math.pi)**0.5))*math.exp(-((test[i]-X_train[1][i]['mean'])**2/(2*(X_train[1][i]['sd'])**2)))
    if probabilities[0] > probabilities[1]:
      prediction = 'no'
    else:
      prediction = 'yes'
    y_test.append(prediction)

  return y_test