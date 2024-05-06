from sklearn.model_selection import StratifiedKFold
def classify_nn(training_filename, k):
  
  X = []
  y = []
  tp = 0
  fp = 0
  tn = 0
  fn = 0
  no_attributes = 0
  
  #Load Training Dataset
  f = open(training_filename, 'r')
  f.readline()
  lines = f.read().split('\n')
  no_attributes = len(lines[0].split(",")) - 1
  for i in range(len(lines)):
    splitted = lines[i].split(",")
    if len(splitted) > 1:
      X.append([float(i) for i in splitted[:no_attributes]])
      y.append(splitted[no_attributes])
  
  #Function for calculating Euclidean Distance
  def euclidean_distance(X_1, X_2):
    squared_difference_sum = 0.0
    for i in range(len(X_1)):
      squared_difference_sum += (X_1[i]-X_2[i])**2
    distance = squared_difference_sum**0.5
    return distance
  
  #Nearest Neighbour Prediction
  def predict(X_train, y_train, X_test, y_test):
    acc = 0.0
    for i in range(len(X_test)):
      neighbours = [] # type: class of length k
      distances = [] # (distance, class)
      for j in range(len(X_train)):
        distances.append((euclidean_distance(X_train[j], X_test[i]), y_train[j]))
      distances.sort(key=lambda tup: tup[0])
      for j in range(k):
        neighbours.append(distances[j][1])
      prediction = max(neighbours, key=neighbours.count)
      if prediction == y_test[i]:
        acc += 1
      nonlocal tp, fp, fn, tn
      if prediction == 'yes':
        if y_test[i] == 'yes':
          tp += 1
        elif y_test[i] == 'no':
          fp += 1
      elif prediction == 'no':
        if y_test[i] == 'yes':
          fn += 1
        elif y_test[i] == 'no':
          tn += 1
    #print(acc/float(len(X_test)))
    return acc/float(len(X_test))
  
  kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
  ave_acc = 0.0
  for train_ix, test_ix in kfold.split(X, y):
    X_train, X_test = [X[i] for i in train_ix], [X[i] for i in test_ix]
    y_train, y_test = [y[i] for i in train_ix], [y[i] for i in test_ix]

    ave_acc += predict(X_train, y_train, X_test, y_test)
  print(f"True Positive : {tp}")
  print(f"False Positive : {fp}")
  print(f"True Negative : {tn}")
  print(f"False Negative : {fn}")
  print(f"Precision : {tn/(tn+fn)}")
  print(f"Recall : {tn/(tn+fp)}")
  return (ave_acc/10)*100

print(f"Accuracy : {classify_nn('pima.csv', 1)}")
print(f"Accuracy : {classify_nn('occupancy.csv', 1)}")
