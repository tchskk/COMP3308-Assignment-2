def classify_nn(training_filename, testing_filename, k):
  
  X_train = []
  y_train = []
  X_test = []
  y_test = []
  no_attributes = 0
  
  #Load Training Dataset
  f = open(training_filename, 'r')
  lines = f.read().split('\n')
  no_attributes = len(lines[0].split(",")) - 1
  for i in range(len(lines)):
    splitted = lines[i].split(",")
    X_train.append([float(i) for i in splitted[:no_attributes]])
    y_train.append(splitted[no_attributes])
    
  #Load Testing Dataset
  f = open(testing_filename, 'r')
  lines = f.read().split('\n')
  for i in range(len(lines)):
    splitted = lines[i].split(",")
    X_test.append([float(i) for i in splitted])
  
  #Function for calculating Euclidean Distance
  def euclidean_distance(X_1, X_2):
    squared_difference_sum = 0.0
    for i in range(len(X_1)):
      squared_difference_sum += (X_1[i]-X_2[i])**2
    distance = squared_difference_sum**0.5
    return distance
  
  #Nearest Neighbour Prediction
  for test_example in X_test:
    neighbours = [] # type: class of length k
    distances = [] # (distance, class)
    for i in range(len(X_train)):
      distances.append((euclidean_distance(X_train[i], test_example), y_train[i]))
    distances.sort(key=lambda tup: tup[0])
    for i in range(k):
      neighbours.append(distances[i][1])
    prediction = max(neighbours, key=neighbours.count)
    y_test.append(prediction)
  
  return y_test