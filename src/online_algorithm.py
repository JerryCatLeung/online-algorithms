import numpy as np
import matplotlib.pyplot as plt

class OnlineAlgorithm:
  
  def __init__(self, training_data):
    self.training_data = training_data
    [self.num_examples, self.num_columns] = self.training_data.shape
    self.num_features = self.num_columns - 1
    self.subsample_size = 0.1*self.num_examples

    self.training_examples = self.training_data[:,:self.num_features]
    self.training_labels = self.training_data[:,self.num_features]
    
    self.d1 = self.training_examples[:self.subsample_size,:]
    self.d2 = self.training_examples[self.subsample_size:2*self.subsample_size,:] 
  
    self.training_labels_d2 = self.training_labels[self.subsample_size:2*self.subsample_size]
    
    self.learning_rates = [1.5,0.25,0.03,0.005,0.001]
    self.promotion_parameters = [1.1,1.01,1.005,1.0005,1.0001]

  def update(self):
    #implemented by derived classes
    print "update"

  def update_condition(self):
    #implemented by derived classes
    print "update_condition"

  def run(self):
    for i in range(0,20):
      for k, example in enumerate(self.d1):
        label = self.training_labels[k]
        if self.update_condition(example,label):
          self.update(example,label)
    return self.w

  def run_plot(self, N):
    mistake_list = []
    curr_iter = 1
    num_mistakes = 0
    for i in range(0,N):
      for k, example in enumerate(self.training_examples):
        label = self.training_labels[k]
        if np.sign(np.dot(self.w, example)+self.thetha) != label:
          num_mistakes += 1
        if self.update_condition(example,label):
          self.update(example,label)
        mistake_list.append((curr_iter, num_mistakes)) 
        curr_iter += 1
    return mistake_list

  def run_best(self, output_file):
    f = open(output_file, 'w+')
    num_mistakes = 0.0
    for k,example in enumerate(self.training_examples):
      label = self.training_labels[k]
      if np.sign(np.dot(self.w, example)+self.thetha) != label:
        num_mistakes += 1
      if self.update_condition(example,label):
        self.update(example,label)
    out = "Total number of mistakes in training: {}".format(num_mistakes)
    f.write(out)

  def run_best_20(self):
    for i in range(0,20):
      for k, example in enumerate(self.training_examples):
        label = self.training_labels[k]
        if self.update_condition(example,label):
          self.update(example,label)
    return (self.w, self.thetha)

  def mistakes_until_r_pefect(self, R):
    #compute the number of mistakes until R correct in a row
    curr_streak = 0
    num_mistakes = 0
    for i in range(0,self.num_examples):
      if (curr_streak == R):
        break
      label = self.training_labels[i]
      example = self.training_examples[i]
      if np.sign(np.dot(self.w, example)+self.thetha) != label:
        num_mistakes += 1
        curr_streak = 0
      else:
        curr_streak += 1
      if self.update_condition(example,label):
        self.update(example,label)
      if i == self.num_examples - 1:
        i = 0
    return num_mistakes

  def evaluate(self):
    #implemented by derived classes
    print "evaluate"

class Perceptron(OnlineAlgorithm):

  def __init__(self,training_data):
    OnlineAlgorithm.__init__(self, training_data)
    self.thetha = 0
    self.learning_rate = 1
    self.w = np.zeros(self.num_features) 
  
  def update_condition(self, example, label):
    pred = np.sign(np.dot(self.w,example)+self.thetha)
    return pred != label

  def update(self,example,label):
    self.w += self.learning_rate*label*example
    self.thetha += self.learning_rate*label

class PerceptronMargin(OnlineAlgorithm):
  
  def __init__(self, training_data, learning_rate):
    OnlineAlgorithm.__init__(self, training_data)
    self.thetha = 0
    self.learning_rate = learning_rate
    self.w = np.zeros(self.num_features) 
    self.margin = 1
 
  def update_condition(self, example, label):
    pred = np.dot(self.w,example)+self.thetha
    return pred*label <= self.margin

  def update(self, example, label):
    self.w += self.learning_rate*label*example
    self.thetha += self.learning_rate*label

class Winnow(OnlineAlgorithm):
  
  def __init__(self, training_data, promotion_param):
    OnlineAlgorithm.__init__(self, training_data)
    self.promotion_parameter = promotion_param
    self.thetha = -1*self.num_features
    self.w = np.ones(self.num_features)

  def update_condition(self, example, label):
    pred = np.sign(np.dot(self.w,example)+self.thetha)
    return pred != label

  def update(self, example, label):
    alpha_vector = np.empty(self.num_features)
    alpha_vector.fill(self.promotion_parameter)
    alpha_exp = np.power(alpha_vector, label*example)
    self.w = np.multiply(self.w,alpha_exp) 

class WinnowMargin(OnlineAlgorithm):

  def __init__(self, training_data, promotion_param, margin):
    OnlineAlgorithm.__init__(self, training_data)
    self.promotion_parameter = promotion_param
    self.margin = margin
    self.thetha = -1*self.num_features
    self.w = np.ones(self.num_features)
  
  def update_condition(self, example, label):
    pred = np.dot(self.w,example)+self.thetha
    return pred*label <= self.margin

  def update(self, example, label):
    alpha_vector = np.empty(self.num_features)
    alpha_vector.fill(self.promotion_parameter)
    alpha_exp = np.power(alpha_vector, label*example)
    self.w = np.multiply(self.w,alpha_exp) 

class AdaGrad(OnlineAlgorithm):

  def __init__(self, training_data, learning_rate):
    OnlineAlgorithm.__init__(self, training_data)
    self.learning_rate = learning_rate
    self.w = np.zeros(self.num_features)
    self.thetha = 0.
    self.G_thetha = 10**-10
    self.G = np.empty(self.num_features)
    self.G.fill(10**-10) 
    #self.G = np.zeros(self.num_features)
  
  def mistakes_until_r_pefect(self, R):
    #compute the number of mistakes until R correct in a row
    curr_streak = 0
    num_mistakes = 0
    for i in range(0,self.num_examples):
      if (curr_streak == R):
        break
      label = self.training_labels[i]
      example = self.training_examples[i]
      self.update_gradient(example,label)
      if np.sign(np.dot(self.w, example)+self.thetha) != label:
        num_mistakes += 1
        curr_streak = 0
      else:
        curr_streak += 1
      if self.update_condition(example,label):
        self.update(example,label)
      if i == self.num_examples - 1:
        i = 0
    return num_mistakes

  def run(self):
    for i in range(0,20):
      for k, example in enumerate(self.d1):
        label = self.training_labels[k]
        self.update_gradient(example,label)
        if self.update_condition(example,label):
          self.update(example,label)
    return self.w

  def run_plot(self, N):
    mistake_list = []
    curr_iter = 1
    num_mistakes = 0
    for i in range(0,N):
      for k, example in enumerate(self.training_examples):
        label = self.training_labels[k]
        self.update_gradient(example,label)
        if np.sign(np.dot(self.w, example)+self.thetha) != label:
          num_mistakes += 1
        if self.update_condition(example,label):
          self.update(example,label)
        mistake_list.append((curr_iter, num_mistakes)) 
        curr_iter += 1
    return mistake_list

  def run_best(self, output_file):
    f = open(output_file, 'w+')
    num_mistakes = 0.0
    for k, example in enumerate(self.training_examples):
      label = self.training_labels[k]
      self.update_gradient(example,label)
      if np.sign(np.dot(self.w, example)+self.thetha) != label:
        num_mistakes += 1
      if self.update_condition(example,label):
        self.update(example,label)
    out = "Total number of mistakes in training: {}".format(num_mistakes)
    f.write(out)

  def run_best_20(self):
    for i in range(0,20):
      for k, example in enumerate(self.training_examples):
        label = self.training_labels[k]
        self.update_gradient(example,label)
        if self.update_condition(example,label):
          self.update(example,label)
    return (self.w, self.thetha)

  def update_condition(self, example, label):
    pred = np.dot(self.w,example)+self.thetha
    return pred*label <= 1

  def update(self, example, label):
    self.w += np.divide(self.learning_rate*label*example,np.sqrt(self.G))
    self.thetha += self.learning_rate*label/np.sqrt(self.G_thetha)

  def update_gradient(self, example, label):
    self.G += ((-1*label*example)**2)
    self.G_thetha += ((-1*label)**2)

class Evaluation:

  def __init__(self, training_file):
    self.learning_rates = [1.5,0.25,0.03,0.005,0.001]
    self.promotion_parameters = [1.1,1.01,1.005,1.0005,1.0001]
    self.margins = [2.0,0.3,0.04,0.006,0.001]
    self.training_file = training_file
    self.training_data = np.genfromtxt(training_file, delimiter=',')
  
  def evaluate_perceptron_margin(self, out_file):
    f = open(out_file, 'w+')
    best_learning_rate = None
    max_percent = 0.0
    for learning_rate in self.learning_rates:
      perceptron_margin = PerceptronMargin(self.training_data, learning_rate)
      computed_w = perceptron_margin.run()
      computed_thetha = perceptron_margin.thetha
      num_correct = 0.0
      num_incorrect = 0.0
      total = perceptron_margin.subsample_size
      for k,test_example in enumerate(perceptron_margin.d2):
        pred = np.sign(np.dot(computed_w,test_example)+computed_thetha)
        if pred == perceptron_margin.training_labels_d2[k]:
          num_correct += 1
      percent_correct = num_correct/total
      if percent_correct > max_percent:
        max_percent = percent_correct
        best_learning_rate = learning_rate
      num_incorrect = perceptron_margin.subsample_size-num_correct
      out_1 = "Percent correct for learning rate {}: {}".format(learning_rate, percent_correct)
      print out_1
      f.write(out_1+'\n')
      out = "Number Incorrect: {}".format(num_incorrect)
      print out
      f.write(out+'\n')
    out_2 = "Best performing learning rate was {} with {} percent correct".format(best_learning_rate, max_percent)
    print out_2
    f.write(out_2+'\n')

  def evaluate_winnow(self, out_file):
    f = open(out_file, 'w+')
    best_promotion_param = None
    max_percent = 0.0
    for promotion_param in self.promotion_parameters:
      winnow = Winnow(self.training_data, promotion_param)
      computed_w = winnow.run()
      computed_thetha = winnow.thetha
      num_correct = 0.0
      num_incorrect = 0.0
      total = winnow.subsample_size
      for k,test_example in enumerate(winnow.d2):
        pred = np.sign(np.dot(computed_w,test_example)+computed_thetha)
        if pred == winnow.training_labels_d2[k]:
          num_correct += 1
      percent_correct = num_correct/total
      if percent_correct > max_percent:
        max_percent = percent_correct
        best_promotion_param = promotion_param
      num_incorrect = winnow.subsample_size-num_correct
      out_1 = "Percent correct for promotion param {}: {}".format(promotion_param, percent_correct)
      print out_1
      f.write(out_1+'\n')
      out = "Number Incorrect: {}".format(num_incorrect)
      print out
      f.write(out+'\n')
    out_2 = "Best performing promotion param was {} with {} percent correct".format(best_promotion_param, max_percent)
    print out_2
    f.write(out_2+'\n')

  def evaluate_winnow_margin(self, out_file):
    f = open(out_file, 'w+')
    best_parameters = (None,None)
    max_percent = 0.0
    for promotion_param in self.promotion_parameters:
      for margin in self.margins:
        winnow_margin = WinnowMargin(self.training_data, promotion_param, margin)
        computed_w = winnow_margin.run()
        computed_thetha = winnow_margin.thetha
        num_correct = 0.0
        num_incorrect = 0.0
        total = winnow_margin.subsample_size
        for k,test_example in enumerate(winnow_margin.d2):
          pred = np.sign(np.dot(computed_w,test_example)+computed_thetha)
          if pred == winnow_margin.training_labels_d2[k]:
            num_correct += 1
        percent_correct = num_correct/total
        if percent_correct > max_percent:
          max_percent = percent_correct
          best_parameters = (promotion_param, margin)
        num_incorrect = winnow_margin.subsample_size-num_correct
        out_1 = "Percent correct for promotion param {} and margin {}: {}".format(promotion_param, margin, percent_correct)
        print out_1
        f.write(out_1+'\n')
        out = "Number Incorrect: {}".format(num_incorrect)
        print out
        f.write(out+'\n')
    out_2 = "Best performing promotion param and margin were {} {} with {} percent correct".format(best_parameters[0], best_parameters[1], max_percent)
    print out_2 
    f.write(out_2+'\n')

  def evaluate_adagrad(self, out_file):
    f = open(out_file, 'w+')
    best_learning_rate = None
    max_percent = 0.0
    for learning_rate in self.learning_rates:
      adagrad = AdaGrad(self.training_data, learning_rate)
      computed_w = adagrad.run()
      computed_thetha = adagrad.thetha
      num_correct = 0.0
      num_incorrect = 0.0
      total = adagrad.subsample_size
      for k,test_example in enumerate(adagrad.d2):
        pred = np.sign(np.dot(computed_w,test_example)+computed_thetha)
        if pred == adagrad.training_labels_d2[k]:
          num_correct += 1
      percent_correct = num_correct/total
      if percent_correct > max_percent:
        max_percent = percent_correct
        best_learning_rate = learning_rate
      num_incorrect = adagrad.subsample_size-num_correct
      out = "Percent correct for learning rate {}: {}".format(learning_rate, percent_correct)
      print out
      f.write(out+'\n')
      out = "Number Incorrect: {}".format(num_incorrect)
      print out
      f.write(out+'\n')
    out_2 = "Best performing learning rate was {} with {} percent correct".format(best_learning_rate, max_percent)
    print out_2
    f.write(out_2+'\n')

  def evaluate_training_mistakes(self):
    perceptron_margin = PerceptronMargin(self.training_data, 0.005)
    perceptron_margin.run_best('../output/1c_perceptron_margin_500')
    winnow = Winnow(self.training_data, 1.1)
    winnow.run_best('../output/1c_winnow_500')
    winnow_margin = WinnowMargin(self.training_data, 1.1, 2.0)
    winnow_margin.run_best('../output/1c_winnow_margin_500')
    adagrad = AdaGrad(self.training_data, 0.25)
    adagrad.run_best('../output/1c_adagrad_500')
  
  def get_best_weight_vectors(self):
    vectors = []
    perceptron = Perceptron(self.training_data)
    vectors.append(perceptron.run_best_20())
    perceptron_margin = PerceptronMargin(self.training_data, 0.03)
    vectors.append(perceptron_margin.run_best_20())
    winnow = Winnow(self.training_data, 1.1)
    vectors.append(winnow.run_best_20())
    winnow_margin = WinnowMargin(self.training_data, 1.1, 0.3)
    vectors.append(winnow_margin.run_best_20())
    adagrad = AdaGrad(self.training_data, 1.5)
    vectors.append(adagrad.run_best_20())
    return vectors

  def plot_cumulative_mistakes(self, N):
    #N is the number of examples
    perceptron = Perceptron(self.training_data)
    mistake_list = perceptron.run_plot(N)
    perceptron_plot, = plt.plot(*zip(*mistake_list))
    perceptron_margin = PerceptronMargin(self.training_data, 0.03)
    mistake_list = perceptron_margin.run_plot(N)
    perceptron_margin_plot, = plt.plot(*zip(*mistake_list))
    winnow = Winnow(self.training_data, 1.1)
    mistake_list = winnow.run_plot(N)
    winnow_plot, = plt.plot(*zip(*mistake_list))
    winnow_margin = WinnowMargin(self.training_data, 1.1, 2.0)
    mistake_list = winnow_margin.run_plot(N)
    winnow_margin_plot, = plt.plot(*zip(*mistake_list))
    adagrad = AdaGrad(self.training_data, 0.25)
    mistake_list = adagrad.run_plot(N)
    adagrad_plot, = plt.plot(*zip(*mistake_list))
    plt.legend([perceptron_plot, perceptron_margin_plot, winnow_plot, winnow_margin_plot, adagrad_plot], ['Perceptron', 'Perceptron Margin', 'Winnow', 'Winnow Margin', 'AdaGrad'], loc='upper left')
    plt.xlabel('# of examples')
    plt.ylabel('# of mistakes')
    plt.title('n=1000')
    plt.show()

  def plot_num_until_perfect(self, R, perceptron_margin_learning, winnow_param, winnow_margin_param, winnow_margin_margin, adagrad_learning):
    mistakes_list = []
    perceptron = Perceptron(self.training_data)
    mistakes_list.append(perceptron.mistakes_until_r_pefect(R))
    perceptron_margin = PerceptronMargin(self.training_data, perceptron_margin_learning)
    mistakes_list.append(perceptron_margin.mistakes_until_r_pefect(R))
    winnow = Winnow(self.training_data, winnow_param)
    mistakes_list.append(winnow.mistakes_until_r_pefect(R))
    winnow_margin = WinnowMargin(self.training_data, winnow_margin_param, winnow_margin_margin)
    mistakes_list.append(winnow_margin.mistakes_until_r_pefect(R))
    adagrad = AdaGrad(self.training_data, adagrad_learning)
    mistakes_list.append(adagrad.mistakes_until_r_pefect(R))
    return mistakes_list


def compute_1b():
  e = Evaluation('../res/L10M100N1000CLEAN')
  e.evaluate_perceptron_margin('../output/1b_perceptron_margin_1000')
  e.evaluate_winnow('../output/1b_winnow_1000') 
  e.evaluate_winnow_margin('../output/1b_winnow_margin_1000')
  e.evaluate_adagrad('../output/1b_adagrad_1000') 

def compute_1c():
  e = Evaluation('../res/L10M100N500CLEAN')
  e.evaluate_training_mistakes()

def plot_1d():
  e = Evaluation('../res/L10M100N1000CLEAN')
  e.plot_cumulative_mistakes(1)

def compute_2_params():
  for i in [40,80,120,160,200]:
    e = Evaluation("../res/L10M20N{}CLEAN".format(i))
    e.evaluate_perceptron_margin('../output/p2_perceptron_n{}'.format(i))
    e.evaluate_winnow('../output/p2_winnow_n{}'.format(i)) 
    e.evaluate_winnow_margin('../output/p2_winnow_margin_n{}'.format(i))
    e.evaluate_adagrad('../output/p2_adagrad_n{}'.format(i)) 

def compute_2():
  e_40 = Evaluation('../res/L10M20N40CLEAN')
  num_mistakes_40 = e_40.plot_num_until_perfect(1000, 1.5, 1.1, 1.1, 2.0, 1.5)
  
  e_80 = Evaluation('../res/L10M20N80CLEAN')
  num_mistakes_80 = e_80.plot_num_until_perfect(1000, 1.5, 1.1, 1.1, 2.0, 1.5)
 
  e_120 = Evaluation('../res/L10M20N120CLEAN')
  num_mistakes_120 = e_120.plot_num_until_perfect(1000, 0.25, 1.1, 1.1, 2.0, 1.5)

  e_160 = Evaluation('../res/L10M20N160CLEAN')
  num_mistakes_160 = e_160.plot_num_until_perfect(1000, 0.03, 1.1, 1.1, 2.0, 1.5)

  e_200 = Evaluation('../res/L10M20N200CLEAN')
  num_mistakes_200 = e_200.plot_num_until_perfect(1000, 0.03, 1.1, 1.1, 2.0, 1.5)

  x_coords = [40,80,120,160,200]
  perceptron_plot, = plt.plot(x_coords, [num_mistakes_40[0], num_mistakes_80[0], num_mistakes_120[0], num_mistakes_160[0], num_mistakes_200[0]])
  perceptron_margin_plot, = plt.plot(x_coords, [num_mistakes_40[1], num_mistakes_80[1], num_mistakes_120[1], num_mistakes_160[1], num_mistakes_200[1]])
  winnow_plot, = plt.plot(x_coords, [num_mistakes_40[2], num_mistakes_80[2], num_mistakes_120[2], num_mistakes_160[2], num_mistakes_200[2]])
  winnow_margin_plot, = plt.plot(x_coords, [num_mistakes_40[3], num_mistakes_80[3], num_mistakes_120[3], num_mistakes_160[3], num_mistakes_200[3]])
  adagrad_plot, = plt.plot(x_coords, [num_mistakes_40[4], num_mistakes_80[4], num_mistakes_120[4], num_mistakes_160[4], num_mistakes_200[4]])
  plt.legend([perceptron_plot, perceptron_margin_plot, winnow_plot, winnow_margin_plot, adagrad_plot], ['Perceptron', 'Perceptron Margin', 'Winnow', 'Winnow Margin', 'Adagrad'], loc='upper left')
  plt.xlabel('n (number of features)')
  plt.ylabel('# of mistakes')
  plt.title('Number of mistakes until 1000 classified correctly in a row')
  plt.show() 

def compute_3b():
  e = Evaluation('../res/L10M1000N1000NOISY')
  e.evaluate_perceptron_margin('../output/3b_perceptron_margin_m1000')
  e.evaluate_winnow('../output/3b_winnow_m1000') 
  e.evaluate_winnow_margin('../output/3b_winnow_margin_m1000')
  e.evaluate_adagrad('../output/3b_adagrad_m1000') 

def compute_3d(training_file, test_file):
  e = Evaluation(training_file)
  data = np.genfromtxt(test_file, delimiter=',')
  test_examples = data[:,:-1]
  test_labels = data[:,-1] 
  vectors = e.get_best_weight_vectors()
  name_dict = {0:'Perceptron', 1:'Perceptron Margin', 2:'Winnow', 3:'Winnow Margin', 4:'Adagrad'}
  for k,pair in enumerate(vectors):
    w = pair[0]
    thetha = pair[1]
    num_correct = 0.0
    for j,example in enumerate(test_examples):
      pred = np.sign(np.dot(w,example)+thetha)
      if pred == test_labels[j]:
        num_correct += 1
    print "Percent Correct for {} on m=1000: {}".format(name_dict[k], num_correct/10000.0) 

if __name__=="__main__":
  #compute_2_params()
  compute_2()
  #compute_1c()
  #plot_1d()
  #compute_3d('../res/L10M1000N1000NOISY', '../res/L10M1000N1000CLEAN_10000EX')
