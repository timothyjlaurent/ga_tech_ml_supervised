from sklearn.learning_curve import learning_curve
from sklearn.cross_validation import StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.learning_curve import learning_curve
import pickle

def make_learning_curve(classifier, X, y, 
                        cv=None, 
                        train_sizes=np.linspace(.1, .4, 6) ):
    """
    Generate output of the learning curve using a given classifier and feature sets and class designation. 
    """
    
    cv = cv or StratifiedKFold(y, n_folds=10)
    
    train_sizes, train_scores, test_scores = learning_curve(classifier,
                                            X,
                                            y,
                                            cv=cv,
                                            train_sizes=train_sizes,
                                            n_jobs=-1)
    return (train_sizes, train_scores, test_scores)

def make_learning_curve_dict( input_dict, classes, gridsearch ,cv=3, train_sizes=[200,400,600,800,1000,5000,10000], file_name="" ):
    learning_curve_dict = {}
    for key in sorted(input_dict.keys()):
        classifier = gridsearch[key]['gs'].best_estimator_
        start = time.time()
        input = input_dict[key]
        print input
        lc_output = make_learning_curve(classifier, input, training_classes, cv=cv, train_sizes=train_sizes)
        print gridsearch[key]['gs'].best_estimator_.get_params()
        print lc_output
        learning_curve_dict[key] = {'lc_output' : lc_output, 'time': time.time() - start, 'params':classifier.get_params()  }
    print learning_curve_dict
    with open(file_name, 'wb') as data_file:
        print 'saving: '.format(file_name)
        pickle.dump(learning_curve_dict, data_file)
    return learning_curve_dict
    
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def plot_learning_curve_from_learning_curve_data(title='Learning Curve:', *lc_args):
    """
    Generate a simple plot of the test and training learning curve with the our

    Parameters
    ----------
    title : string
        Title for the chart.
    *lc_args : output of shape: train_sizes, 
                                train_scores_mean, 
                                train_scores_std, 
                                test_scores_mean, 
                                test_scores_std
  
    """
    
    train_sizes, train_scores, test_scores = lc_args
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    print "train test sizes: {}".format(train_sizes)
    print "train scores means: {}".format(train_scores_mean)
    print "train scores std: {}".format(train_scores_std)
    print "test scores mean: {}".format(test_scores_mean)
    print "test scores std: {}".format(test_scores_std)
    
    plt.figure()
    plt.title(title)
#     if ylim is not None:
#         plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# make make learning curve plots
def make_learning_curve_plots(base_title, lc_dict, lc_gridsearch ):
    plots = {}
    for key in sorted(lc_dict):
        lc_output = lc_dict[key]['lc_output']
        title = '{}\n(n_components={}\nparams={}\nLearning Curve'.format(base_title, key, lc_gridsearch[key]['gs'].best_estimator_.get_params() , lc_dict[n]['time'] )
        plot = plot_learning_curve_from_learning_curve_data(title, *lc_output)
        plots[key] = plot
    return plots

def save_thing(thing, path):
    with open(path, 'wb') as file:
        pickle.dump(thing, file)
        
def load_thing(path):
    with open(path, 'rb') as file:
        return pickle.load(path)
    
def replace_missing_values_with_mean(df, col, query, round=True):
    mean = df.query(query)
