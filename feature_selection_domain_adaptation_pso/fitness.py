import math
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import warnings
from data_complexity import f2, f3
from sklearn import svm
from sklearn.naive_bayes import CategoricalNB, GaussianNB

warnings.filterwarnings(action="ignore", category=FutureWarning)

def mmd_rbf(X, Y, gamma=1.0):
    """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})
    Returns:
        [scalar] -- [MMD value]
    """
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)

    m = XX.shape[1]
    n = YY.shape[1]

    return math.sqrt(max(0, np.sum(XX)/(m*m) + np.sum(YY)/(n*n) - 2/m/n * np.sum(XY)))


def fitness_knn(position, fn_arguments):
    # Get fn arguments
    src_x = fn_arguments['src_x']
    src_y = fn_arguments['src_y']
    tar_x = fn_arguments['tar_x']
    tar_y = fn_arguments['tar_y']
    sw = fn_arguments['sw']
    tw = fn_arguments['tw']
    stw = fn_arguments['stw']
    min_features = fn_arguments['min_features']
    # Check min features
    if sum(position) <= len(position) * min_features:
        return math.inf

    # Subset training data
    src_x = src_x[:, position]
    tar_x = tar_x[:, position]

    src_pen = 0
    tar_pen = 0
    diffst = 0

    # model = KNeighborsClassifier(n_neighbors=3)
    model = KNeighborsClassifier(n_neighbors=1)

    # srcPen
    if sw > 0:
        src_x_split1, src_x_split2, src_x_split3 = np.array_split(src_x, 3)
        src_y_split1, src_y_split2, src_y_split3 = np.array_split(src_y, 3)

        # Test on split 1
        model.fit(np.vstack((src_x_split2, src_x_split3)),
                  np.append(src_y_split2, src_y_split3))
        predicted = model.predict(src_x_split1)
        split1_error = np.mean(predicted != src_y_split1)
        
        # Test on split 2
        model.fit(np.vstack((src_x_split1, src_x_split3)),
                  np.append(src_y_split1, src_y_split3))
        predicted = model.predict(src_x_split2)
        split2_error = np.mean(predicted != src_y_split2)
        # Test on split 3
        model.fit(np.vstack((src_x_split2, src_x_split1)),
                  np.append(src_y_split2, src_y_split1))
        predicted = model.predict(src_x_split3)
        split3_error = np.mean(predicted != src_y_split3)

        src_pen = (split1_error + split2_error + split3_error) / 3

    # tarPen
    if tw > 0:
        model.fit(src_x, src_y)
        predicted = model.predict(tar_x)
        tar_pen = np.mean(predicted != tar_y)

    if stw > 0:
        diffst = mmd_rbf(src_x, tar_x)

    return sw * src_pen + tw * tar_pen + stw * diffst



def fitness_f1(position, fn_arguments):
    # Get fn arguments
    src_x = fn_arguments['src_x']
    tar_x = fn_arguments['tar_x']
    src_f1 = fn_arguments['src_f1']
    tar_f1 = fn_arguments['tar_f1']
    sw = fn_arguments['sw']
    tw = fn_arguments['tw']
    stw = fn_arguments['stw']
    min_features = fn_arguments['min_features']
    # Check min features
    if sum(position) <= len(position) * min_features:
        return math.inf

    src_pen = 0
    tar_pen = 0
    diffst = 0

    if sw > 0:
        src_pen = min(src_f1[position])
    if tw > 0:
        tar_pen = min(tar_f1[position])
    if stw > 0:
        src_x = src_x[:, position]
        tar_x = tar_x[:, position]
        diffst = mmd_rbf(src_x, tar_x)
    
    return sw * src_pen + tw * tar_pen + stw * diffst

def fitness_f2(position, fn_arguments):
    # Get fn arguments
    src_x = fn_arguments['src_x']
    tar_x = fn_arguments['tar_x']
    src_f2_pre = fn_arguments['src_f2_pre']
    tar_f2_pre = fn_arguments['tar_f2_pre']
    sw = fn_arguments['sw']
    tw = fn_arguments['tw']
    stw = fn_arguments['stw']
    min_features = fn_arguments['min_features']
    
    # Check min features
    if sum(position) <= len(position) * min_features:
        return math.inf

    src_pen = 0
    tar_pen = 0
    diffst = 0

    if sw > 0:
        src_pen = f2(src_f2_pre, position)
    if tw > 0:
        tar_pen = f2(tar_f2_pre, position)
    if stw > 0:
        src_x = src_x[:, position]
        tar_x = tar_x[:, position]
        diffst = mmd_rbf(src_x, tar_x)

    return sw * src_pen + tw * tar_pen + stw * diffst


def fitness_f3(position, fn_arguments):
    # Get fn arguments
    src_x = fn_arguments['src_x']
    tar_x = fn_arguments['tar_x']
    src_f3_pre = fn_arguments['src_f3_pre']
    tar_f3_pre = fn_arguments['tar_f3_pre']
    sw = fn_arguments['sw']
    tw = fn_arguments['tw']
    stw = fn_arguments['stw']
    min_features = fn_arguments['min_features']
    # Check min features
    if sum(position) <= len(position) * min_features:
        return math.inf

    src_pen = 0
    tar_pen = 0
    diffst = 0

    if sw > 0:
        src_pen = f3(src_f3_pre, position)
    if tw > 0:
        tar_pen = f3(tar_f3_pre, position)
    if stw > 0:
        src_x = src_x[:, position]
        tar_x = tar_x[:, position]
        diffst = mmd_rbf(src_x, tar_x)

    return sw * src_pen + tw * tar_pen + stw * diffst



def fitness_svm(position, fn_arguments):
    # Get fn arguments
    src_x = fn_arguments['src_x']
    src_y = fn_arguments['src_y']
    tar_x = fn_arguments['tar_x']
    tar_y = fn_arguments['tar_y']
    sw = fn_arguments['sw']
    tw = fn_arguments['tw']
    stw = fn_arguments['stw']
    min_features = fn_arguments['min_features']
    # Check min features
    if sum(position) <= len(position) * min_features:
        return math.inf

    # Subset training data
    src_x = src_x[:, position]
    tar_x = tar_x[:, position]

    src_pen = 0
    tar_pen = 0
    diffst = 0

    model = svm.SVC(kernel='rbf')

    # srcPen
    if sw > 0:
        src_x_split1, src_x_split2, src_x_split3 = np.array_split(src_x, 3)
        src_y_split1, src_y_split2, src_y_split3 = np.array_split(src_y, 3)

        # Test on split 1
        model.fit(np.vstack((src_x_split2, src_x_split3)),
                  np.append(src_y_split2, src_y_split3))
        predicted = model.predict(src_x_split1)
        split1_error = np.mean(predicted != src_y_split1)
        
        # Test on split 2
        model.fit(np.vstack((src_x_split1, src_x_split3)),
                  np.append(src_y_split1, src_y_split3))
        predicted = model.predict(src_x_split2)
        split2_error = np.mean(predicted != src_y_split2)
        # Test on split 3
        model.fit(np.vstack((src_x_split2, src_x_split1)),
                  np.append(src_y_split2, src_y_split1))
        predicted = model.predict(src_x_split3)
        split3_error = np.mean(predicted != src_y_split3)

        src_pen = (split1_error + split2_error + split3_error) / 3

    # tarPen
    if tw > 0:
        model.fit(src_x, src_y)
        predicted = model.predict(tar_x)
        tar_pen = np.mean(predicted != tar_y)

    if stw > 0:
        diffst = mmd_rbf(src_x, tar_x)

    return sw * src_pen + tw * tar_pen + stw * diffst

def fitness_nb(position, fn_arguments):
    # Get fn arguments
    src_x = fn_arguments['src_x']
    src_x_discretized = fn_arguments['src_x_discretized']
    src_y = fn_arguments['src_y']
    tar_x = fn_arguments['tar_x']
    tar_x_discretized = fn_arguments['tar_x_discretized']
    tar_y = fn_arguments['tar_y']
    sw = fn_arguments['sw']
    tw = fn_arguments['tw']
    stw = fn_arguments['stw']
    min_features = fn_arguments['min_features']
    bins = fn_arguments['bins']

    # Check min features
    if sum(position) <= len(position) * min_features:
        return math.inf

    # Subset training data
    src_x = src_x[:, position]
    src_x_discretized = src_x_discretized[:, position]
    tar_x = tar_x[:, position]
    tar_x_discretized = tar_x_discretized[:, position]

    src_pen = 0
    tar_pen = 0
    diffst = 0

    model = CategoricalNB(min_categories = bins)

    # srcPen
    if sw > 0:
        src_x_split1, src_x_split2, src_x_split3 = np.array_split(src_x_discretized, 3)
        src_y_split1, src_y_split2, src_y_split3 = np.array_split(src_y, 3)

        # Test on split 1
        model.fit(np.vstack((src_x_split2, src_x_split3)),
                  np.append(src_y_split2, src_y_split3))
        predicted = model.predict(src_x_split1)
        split1_error = np.mean(predicted != src_y_split1)
        
        # Test on split 2
        model.fit(np.vstack((src_x_split1, src_x_split3)),
                  np.append(src_y_split1, src_y_split3))
        predicted = model.predict(src_x_split2)
        split2_error = np.mean(predicted != src_y_split2)
        # Test on split 3
        model.fit(np.vstack((src_x_split2, src_x_split1)),
                  np.append(src_y_split2, src_y_split1))
        predicted = model.predict(src_x_split3)
        split3_error = np.mean(predicted != src_y_split3)

        src_pen = (split1_error + split2_error + split3_error) / 3

    # tarPen
    if tw > 0:
        model.fit(src_x_discretized, src_y)
        predicted = model.predict(tar_x_discretized)
        tar_pen = np.mean(predicted != tar_y)

    if stw > 0:
        diffst = mmd_rbf(src_x, tar_x)

    return sw * src_pen + tw * tar_pen + stw * diffst