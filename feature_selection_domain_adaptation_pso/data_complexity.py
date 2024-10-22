import numpy as np
import itertools

# F1 - Maximum fisher's discriminant ratio
def f1(x, y):
    num = den = None
    colmeans_x = np.mean(x, axis=0)
    for label in np.unique(y):
        tmp = x[y == label]
        colmeans_tmp = np.mean(tmp, axis=0)
        aux_num = [tmp.shape[0] * (colmeans_tmp - colmeans_x) ** 2]
        aux_den = np.transpose(np.sum((tmp - colmeans_tmp) ** 2, axis=1))

        if num is None:
            num = np.array(aux_num)
            den = np.array(aux_den)
        else:
            num = np.append(num, aux_num, axis=0)
            den = np.append(den, aux_den, axis=0)

    aux = np.sum(num, axis=0) / np.sum(den, axis=0)
    return 1 / (aux + 1)

# Returns a list with the data corresponding to every combination of 2 of the classes in the dataset
def ovo(x,y):
    l = np.unique(y)
    # Generate all possible combinations of two classes
    aux = list(itertools.combinations(l,2))

    data_combinations = []

    for classes in aux:
        tmp_x = x[(y == classes[0]) | (y == classes[1])]
        tmp_y = y[(y == classes[0]) | (y == classes[1])]
        data_combinations.append((tmp_x, tmp_y))
    
    return data_combinations


def f2_regionOver_custom(x, y):
    l = np.unique(y)
    a = x[y == l[0]]
    b = x[y == l[1]]

    maxmax = np.array([[np.max(a, axis=0)], [np.max(b, axis=0)]])
    minmin = np.array([[np.min(a, axis=0)], [np.min(b, axis=0)]])

    # Clip hace que el valor mínimo sea 0, es decir, los negativos pasarán a ser 0
    over = np.max(maxmax - minmin, axis=0).clip(min=0)
    rang = np.max(maxmax, axis=0) - np.min(minmin, axis=0)

    over_rang = over / rang

    nan_pos = np.isnan(over_rang)
    over_rang[nan_pos] = 0

    return over_rang

def f2_pre(x, y):
    data = ovo(x,y)
    for i in range(0, len(data)):
        data[i] = f2_regionOver_custom(data[i][0], data[i][1])
    return data

# F2 - Length of overlapping region
def f2(pre, position):
    f2_sum = 0
    for f2_pre in pre:
        f2_sum += sum(f2_pre[0][position])
    return f2_sum


def f3_nonOverlap_custom(x,y):
    l = np.unique(y)
    a = x[y == l[0]]
    b = x[y == l[1]]

    minmax = np.min(np.array([[np.max(a, axis=0)], [np.max(b, axis=0)]]), axis=0)
    maxmin = np.max(np.array([[np.min(a, axis=0)], [np.min(b, axis=0)]]), axis=0)

    return np.mean((x < maxmin) | (x > minmax), axis=0)

def f3_pre(x,y):
    data = ovo(x,y)

    for i in range(0, len(data)):
        data[i] = f3_nonOverlap_custom(data[i][0], data[i][1])
    
    data = np.vstack(data)

    return 1 - np.max(data, axis=0)

# F3 - Maximum individual feature efficiency
def f3(pre, position):
    return np.mean(pre[position])