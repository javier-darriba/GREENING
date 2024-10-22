import argparse
from data import *
from fitness import *
from data_complexity import *
from sbpso import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
from sklearn import svm
from datetime import datetime
import pandas as pd
import os.path
import numpy as np
from collections import Counter
import ast
import random


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=1, help="Random seed")
parser.add_argument("--dataset", type=str, default="gas2", help="Dataset used")
parser.add_argument("--fitness", type=str, default="knn",
                    help="Fitness function used")
parser.add_argument("--swarm", type=int, default=50, help="Swarm size")
parser.add_argument("--maxlife", type=int, default=40,
                    help="Maximum number of iterations that a bit of a particle can keep its value")
parser.add_argument("--maxit", type=int, default=3000,
                    help="Maximum number of iterations of the algorithm")
parser.add_argument("--maxitstagnate", type=int, default=300,
                    help="Maximum number of iterations the algorithm will take without improving its best value")
parser.add_argument("--report", type=int, default=100,
                    help="Number of iterations after which the results will be logged")
parser.add_argument("--sw", type=float, default=0.1,
                    help="sw weight of the fitness function")
parser.add_argument("--tw", type=float, default=0.9,
                    help="tw weight of the fitness function")
parser.add_argument("--stw", type=float, default=0.0,
                    help="stw weight of the fitness function")
parser.add_argument("--minfeatures", type=float, default=0.1,
                    help="Minimum number of features selected")
parser.add_argument("--mode", type=str, default="prod",
                    help="Mode (desa for writing log and results in separate files)")
parser.add_argument("--bins", type=int, default=5,
                    help="Number of bins to discretize for testing on a Naive Bayes classifier")


args = parser.parse_args()

if args.mode == "prod":
    log_filename = "sbpso.log"
    results_filename = "results.csv"
else:
    log_filename = "sbpso_DESA.log"
    results_filename = "results_DESA.csv"


# Logger config
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]:  %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)


def train(src_train_x, src_train_y, tar_train_x, tar_train_y, nprocess, lock):
    # PRECALCULATE COMPLEXITY METRICS
    src_f1 = tar_f1 = src_f2_pre = tar_f2_pre = src_f3_pre = tar_f3_pre = 0
    src_x_discretized_all = tar_train_x_discretized_all = None

    fitness_fn = fitness_knn
    if args.fitness == "f1":
        fitness_fn = fitness_f1
        src_f1 = f1(src_train_x, src_train_y)
        tar_f1 = f1(tar_train_x, tar_train_y)
    elif args.fitness == "f2":
        fitness_fn = fitness_f2
        src_f2_pre = f2_pre(src_train_x, src_train_y)
        tar_f2_pre = f2_pre(tar_train_x, tar_train_y)
    elif args.fitness == "f3":
        fitness_fn = fitness_f3
        src_f3_pre = f3_pre(src_train_x, src_train_y)
        tar_f3_pre = f3_pre(tar_train_x, tar_train_y)
    elif args.fitness == "svm":
        fitness_fn = fitness_svm
    elif args.fitness == "nb":
        fitness_fn = fitness_nb
        src_x_discretized_all, tar_train_x_discretized_all = discretize(
            src_train_x, tar_train_x, args.bins)

    logging.info("Starting training...")
    training_time_start = datetime.now()
    position, value, iterations, time_per_iteration = sbpso(args.seed, nprocess, lock, num_features=src_train_x.shape[1],
                                                            fn=fitness_fn,
                                                            swarm_size=args.swarm,
                                                            max_life=args.maxlife,
                                                            max_it=args.maxit,
                                                            max_it_stagnate=args.maxitstagnate,
                                                            report=args.report,
                                                            # fn_args:
                                                            src_x=src_train_x,
                                                            src_y=src_train_y,
                                                            tar_x=tar_train_x,
                                                            tar_y=tar_train_y,
                                                            src_x_discretized=src_x_discretized_all,
                                                            tar_x_discretized=tar_train_x_discretized_all,
                                                            sw=args.sw,
                                                            tw=args.tw,
                                                            stw=args.stw,
                                                            min_features=args.minfeatures,
                                                            src_f1=src_f1,
                                                            tar_f1=tar_f1,
                                                            src_f2_pre=src_f2_pre,
                                                            tar_f2_pre=tar_f2_pre,
                                                            src_f3_pre=src_f3_pre,
                                                            tar_f3_pre=tar_f3_pre,
                                                            bins=args.bins)
    training_time = datetime.now() - training_time_start
    return position, iterations, training_time, time_per_iteration

# Compute true positive rate and true negative rate
def tprtnr(predicted, real):
    nlabels = np.unique(real)
    if nlabels.size > 2:
        conf_matrix = multilabel_confusion_matrix(real, predicted)

        # TPR is the number of true positives divided by the number of true positives plus the number of false negatives
        tpr = conf_matrix[:, 1, 1] / \
            (conf_matrix[:, 1, 1] + conf_matrix[:, 1, 0])

        # TNR is the number of true negatives divided by the number of true negatives plus the number of false positives
        tnr = conf_matrix[:, 0, 0] / \
            (conf_matrix[:, 0, 0] + conf_matrix[:, 0, 1])

        tpr = np.mean(tpr)
        tnr = np.mean(tnr)
    else:
        tn, fp, fn, tp = confusion_matrix(real, predicted).ravel()
        tpr = tp / (tp + fn)
        tnr = tn / (tn + fp)

    return tpr, tnr


def test(tar_train_name, src_test_x, src_test_y, tar_test_x, tar_test_y, position, iterations, training_time, time_per_iteration, column_names):
    logging.info("***** RESULTS *****")
    logging.info("Selected features: " + str(sum(position)))
    # logging.info(position)
    # logging.info("Value: " + str(value))
    logging.info("Iterations: " + str(iterations))
    logging.info("Time per iteration: " + str(time_per_iteration))

    # kNN
    # model = KNeighborsClassifier(n_neighbors=3)
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(src_test_x[:, position], src_test_y)
    predicted = model.predict(tar_test_x[:, position])
    error_knn = np.mean(predicted != tar_test_y)
    logging.info("===> Accuracy kNN: " + str(1-error_knn))
    knn_tpr, knn_tnr = tprtnr(predicted, tar_test_y)
    logging.info(f'        Mean kNN TPR: {knn_tpr} - Mean kNN TNR: {knn_tnr}')

    knnPredictFreqStr = '{'
    freq = Counter(predicted)
    for kv in freq.most_common():
        knnPredictFreqStr += f' {kv[0]}: {kv[1]},'
    real_labels = list(set(tar_test_y))
    for label in real_labels:
        if label not in freq.keys():
            knnPredictFreqStr += f' {label}: 0,'
    knnPredictFreqStr = knnPredictFreqStr[:-1]
    knnPredictFreqStr += ' }'

    # SVM
    # model = svm.SVC(kernel='rbf')
    model = svm.SVC(kernel='linear')
    model.fit(src_test_x[:, position], src_test_y)
    predicted = model.predict(tar_test_x[:, position])
    error_svm = np.mean(predicted != tar_test_y)
    logging.info("===> Accuracy SVM: " + str(1-error_svm))
    svm_tpr, svm_tnr = tprtnr(predicted, tar_test_y)
    logging.info(f'        Mean SVM TPR: {svm_tpr} - Mean SVM TNR: {svm_tnr}')

    svmPredictFreqStr = '{'
    freq = Counter(predicted)
    for kv in freq.most_common():
        svmPredictFreqStr += f' {kv[0]}: {kv[1]},'
    real_labels = list(set(tar_test_y))
    for label in real_labels:
        if label not in freq.keys():
            svmPredictFreqStr += f' {label}: 0,'
    svmPredictFreqStr = svmPredictFreqStr[:-1]
    svmPredictFreqStr += ' }'

    # Naive Bayes
    src_x_discretized, tar_x_discretized = discretize(
        src_test_x[:, position], tar_test_x[:, position], args.bins)

    model = CategoricalNB(min_categories=args.bins)
    model.fit(src_x_discretized, src_test_y)
    predicted = model.predict(tar_x_discretized)
    error_nb = np.mean(predicted != tar_test_y)
    logging.info("Accuracy NB: " + str(1-error_nb))
    nb_tpr, nb_tnr = tprtnr(predicted, tar_test_y)
    logging.info(f'        Mean NB TPR: {nb_tpr} - Mean NB TNR: {nb_tnr}')

    nbPredictFreqStr = '{'
    freq = Counter(predicted)
    for kv in freq.most_common():
        nbPredictFreqStr += f' {kv[0]}: {kv[1]},'
    real_labels = list(set(tar_test_y))
    for label in real_labels:
        if label not in freq.keys():
            nbPredictFreqStr += f' {label}: 0,'
    nbPredictFreqStr = nbPredictFreqStr[:-1]
    nbPredictFreqStr += ' }'

    positionStr = '['
    for pos in position:
        positionStr += f' {str(pos)},'
    positionStr = positionStr[:-1]
    positionStr += ' ]'

    featureNamesStr = '['
    for f in [x for x, boolean in zip(column_names, position) if boolean]:
        featureNamesStr += f' {str(f)},'
    featureNamesStr = featureNamesStr[:-1]
    featureNamesStr += ' ]'

    realFreqStr = '{'
    freq = Counter(tar_test_y)
    for kv in freq.most_common():
        realFreqStr += f' {kv[0]}: {kv[1]},'
    realFreqStr = realFreqStr[:-1]
    realFreqStr += ' }'

    row = [args.seed, args.fitness, tar_train_name, sum(position), 1-error_knn, knn_tpr, knn_tnr, 1-error_svm, svm_tpr, svm_tnr, 1-error_nb, nb_tpr, nb_tnr, iterations, training_time, time_per_iteration,
           args.sw, args.tw, args.stw, positionStr, args.swarm, args.maxlife, args.maxit, args.maxitstagnate, args.minfeatures, featureNamesStr, realFreqStr, knnPredictFreqStr, svmPredictFreqStr, nbPredictFreqStr]
    columns = ["Seed", "Fitness", "Dataset tar train", "# Features", "kNN test acc.", "kNN TPR", "kNN TNR", "SVM test acc.", "SVM TPR", "SVM TNR", "NB test acc.", "NB TPR", "NB TNR", "Iterations", "Time", "Time per iteration", "sw", "tw",
               "stw", "Features", "Swarm size", "Max life", "Max iterations", "Max iterations stagnate", "Min features", "Feature names", "Real label distribution", "kNN Predict distribution", "SVM Predict distribution", "NB Predict distribution"]

    result = pd.DataFrame([row], columns=columns)
    try:
        results_df = pd.read_csv(results_filename, index_col=0)
        results_df = pd.concat([results_df, result], ignore_index=True)
        results_df.to_csv(results_filename)
    except FileNotFoundError:
        result.to_csv(results_filename)


def main(nprocess, lock):
    # Check if test already carried out
    tests_completed = pd.DataFrame()

    _, tar_dataset_name = get_dataset_name(args.dataset)
    if os.path.isfile(results_filename):
        results_df = pd.read_csv(results_filename, index_col=0)
        tests_completed = results_df.loc[(results_df['Seed'] == args.seed)
                                         & (results_df['Dataset tar train'] == tar_dataset_name)
                                         & (results_df['Fitness'] == args.fitness)
                                         & (results_df['sw'] == args.sw)
                                         & (results_df['tw'] == args.tw)
                                         & (results_df['stw'] == args.stw)
                                         & (results_df['Swarm size'] == args.swarm)
                                         & (results_df['Max life'] == args.maxlife)
                                         & (results_df['Max iterations'] == args.maxit)
                                         & (results_df['Max iterations stagnate'] == args.maxitstagnate)
                                         & (results_df['Min features'] == args.minfeatures)]

    try:
        if tests_completed.shape[0] == 0:
            logging.info("-------------------------------------------------")
            logging.info("                  STARTING TEST")
            if args.mode == "desa":
                logging.info("                  *** DESA ***")
            for arg in vars(args):
                logging.info("    " + arg + " = " + str(getattr(args, arg)))
            logging.info(f"    Dataset tar train = {tar_dataset_name}")
            logging.info("-------------------------------------------------")

            # READ DATA
            logging.info("Reading data...")
            np.random.seed(args.seed)  # args.seed es el seed para entrenar
            src_train_x, src_train_y, src_test_x, src_test_y, tar_train_x, tar_train_y, tar_test_x, tar_test_y, column_names = read_data(
                args.dataset)

            # TRAIN
            position, iterations, training_time, time_per_iteration = train(
                src_train_x, src_train_y, tar_train_x, tar_train_y, nprocess, lock)

            test(tar_dataset_name, src_test_x, src_test_y, tar_test_x, tar_test_y,
                 position, iterations, training_time, time_per_iteration.total_seconds(), column_names)

        else:
            print("Tests already completed")
    except:
        logging.exception('Oh no! Error :(  -->  Here is the traceback info:')


if __name__ == "__main__":
    manager = mp.Manager()
    nprocess = manager.Value(int, 0)
    lock = manager.Lock()

    main(nprocess, lock)
