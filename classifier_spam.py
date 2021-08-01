
# ******************************************************************************
# classifier.py
#
# Date      Name       Description
# ========  =========  ========================================================
# 01/08/21   Abuton     Initial version,
# ******************************************************************************
import numpy as np
from datetime import datetime
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, auc, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import cross_val_predict
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.utils import resample
from imblearn.metrics import geometric_mean_score

class Classifier():
    K_FOLD = 5
    ALGORITHMS = {
        "svm": "svm",
        "random_forest": "random_forest",
        "log_reg": "logistic_regression"
    }

    def read_dataset_bene(self, feature_file):
        data_file = pd.read_csv(feature_file)
        encoded_data = data_file.apply(LabelEncoder().fit_transform)

        input_cols = ['n_spam_word', 'hashtag_per_word', 'url_per_word', 'num_word', 'num_char', 'num_digit',
                  'num_url', 'num_hashtag', 'num_user_mention', 'num_rt', 'is_reply']
        return encoded_data[input_cols], encoded_data['is_spam']

    def read_dataset_chen(self, feature_file):
        data_file = pd.read_csv(feature_file)
        encoded_data = data_file.apply(LabelEncoder().fit_transform)

        input_cols = ['account_age', 'no_of_follower', 'no_of_following', 'no_user_fav', 'no_list', 'no_tweet',
                      'num_char', 'num_digit', 'num_url', 'num_hashtag', 'num_user_mention', 'num_rt']

        return encoded_data[input_cols], encoded_data['is_spam']

    def read_dataset_bene_upsample(self, feature_file, sample_size):
        data_file = pd.read_csv(feature_file)
        encoded_data = data_file.apply(LabelEncoder().fit_transform)

        input_cols = ['n_spam_word', 'hashtag_per_word', 'url_per_word', 'num_word', 'num_char', 'num_digit',
                  'num_url', 'num_hashtag', 'num_user_mention', 'num_rt', 'is_reply', 'is_spam']

        #print encoded_data['mci']
        # partition data based on class
        encoded_data =  encoded_data[input_cols]
        normal = encoded_data[encoded_data['is_spam'] == False]
        spam = encoded_data[encoded_data['is_spam'] == True]
        # down sample the majority class (here no-apply class) to match minority class
        minority_upsample = resample(spam,
                                        replace=True,  # sample without replacement
                                        n_samples=sample_size,  # to match minority class
                                        random_state=123)  # reproducible results
        # concat minority class as well downsampled majority class, this will be the balanced dataset
        upsampled = pd.concat([minority_upsample, normal])

        y = upsampled['is_spam']
        X = upsampled.drop('is_spam', axis=1)
        return X, y
        #return encoded_data[input_cols], encoded_data['is_spam']

    def read_dataset_chen_upsample(self, feature_file, sample_size):
        data_file = pd.read_csv(feature_file)
        encoded_data = data_file.apply(LabelEncoder().fit_transform)

        input_cols = ['account_age', 'no_of_follower', 'no_of_following', 'no_user_fav', 'no_list', 'no_tweet', 'num_char', 'num_digit', 'num_url', 'num_hashtag', 'num_user_mention', 'num_rt', 'is_spam']

        #return encoded_data[input_cols], encoded_data['is_spam']
        encoded_data = encoded_data[input_cols]
        normal = encoded_data[encoded_data['is_spam'] == False]
        spam = encoded_data[encoded_data['is_spam'] == True]
        # down sample the majority class (here no-apply class) to match minority class
        minority_upsample = resample(spam,
                                     replace=False,  # sample without replacement
                                     n_samples=sample_size,  # to match minority class
                                     random_state=123)  # reproducible results
        # concat minority class as well downsampled majority class, this will be the balanced dataset
        upsampled = pd.concat([minority_upsample, normal])

        y = upsampled['is_spam']
        X = upsampled.drop('is_spam', axis=1)
        return X, y

    def __init__(self, feature_file, flag, sample_size):
        if flag == 'B':
            if sample_size > 0:
                self.X, self.Y = self.read_dataset_bene_upsample(feature_file, sample_size)
            else:
                self.X, self.Y = self.read_dataset_bene(feature_file)
        elif flag == 'C':
            if sample_size > 0:
                self.X, self.Y = self.read_dataset_chen_upsample(feature_file, sample_size)
            else:
                self.X, self.Y = self.read_dataset_chen(feature_file)

    def check_data_distribution(self):
        print("\n------- Mean ------")
        print(self.data.mean(axis=0))
        print("\n\n------- Standard Deviation -------")
        print(self.data.std(axis=0))

        dt = preprocessing.scale(self.data)

    def svm_classify(self):
        """
        Uses Support Vector Machine Classifier
        :return: None
        """

        svm_classifier = svm.SVC(kernel="rbf", gamma='auto')
        before = datetime.now()
        svm_output = cross_val_predict(svm_classifier, self.X, self.Y, cv=self.K_FOLD)
        after = datetime.now()
        runtime = (after - before).total_seconds()
        #print("Time: ", runtime)
        #print("Support Vector Machine Classifier Report")
        #print("===========================================\n")
        return self.display_report(svm_output, self.ALGORITHMS["svm"])


    def random_forest(self):
        random_forest_classifier = RandomForestClassifier(n_estimators=100)
        before = datetime.now()
        classification = cross_val_predict(random_forest_classifier, self.X, self.Y, cv=self.K_FOLD)
        after = datetime.now()
        runtime = (after - before).total_seconds()
        #print("Time: ", runtime)
        #print("Random Forest Classifier Report")
        #print("===========================================\n")
        return self.display_report(classification, self.ALGORITHMS["random_forest"])


    def display_report(self, prediction, algorithm):
        """
        Displays the classification report
        :param prediction: Prediction of the model
        :param algorithm: Algorithm used to generate predictions
        :return: None
        """
        result = {}
        #print("Confusion Matrix: ")
        #print("------------------\n")
        #self.plot_confusion_matrix(confusion_matrix(self.Y, prediction), algorithm)

        # print ("\nClassification Report: ")
        # print("-----------------------")
        # print("Accuracy:  ", metrics.accuracy_score(self.Y, prediction))
        # print("Precision: ", metrics.precision_score(self.Y, prediction))
        # print("Recall:    ", metrics.recall_score(self.Y, prediction))
        # print("F1-Score:  ", metrics.f1_score(self.Y, prediction))
        # #print("G-Mean:    ", metrics.)
        # print("area under curve (auc): ", metrics.roc_auc_score(self.Y, prediction))
        # print(classification_report(self.Y, prediction))
        # false_positive_rate, true_positive_rate, thresholds = roc_curve(self.Y, prediction)
        # print("(auc): ", auc(false_positive_rate, true_positive_rate))
        result['acc'] = metrics.accuracy_score(self.Y, prediction)
        result['p'] = metrics.precision_score(self.Y, prediction)
        result['r'] = metrics.recall_score(self.Y, prediction)
        result['f1'] = metrics.f1_score(self.Y, prediction)
        result['g'] = geometric_mean_score(self.Y, prediction)
        return result

    def plot_confusion_matrix(self, con_mat, algorithm,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        :param con_mat: confusion matrix
        :param algorithm: algorithm used
        :param normalize: boolean to normalize or not
        :param title: title of the graph
        :param cmap: map color palate
        :return: None
        """

        plt.figure()
        plt.imshow(con_mat, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ["normal", "spam"], rotation=45)
        plt.yticks(tick_marks, ["normal", "spam"])

        if normalize:
            con_mat = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(con_mat)

        thresh = con_mat.max() / 2.
        for i, j in itertools.product(range(con_mat.shape[0]), range(con_mat.shape[1])):
            plt.text(j, i, con_mat[i, j],
                     horizontalalignment="center",
                     color="white" if con_mat[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig("figures/" + algorithm + ".png")
