import argparse

from data_manager import DataManager
from data_processor import DataProcessor
from hog_lda_bayes_classifier import HogLdaBayesClassifier
from hog_pca_knn_classifier import HogPcaKnnClassifier
from performance_evaluator import PerformanceEvaluator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Traffic signals detection through image classification')
    parser.add_argument(
        '--train_path', type=str, default="train_jpg", help='Select the training data dir')
    parser.add_argument(
        '--test_path', type=str, default="test_alumnos_jpg", help='Select the testing data dir')
    parser.add_argument(
        '--classifier', type=str, default="HOG_LDA_BAYES", help='Select classifier')

    args = parser.parse_args()

    path_train = args.train_path
    path_test = args.test_path
    type_classifier = args.classifier

    data_manager = DataManager()
    data_processor = DataProcessor()
    performance_evaluator = PerformanceEvaluator()

    path_serialized_classifier = data_manager.join_path("classifiers", type_classifier)

    if not data_manager.exists(path_serialized_classifier):  # train

        # 1- Load train data

        print("Loading train data...")
        files_train = data_manager.list_directory(path_train)

        # 2- Data treatment
        print("Generating train data:")
        X, y, X_train, y_train, X_val, y_val = data_processor.get_train_data(data_manager,
                                                                             path_train,
                                                                             files_train,
                                                                             type_classifier)

        # 3- Create classifier
        print("Creating classifier...")

        if type_classifier == "HOG_LDA_BAYES":
            img_classifier = HogLdaBayesClassifier()
        elif type_classifier == "HOG_PCA_KNN":
            img_classifier = HogPcaKnnClassifier()
        else:
            raise ValueError('Incorrect classifier type')

        # 4- Train classifier
        print("Training classifier...")

        img_classifier.fit(X_train, y_train)

        # 5- Get metrics

        y_pred_val, y_prob_val, y_scores_val = img_classifier.predict(X_val)

        # 6- Validate the model

        # Confusion matrix val
        performance_evaluator.multiclass_confusion_matrix(y_val, y_pred_val)
        # Performance metrics val
        performance_evaluator.performance_metrics_report(y_val, y_pred_val)

        # 7- Fit the model again with combined train and val sets

        img_classifier.fit(X, y)

        # 8- Store classifier
        print("Saving classifier...")
        data_manager.save_classifier(path_serialized_classifier, img_classifier)

    else:  # Load trained and serialized classifier
        print("Loading classifier...")

        img_classifier = data_manager.load_classifier(path_serialized_classifier)

    # 9- Load and process test images
    print("Loading test data:")

    files_test = data_manager.list_directory(path_test)

    info_X_test, X_test, y_test = data_processor.get_test_data(data_manager, path_test, files_test)

    print("Predicting results...")
    y_pred_test, y_prob_test, y_scores_test = img_classifier.predict(X_test)

    # 10- Store results in resultado_por_tipo.txt
    print("Generating results...")
    data_processor.generate_results(data_manager, path_test, files_test,
                                    info_X_test, y_pred_test, y_scores_test)
