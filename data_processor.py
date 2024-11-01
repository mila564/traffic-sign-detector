import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split


class DataProcessor:
    def __init__(self):
        self.mser_detector = cv2.MSER_create(delta=2, max_variation=0.8)

        self.hog = cv2.HOGDescriptor(_winSize=(32, 32), _blockSize=(8, 8), _blockStride=(4, 4), _cellSize=(4, 4),
                                     _nbins=9)

    @staticmethod
    def is_square(w, h):
        if h < w:
            return 0.6 <= h / w
        return 0.6 <= w / h

    @staticmethod
    def crop(np_image, x_point, y_point, w, h):
        x2 = x_point + w
        y2 = y_point + h
        resized_detection = np_image[y_point:y2, x_point:x2]
        return cv2.resize(resized_detection, (32, 32))

    # Code from  https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    @staticmethod
    def get_iou(box_a, box_b):
        # Cetermine the (x, y)-coordinates of the intersection rectangle
        x_a = max(box_a[0], box_b[0])
        y_a = max(box_a[1], box_b[1])
        x_b = min(box_a[2], box_b[2])
        y_b = min(box_a[3], box_b[3])
        # Compute the area of intersection rectangle
        inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)
        # Compute the area of both the prediction and ground-truth
        # Rectangles
        box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
        box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)
        # Compute the intersection over union by taking the intersection
        # Area and dividing it by the sum of prediction + ground-truth
        # areas - the intersection area
        iou = inter_area / float(box_a_area + box_b_area - inter_area)
        # Return the intersection over union value
        return iou

    def create_positive_and_negative_subsets(self, dta_manager, path, files):
        print("     Creating positive and negative subsets...")
        gt_dictionary = dta_manager.load_gt(path)  # Load signal information from gt.txt file
        c = [[], [], [], [], [], [], []]  # toReturn
        for file in files:
            image = dta_manager.load_image(path, file)
            # 2- Get high contrast regions from train images using MSER detector
            detected_regions, _ = self.mser_detector.detectRegions(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            for detection in detected_regions:
                x, y, width, height = cv2.boundingRect(detection)
                file_name = file.split(".")[0]
                # Filtering detections
                if self.is_square(width, height) and file_name in gt_dictionary:
                    bounding_boxes = gt_dictionary[file_name]
                    max_iou = 0
                    max_type = 0
                    for bounding_box in bounding_boxes:
                        iou = self.get_iou(bounding_box[0], (x, y, x + width, y + height))
                        if iou > max_iou:
                            max_iou = iou
                            max_type = bounding_box[1]
                    # Get detection using crop method
                    cropped_detection = self.crop(image, x, y, width, height)

                    # Compute gradients hist
                    hist = self.hog.compute(cropped_detection)

                    # Generate train data
                    # Negative class
                    if max_iou < 0.3:
                        c[0].append((hist, (file, (x, y, x + width, y + height))))
                    # Positive class
                    elif max_iou >= 0.5:
                        c[max_type].append((hist, (file, (x, y, x + width, y + height))))
                        # ------prevent underfitting ------
                        # Generating new cropped detections
                        if max_type == 2:  # Warning
                            for i in range(2):
                                c[max_type].append((self.hog.compute(self.crop(image, x, y, width, height)),
                                                    (file, (x, y, x + width, y + height))))
                        elif max_type == 3:  # Stop
                            for i in range(15):
                                c[max_type].append((self.hog.compute(self.crop(image, x, y, width, height)),
                                                    (file, (x, y, x + width, y + height))))
                        elif max_type == 4:  # Forbidden direction
                            for i in range(15):
                                c[max_type].append((self.hog.compute(self.crop(image, x, y, width, height)),
                                                    (file, (x, y, x + width, y + height))))
                        elif max_type == 5:  # Yield
                            for i in range(5):
                                c[max_type].append((self.hog.compute(self.crop(image, x, y, width, height)),
                                                    (file, (x, y, x + width, y + height))))
                        elif max_type == 6:  # Mandatory
                            for i in range(4):
                                c[max_type].append((self.hog.compute(self.crop(image, x, y, width, height)),
                                                    (file, (x, y, x + width, y + height))))
                # ------Prevent underfitting ------
            # ------Prevent overfitting ------
        c[0], _, _, _ = sklearn.model_selection.train_test_split(c[0], [0] * len(c[0]), train_size=0.05)
        c[0] = list(np.array(c[0], dtype="object"))
        c[1] = list(np.array(c[1], dtype="object"))
        c[2] = list(np.array(c[2], dtype="object"))
        c[3] = list(np.array(c[3], dtype="object"))
        c[4] = list(np.array(c[4], dtype="object"))
        c[5] = list(np.array(c[5], dtype="object"))
        c[6] = list(np.array(c[6], dtype="object"))
        # ------prevent overfitting ------
        return c

    # Generates X,y data to train

    @staticmethod
    def create_X_y_binary_datasets(group):
        print("     Creating data sets...")
        X = [[], [], [], [], [], []]
        y = [[], [], [], [], [], []]

        for hist in group[0]:
            for i in range(len(X)):
                X[i].append(hist)
                y[i].append(0)

        for i in range(1, len(group)):
            for hist in group[i]:
                X[i - 1].append(hist)
                y[i - 1].append(i)

        return X, y

    @staticmethod
    def create_X_y_multiclass_datasets(group):
        print("     Creating datasets...")
        X = []
        y = []
        for i in range(len(group)):
            for j in range(len(group[i])):
                X.append(group[i][j])
                y.append(i)
        return X, y

    # Separate data

    @staticmethod
    def split_data_binary(X, y):
        print("     Splitting data...")
        X_train = [[], [], [], [], [], []]
        y_train = [[], [], [], [], [], []]
        X_val = [[], [], [], [], [], []]
        y_val = [[], [], [], [], [], []]

        for i in range(len(X)):
            X_train[i], X_val[i], y_train[i], y_val[i] = train_test_split(X[i], y[i], test_size=0.1, train_size=0.9)

        return X_train, X_val, y_train, y_val

    @staticmethod
    def split_data_multiclass(X, y):
        print("     Splitting data...")
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, train_size=0.9)
        return X_train, X_val, y_train, y_val

    def get_train_data(self, dta_manager, path, files, type_classifier):
        c = self.create_positive_and_negative_subsets(dta_manager, path, files)
        if type_classifier == "HOG_LDA_BAYES":
            X, y = self.create_X_y_binary_datasets(c)
            X_train, X_val, y_train, y_val = self.split_data_binary(X, y)
            X_train_data, X_train_info = self.separate_data_binary(X_train)
            X_val_data, X_val_info = self.separate_data_binary(X_val)
            X_data, y_data = self.combine_data_binary(X_train_data, X_val_data, y_train, y_val)
            # Flat and filter no signals in validation dataset
            X_val_data_flatten, y_val_flatten, _ = self.flat_data(
                X_val_data,
                y_val,
                X_val_info)
            return X_data, y_data, X_train_data, y_train, \
                X_val_data_flatten, y_val_flatten
        else:  # HOG_PCA_KNN
            X, y = self.create_X_y_multiclass_datasets(c)
            X_train, X_val, y_train, y_val = self.split_data_multiclass(X, y)
            X_train_data, X_train_info = self.separate_data_multiclass(X_train)
            X_val_data, X_val_info = self.separate_data_multiclass(X_val)
            X_data, y_data = self.combine_data_multiclass(X_train_data, X_val_data, y_train, y_val)
            return X_data, y_data, X_train_data, y_train, X_val_data, y_val

    def get_test_data(self, dta_manager, path, files):
        c = self.create_positive_and_negative_subsets(dta_manager, path, files)
        X_test, y_test = self.create_X_y_multiclass_datasets(c)
        X_test_data, X_test_info = self.separate_data_multiclass(X_test)
        return X_test_info, X_test_data, y_test

    # Delete duplicates
    def delete_duplicates(self, detection_list_with_duplicates):
        detection_list = list()
        for detection in detection_list_with_duplicates:
            duplicates = list()
            max_score = 0
            max_detection = tuple()
            # Calculate iou for the detections
            for duplicate in detection_list_with_duplicates:
                iou = self.get_iou(detection[0], duplicate[0])
                if iou > 0.01:
                    duplicates.append(duplicate)
            # Get the region which has the highest score
            for final_detection in duplicates:
                if final_detection[1] > max_score:
                    max_score = final_detection[1]
                    max_detection = final_detection
            if max_detection not in detection_list:
                detection_list.append(max_detection)
        return detection_list

    # Frame signal detections in image
    @staticmethod
    def draw_final_detections(image, file_name, txt_results, detections):
        for detection in detections:
            type_signal = detection[2]
            score = round(detection[1], ndigits=2)
            x, y, x2, y2 = detection[0]
            # Resize the square
            width = abs(x - x2)
            height = abs(y - y2)
            if height > width:
                width_inc = round(height * 0.25)
                height_inc = round(height * 0.25)
            else:
                width_inc = round(width * 0.25)
                height_inc = round(width * 0.25)
            # Print the results
            # Depending on type signal, the terminal will show the corresponding name
            if type_signal == 1:
                name_type_signal = "Forbidden signal"
            elif type_signal == 2:
                name_type_signal = "Warning signal"
            elif type_signal == 3:
                name_type_signal = "Stop signal"
            elif type_signal == 4:
                name_type_signal = "Forbidden direction signal"
            elif type_signal == 5:
                name_type_signal = "Yield signal"
            else:
                name_type_signal = "Mandatory direction signal"

            print("     Image ", file_name, ":", name_type_signal, " detected with score ", score, sep="")

            result = ';'.join([str(e) for e in [file_name, x, y, x2, y2, type_signal, score]])
            txt_results.write(result + "\n")
            # Draw the square on the original image
            cv2.rectangle(image, (x - width_inc, y - height_inc),
                          (x + width + width_inc, y + height + height_inc), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            # Print the type signal on the original image
            cv2.putText(image, str(type_signal),
                        (x - width_inc, y - height_inc - 10),
                        font, fontScale=1,
                        color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
            cv2.putText(image, str(score),
                        (x + width_inc + width - 50, y + height_inc + height + 25),
                        font, fontScale=0.75,
                        color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

    # Used to sort image detections by file name
    @staticmethod
    def sort_criteria(e):
        return e[0]

    #  Draw results + generate results.txt
    def generate_results(self, data_manager, path_test, files_test,
                         info_X, y_pred, y_scores):

        # Get detection list from info_X, y_pred and y_scores

        info_detections_matrix = np.array([y_pred, y_scores, info_X], dtype="object")
        detection_list_with_duplicates = []
        for i in range(info_detections_matrix.shape[1]):
            column = info_detections_matrix[:, i]
            if column[0] != 0 and column[1] > 0.5:
                detection_list_with_duplicates.append(column)
        detection_list_with_duplicates.sort(key=self.sort_criteria)
        dict_signals = self.init_dict_signals(files_test)
        for i in range(len(detection_list_with_duplicates)):
            type_signal = detection_list_with_duplicates[i][0]
            score = detection_list_with_duplicates[i][1]
            file_name = detection_list_with_duplicates[i][2][0]
            x, y, x2, y2 = detection_list_with_duplicates[i][2][1]
            dict_signals[file_name].append(((x, y, x2, y2), score, type_signal))

        # Filter the duplicate detections + Draw results + Generate results.txt

        results_path = "resultado_imgs/"
        results_txt = data_manager.load_and_open_txt_file(results_path)

        for file_name in dict_signals:
            detection_list = self.delete_duplicates(dict_signals[file_name])
            image = data_manager.load_image(path_test, file_name)
            self.draw_final_detections(image, file_name, results_txt, detection_list)
            data_manager.save_image(results_path, image, file_name)

    @staticmethod
    def flat_data(X, y, info_X):

        info_X_to_return = []
        X_to_return = []
        y_to_return = []

        for i in range(6):
            for j in range(len(y[i])):
                info_X_to_return.append(info_X[i][j])
                X_to_return.append(X[i][j])
                y_to_return.append(y[i][j])

        return np.array(X_to_return), y_to_return, info_X_to_return

    @staticmethod
    def separate_data_binary(X):
        X_data = []
        X_info = []
        for i in range(len(X)):
            X_data.append([])
            X_info.append([])
            for j in range(len(X[i])):
                data, info = X[i][j]
                X_data[i].append(data)
                X_info[i].append(info)
        return X_data, X_info

    @staticmethod
    def separate_data_multiclass(X):
        X_data = []
        X_info = []
        for i in range(len(X)):
            data = X[i][0]
            info = X[i][1]
            X_data.append(data)
            X_info.append(info)
        return X_data, X_info

    @staticmethod
    def init_dict_signals(files_test):
        d_signals = dict()
        for file in files_test:
            d_signals[file] = []
        return d_signals

    @staticmethod
    def combine_data_binary(X_train_data, X_val_data, y_train, y_val):
        # listas de 6 elementos (uno por cada señal) con numpy arrays con los hogs
        X_data = []
        y_data = []
        for i in range(6):
            X_data.append([])
            y_data.append([])
            X_data[i] = X_train_data[i] + X_val_data[i]
            y_data[i] = y_train[i] + y_val[i]
        return X_data, y_data

    @staticmethod
    def combine_data_multiclass(X_train_data, X_val_data, y_train, y_val):
        return X_train_data+X_val_data, y_train+y_val
