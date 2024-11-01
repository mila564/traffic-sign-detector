import pickle
import shutil
import warnings
import cv2
import os


class DataManager:

    def __init__(self):
        path_results = "resultado_imgs"
        path_serialized_classifier = "classifiers"

        if os.path.exists(path_results):
            shutil.rmtree(path_results)

        os.makedirs(path_results)

        file = os.path.join(path_results, "resultado_por_tipo.txt")
        if os.path.exists(file):
            os.remove(file)

        if not os.path.exists(path_serialized_classifier):
            os.makedirs(path_serialized_classifier)

    @staticmethod
    def load_gt(path):
        try:
            gt_dict = dict()
            fd_gt = open(os.path.join(path, "gt.txt"), "r")
            for line in fd_gt:
                line_list = line.split(";")
                file = line_list[0].split(".")[0]
                real_bounding_box = [int(element) for element in line_list[1:-1]]
                if file not in gt_dict:
                    gt_dict[file] = []
                type_signal = 0
                folder_number = line_list[-1].replace("\n", "")
                if folder_number in ["0", "1", "2", "3", "4", "5", "7", "8", "9", "10", "15", "16"]:
                    type_signal = 1
                elif folder_number in ["11", "18", "19", "20", "21", "22", "23", "24", "25",
                                       "26", "27", "28", "29", "30", "31"]:
                    type_signal = 2
                elif folder_number == "14":
                    type_signal = 3
                elif folder_number == "17":
                    type_signal = 4
                elif folder_number == "13":
                    type_signal = 5
                elif folder_number == "38":
                    type_signal = 6
                if type_signal != 0:
                    gt_dict[file].append((tuple(real_bounding_box), type_signal))
            return gt_dict
        except FileNotFoundError:
            print("Error. The gt.txt file does not exist in the current folder")
            quit()

    @staticmethod
    def list_directory(path):
        return_list = [f for f in os.listdir(path) if ".jpg" in f]
        return_list.sort()
        return return_list

    @staticmethod
    def load_image(path, file_name):
        return cv2.imread(os.path.join(path, file_name))

    @staticmethod
    def save_image(path, np_image, file_name):
        suffix = "results_{}".format(file_name)
        cv2.imwrite(os.path.join(path, suffix), np_image)

    @staticmethod
    def load_and_open_txt_file(path):
        return open(os.path.join(path, "resultado_por_tipo.txt"), "a")

    @staticmethod
    def exists(path):
        return os.path.exists(path)

    @staticmethod
    def join_path(path, file):
        return os.path.join(path, file)

    @staticmethod
    def load_classifier(file):
        infile = open(file, "rb")
        warnings.filterwarnings('ignore')
        classifier = pickle.load(infile)
        infile.close()
        return classifier

    @staticmethod
    def save_classifier(file, classifier):
        outfile = open(file, "wb")
        warnings.filterwarnings('ignore')
        pickle.dump(classifier, outfile)
        outfile.close()
