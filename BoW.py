import argparse
import cv2
import numpy as np 
import os
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

def getFiles(train, path):
    images = []
    count = 0
    for folder in os.listdir(path):
        for file in  os.listdir(os.path.join(path, folder)):
            images.append(os.path.join(path, os.path.join(folder, file)))

    if(train is True):
        np.random.shuffle(images)
    
    return images

def getDescriptors(sift, img):
    kp, des = sift.detectAndCompute(img, None)
    return des

def readImage(img_path):
    img = cv2.imread(img_path, 0)
    return img

def vstackDescriptors(descriptor_list):
    descriptors = np.array(descriptor_list[0])
    for descriptor in descriptor_list[1:]:
        descriptors = np.vstack((descriptors, descriptor)) 

    return descriptors

def clusterDescriptors(descriptors, no_clusters):
    kmeans = KMeans(n_clusters = no_clusters).fit(descriptors)
    return kmeans

def extractFeatures(kmeans, descriptor_list, image_count, no_clusters):
    im_features = np.array([np.zeros(no_clusters) for i in range(image_count)])
    for i in range(image_count):
        for j in range(len(descriptor_list[i])):
            feature = descriptor_list[i][j]
            feature = feature.reshape(1, 128)
            idx = kmeans.predict(feature)
            im_features[i][idx] += 1

    return im_features

def normalizeFeatures(scale, features):
    return scale.transform(features)

def plotHistogram(im_features, no_clusters):
    x_scalar = np.arange(no_clusters)
    y_scalar = np.array([abs(np.sum(im_features[:,h], dtype=np.int32)) for h in range(no_clusters)])

    plt.bar(x_scalar, y_scalar)
    plt.xlabel("Índice visual de palabras")
    plt.ylabel("Frecuencia")
    plt.title("Vocabulario completo generado")
    plt.xticks(x_scalar + 0.4, x_scalar)
    plt.show()

def svcParamSelection(X, y, kernel, nfolds):
    Cs = [0.5, 0.1, 0.15, 0.2, 0.3]
    gammas = [0.1, 0.11, 0.095, 0.105]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(SVC(kernel=kernel), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_

def findSVM(im_features, train_labels, kernel):
    features = im_features
    if(kernel == "Precomputado"):
      features = np.dot(im_features, im_features.T)
    
    params = svcParamSelection(features, train_labels, kernel, 5)
    C_param, gamma_param = params.get("C"), params.get("gamma")
    print(C_param, gamma_param)
    class_weight = {
        0: (807 / (10 * 140)),
        1: (807 / (10 * 140)),
        2: (807 / (10 * 133)),
        3: (807 / (10 * 70)),
        4: (807 / (10 * 42)),
        5: (807 / (10 * 140)),
        6: (807 / (10 * 142)),
        7: (807 / (10 * 142)),
        8: (807 / (10 * 142)),
        9: (807 / (10 * 142))
    }
  
    svm = SVC(kernel = kernel, C =  C_param, gamma = gamma_param, class_weight = class_weight)
    svm.fit(features, train_labels)
    return svm

def plotConfusionMatrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Matriz de confusión normalizada'
        else:
            title = 'Matriz de confusión, sin normalización'

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Matriz de confusión normalizada")
    else:
        print('Matriz de confusión, sin normalización')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Etiqueta verdadera',
           xlabel='Etiqueta predicha')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def plotConfusions(true, predictions):
    
    np.set_printoptions(precision=2)

    class_names = ["cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve"]
    plotConfusionMatrix(true, predictions, classes=class_names, title='Matriz de confusión, sin normalización')

    plotConfusionMatrix(true, predictions, classes=class_names, normalize=True, title='Matriz de confusión normalizada')

    plt.show()

def findAccuracy(true, predictions):
    print ('Puntuación de precisión: %0.3f' % accuracy_score(true, predictions))

def trainModel(path, no_clusters, kernel):
    images = getFiles(True, path)
    print("Ruta de las imágenes de entrenamiento detectadas.")
    sift = cv2.xfeatures2d.SIFT_create()
    descriptor_list = []
    train_labels = np.array([])
    label_count = 10
    image_count = len(images)

    for img_path in images:
        if("Sign_0" in img_path):
            class_index = 0
        elif("Sign_1" in img_path):
            class_index = 1
        elif("Sign_2" in img_path):
            class_index = 2
        elif("Sign_3" in img_path):
            class_index = 3
        elif("Sign_4" in img_path):
            class_index = 4
        elif("Sign_5" in img_path):
            class_index = 5
        elif("Sign_6" in img_path):
            class_index = 6
        elif("Sign_7" in img_path):
            class_index = 7
        elif("Sign_8" in img_path):
            class_index = 8
        else:
            class_index = 9

        train_labels = np.append(train_labels, class_index)
        img = readImage(img_path)
        des = getDescriptors(sift, img)
        descriptor_list.append(des)

    descriptors = vstackDescriptors(descriptor_list)
    print("Descriptores Stackeados")

    kmeans = clusterDescriptors(descriptors, no_clusters)
    print("Descriptores agrupados.")

    im_features = extractFeatures(kmeans, descriptor_list, image_count, no_clusters)
    print("Características de las imágenes extraídas.")

    scale = StandardScaler().fit(im_features)        
    im_features = scale.transform(im_features)
    print("Imágenes de entrenamiento normalizadas.")

    plotHistogram(im_features, no_clusters)
    print("Histograma de características trazado.")

    svm = findSVM(im_features, train_labels, kernel)
    print("SVM ajustado.")
    print("Entrenamiento completado")

    return kmeans, scale, svm, im_features

def testModel(path, kmeans, scale, svm, im_features, no_clusters, kernel):
    test_images = getFiles(False, path)
    print("Imagenes de Prueba detectadas")

    count = 0
    true = []
    descriptor_list = []

    name_dict =	{
        "0": "Sign_0",
        "1": "Sign_1",
        "2": "Sign_2",
        "3": "Sign_3",
        "4": "Sign_4",
        "5": "Sign_5",
        "6": "Sign_6",
        "7": "Sign_7",
        "8": "Sign_8",
        "9": "Sign_9"
    }

    sift = cv2.xfeatures2d.SIFT_create()

    for img_path in test_images:
        img = readImage(img_path)
        des = getDescriptors(sift, img)

        if(des is not None):
            count += 1
            descriptor_list.append(des)

            if("Sign_0" in img_path):
                true.append("Sign_0")
            elif("Sign_1" in img_path):
                true.append("Sign_1")
            elif("Sign_2" in img_path):
                true.append("Sign_2")
            elif("Sign_3" in img_path):
                true.append("Sign_3")
            elif("Sign_4" in img_path):
                true.append("Sign_4")
            elif("Sign_5" in img_path):
                true.append("Sign_5")
            elif("Sign_6" in img_path):
                true.append("Sign_6")
            elif("Sign_7" in img_path):
                true.append("Sign_7")
            elif("Sign_8" in img_path):
                true.append("Sign_8")
            else:
                true.append("Sign_9")

    descriptors = vstackDescriptors(descriptor_list)

    test_features = extractFeatures(kmeans, descriptor_list, count, no_clusters)

    test_features = scale.transform(test_features)
    
    kernel_test = test_features
    if(kernel == "Precomputado"):
        kernel_test = np.dot(test_features, im_features.T)
    
    predictions = [name_dict[str(int(i))] for i in svm.predict(kernel_test)]
    print("Imagenes de prueba clasificadas.")

    plotConfusions(true, predictions)
    print("Matrices de confusión trazadas.")

    findAccuracy(true, predictions)
    print("Exactitud calculada.")
    print("Ejecucion completada")

def execute(train_path, test_path, no_clusters, kernel):
    kmeans, scale, svm, im_features = trainModel(train_path, no_clusters, kernel)
    testModel(test_path, kmeans, scale, svm, im_features, no_clusters, kernel)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--train_path', action="store", dest="train_path", default="C:/Users/darka/Desktop/archive/with filters/train")
    parser.add_argument('--test_path', action="store", dest="test_path", default="C:/Users/darka/Desktop/archive/with filters/test")
    ##parser.add_argument('--train_path', action="store", dest="train_path", default="dataset/ASL Digits/train")
    ##parser.add_argument('--test_path', action="store", dest="test_path", default="dataset/ASL Digits/test")
    parser.add_argument('--no_clusters', action="store", dest="no_clusters", default=100)
    parser.add_argument('--kernel_type', action="store", dest="kernel_type", default="linear")

    args =  vars(parser.parse_args())
    if(not(args['kernel_type'] == "linear" or args['kernel_type'] == "precomputed")):
        print("Kernel type must be either linear or precomputed")
        exit(0)

    execute(args['train_path'], args['test_path'], int(args['no_clusters']), args['kernel_type'])
