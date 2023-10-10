import cv2
import os
import numpy as np

def procesar_imagenes_en_carpetas(directorios_entrada):
    for directorio in directorios_entrada:
        if not os.path.isdir(directorio):
            print(f"El directorio {directorio} no existe.")
            continue

        for nombre_imagen in os.listdir(directorio):
            if nombre_imagen.endswith((".jpg", ".png", ".jpeg")):
                ruta_imagen = os.path.join(directorio, nombre_imagen)

                try:
                    # Leer y procesar la imagen aquí
                    img = cv2.imread(ruta_imagen)

                    # Convertir la imagen a escala de color HSV (Hue, Saturation, Value)
                    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

                    # Definir el rango de colores que deseas aislar (en este ejemplo, verde)
                    rango_bajo = np.array([35, 100, 100])  # Rango bajo de tono, saturación y valor
                    rango_alto = np.array([85, 255, 255])  # Rango alto de tono, saturación y valor

                    # Crear una máscara para aislar los píxeles en el rango de colores deseado
                    mascara = cv2.inRange(img_hsv, rango_bajo, rango_alto)

                    # Invertir la máscara para que el objeto sea blanco y el fondo sea negro
                    mascara_invertida = cv2.bitwise_not(mascara)

                    # Crear una imagen en blanco y negro del mismo tamaño que la imagen original
                    img_fondo = np.zeros_like(img, np.uint8)

                    # Copiar el fondo de la imagen original usando la máscara invertida
                    img_fondo[:] = (0, 0, 0)  # Fondo negro
                    img_sin_fondo = cv2.bitwise_and(img, img, mask=mascara_invertida)
                
                    #Aplicar un filtro de denoising con fastNlMeansDenoising
                    imagen_denoised =  cv2.fastNlMeansDenoisingColored(img_sin_fondo, None, 10, 10, 7, 21)
                
                    #Convertir la imagen a escala de grises
                    imagen_gris = cv2.cvtColor(imagen_denoised, cv2.COLOR_BGR2GRAY)

                    #Aplicar un filtro de denoising con fastNlMeansDenoising
                    #imagen_denoised =  cv2.bilateralFilter(imagen_gris, d=50, sigmaColor=75, sigmaSpace=75)

                    #Aplicar un realce de bordes (operador Laplaciano)
                    imagen_realzada = cv2.Laplacian(imagen_gris, cv2.CV_64F)
                
                    #Ajustar el brillo de la imagen resultante
                    imagen_realzada = cv2.convertScaleAbs(imagen_realzada, alpha=20, beta=45)  # Ajusta alpha y beta según sea necesario Low: 2 and 100 High: 45 and 20

                    # Guardar la imagen procesada en el mismo directorio
                    ruta_salida = os.path.join(directorio, nombre_imagen)
                    cv2.imwrite(ruta_salida, imagen_realzada)

                    print(f"Imagen {nombre_imagen} procesada y guardada en {directorio}")

                except Exception as e:
                    print(f"Error al procesar {nombre_imagen}: {str(e)}")

# Lista de directorios de entrada que contienen las imágenes
directorios_entrada = ["C:/Users/darka/Desktop/archive/with filters/train/Sign_4","C:/Users/darka/Desktop/archive/with filters/train/Sign_5",
                       "C:/Users/darka/Desktop/archive/with filters/train/Sign_6","C:/Users/darka/Desktop/archive/with filters/train/Sign_7",
                       "C:/Users/darka/Desktop/archive/with filters/train/Sign_8","C:/Users/darka/Desktop/archive/with filters/train/Sign_9"]

# Procesar imágenes en las carpetas de forma secuencial
procesar_imagenes_en_carpetas(directorios_entrada)