import cv2
import numpy as np
import os
from pathlib import Path

def contains_circle(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred_image = cv2.GaussianBlur(image, (9, 9), 2)
    circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30, param1=50, param2=30, minRadius=10, maxRadius=100)
    return circles is not None

def segment_coins(image_path, threshold_value, min_contour_area, number_image):
    # Načtení obrázku
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print("Error: Unable to load image.")
        return

    # Převod do šedotónu
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Přidání Gaussova rozmazání pro snížení šumu fotograrie 
    blurred_image = cv2.GaussianBlur(gray_image, (11, 11), 0)

    # Binarizace obrázku s Gausovým rozmazáním
    _, binary_mask = cv2.threshold(blurred_image, threshold_value, 255, cv2.THRESH_BINARY_INV)

    # Nalezení obrysů v již binárním obrázku
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Vytvoření složky pro uložení vyříznutých mincí
    os.chdir(r"L:\CVUT\1roc\projekt1\seperatecoins") # Pro funkčnost přepsat cestu do složky s rozdělenými mincemi
    os.makedirs(f"image{number_image}")
    os.chdir(f"image{number_image}")

    # Procházení všech obrysů a vyřezávání mincí
    coin_count = 0
    for i, contour in enumerate(contours):
        # Ohlédnutí do zadané proměnné min_contour_area pro vyzaření falešných detekcí
        if cv2.contourArea(contour) < min_contour_area:
            continue

        # Vykreslení masky pro danou minci
        coin_mask = np.zeros_like(gray_image)
        cv2.drawContours(coin_mask, [contour], -1, 255, thickness=cv2.FILLED)

        # Aplikace masky na obrázek pro extrakci mince
        segmented_coin = cv2.bitwise_and(image, image, mask=coin_mask)

        # Vyříznutí mince z obrázku
        x, y, w, h = cv2.boundingRect(contour)
        cropped_coin = segmented_coin[y:y+h, x:x+w]

        # Uložení vyříznuté mince
        coin_count += 1
        cv2.imwrite(f"coin_{coin_count}.jpg", cropped_coin)

        # Kontrola, zda obrázek obsahuje kruh
        if not contains_circle(f"coin_{coin_count}.jpg"):
            os.remove(f"coin_{coin_count}.jpg")


    print(f"Z obrázků segmentováno a uloženo {coin_count} mincí.\n")

if __name__ == "__main__":

    # Zeptání se uživatele na cestu k složce
    folder_path = input("Vyberte složku s obrázky, které chcete segmentovat: \n")

    # Pokus o změnu aktuálního pracovního adresáře na zadanou složku
    try:
        os.chdir(folder_path)
        print(f"Aktuální pracovní adresář byl změněn na: {os.getcwd()}")
    except FileNotFoundError:
        print("Chyba: Zadaná cesta neexistuje.")
        exit(1)
    except NotADirectoryError:
        print("Chyba: Zadaná cesta není složka.")
        exit(1)
    except PermissionError:
        print("Chyba: Nemáte oprávnění přistupovat k této složce.")
        exit(1)

    list_images = os.listdir()
    list_images = [x for x in list_images if x.endswith(".jpg")]

    for j in range(len(list_images)):
        os.chdir(folder_path)
        segment_coins(list_images[j], 125, 300, j)
        #print("Zadejte cestu k obrázku, hodnotu pro binarizaci (doporučená 200), minimální obrys mince (doporučená 500). [Vstupní hodnoty oddělujte enterem.]")
        #segment_coins(input(), int(input()), int(input()), j)

    #segment_coins("testimage4.jpg", 200, 500, 1) # (200, 500 - testimage4.jpg)