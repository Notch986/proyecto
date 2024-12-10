import mediapipe as mp
import cv2
import numpy as np
import time
import tkinter as tk
from tkinter import filedialog

# Configuración inicial
ml = 150
max_x, max_y = 250 + ml, 50
curr_tool = "seleccione herramienta"
time_init = True
rad = 40
var_inits = False
thick = 4
prevx, prevy = 0, 0
screenshot_counter = 0  # Contador para las capturas de pantalla

# Variable para almacenar la imagen seleccionada y sus coordenadas iniciales
selected_img = None
img_coords = [0, 0]
img_scale = 0.7  # Escala inicial de la imagen
img_add = False
roi_img = 0
alto = 0
ancho = 0
bgr_icon_img = 0
alpha_channel_img = 0

def getTool(x):
    if x < 50 + ml:
        return "linea"
    elif x < 100 + ml:
        return "rectangulo"
    elif x < 150 + ml:
        return "dibujar"
    elif x < 200 + ml:
        return "circulo"
    else:
        return "borrador"

def index_raised(yi, y9):
    return (y9 - yi) > 40

def is_fist(landmarks):
    """
    Determina si la mano está en forma de puño.
    """
    finger_tips = [8, 12, 16, 20]  # Índices de las puntas de los dedos
    for tip in finger_tips:
        if landmarks[tip].y < landmarks[tip - 2].y:  # Si cualquier punta no está cerrada
            return False
    return True

# Inicialización de MediaPipe y OpenCV
hands = mp.solutions.hands
hand_landmark = hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=1)
draw = mp.solutions.drawing_utils

# Cargar herramientas y máscaras
tools = cv2.imread("tools.png")
tools = tools.astype('uint8')
mask = np.ones((480, 640)) * 255
mask = mask.astype('uint8')

# Cargar el icono para añadir imágenes png
icon_add_img = cv2.imread("add_imagen.png", cv2.IMREAD_UNCHANGED)
if icon_add_img.shape[-1] == 4:  # Verificar si tiene 4 canales (RGBA)
    icon_add_img = cv2.cvtColor(icon_add_img, cv2.COLOR_BGRA2RGBA)

# Separar los canales de la imagen
bgr_icon = icon_add_img[:, :, :3]  # Canales RGB
alpha_channel = icon_add_img[:, :, 3] / 255.0  # Canal alfa normalizado a [0, 1]

# Cargar el icono para borrar la imagen
icon_delete_img = cv2.imread("delete_imagen.png")
icon_delete_img = icon_delete_img.astype('uint8')

# Coordenadas para el icono
icon_x, icon_y = 10, 10

cap = cv2.VideoCapture(0)
while True:
    _, frm = cap.read()
    frm = cv2.flip(frm, 1)

    rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
    op = hand_landmark.process(rgb)

    if op.multi_hand_landmarks:
        for i in op.multi_hand_landmarks:
            draw.draw_landmarks(frm, i, hands.HAND_CONNECTIONS)
            x, y = int(i.landmark[8].x * 640), int(i.landmark[8].y * 480)

            # Detectar clic en el icono de añadir imagen
            if x < icon_x + 50 and y < icon_y + 50:
                if time_init:
                    ctime = time.time()
                    time_init = False
                ptime = time.time()

                cv2.circle(frm, (x, y), rad, (0, 255, 255), 2)
                rad -= 1

                if (ptime - ctime) > 0.8:
                    curr_tool = "imagen"
                    print("Herramienta seleccionada: Añadir imagen")
                    time_init = True
                    rad = 40

            # Selección de herramienta original
            elif x < max_x and y < max_y and x > ml:
                if time_init:
                    ctime = time.time()
                    time_init = False
                ptime = time.time()

                cv2.circle(frm, (x, y), rad, (0, 255, 255), 2)
                rad -= 1

                if (ptime - ctime) > 0.8:
                    curr_tool = getTool(x)
                    print("Estás usando: ", curr_tool)
                    time_init = True
                    rad = 40

            # Botón de eliminar imagen
            elif img_add and x < icon_x + 50 and y < icon_y + 100 and y > icon_y + 50:
                if time_init:
                    ctime = time.time()
                    time_init = False
                ptime = time.time()

                cv2.circle(frm, (x, y), rad, (0, 255, 255), 2)
                rad -= 1

                if (ptime - ctime) > 0.8:
                    curr_tool = "seleccione herramienta"
                    print("Imagen Eliminada")
                    selected_img = None
                    img_add = False

            else:
                time_init = True
                rad = 40

            # Herramientas de dibujo
            if curr_tool == "dibujar":
                xi, yi = int(i.landmark[12].x * 640), int(i.landmark[12].y * 480)
                y9 = int(i.landmark[9].y * 480)

                if index_raised(yi, y9):
                    cv2.line(mask, (prevx, prevy), (x, y), 0, thick)
                prevx, prevy = x, y

            elif curr_tool == "linea":
                xi, yi = int(i.landmark[12].x * 640), int(i.landmark[12].y * 480)
                y9 = int(i.landmark[9].y * 480)

                if index_raised(yi, y9):
                    if not var_inits:
                        xii, yii = x, y
                        var_inits = True

                    cv2.line(frm, (xii, yii), (x, y), (50, 152, 255), thick)

                else:
                    if var_inits:
                        cv2.line(mask, (xii, yii), (x, y), 0, thick)
                        var_inits = False

            elif curr_tool == "rectangulo":
                xi, yi = int(i.landmark[12].x * 640), int(i.landmark[12].y * 480)
                y9 = int(i.landmark[9].y * 480)

                if index_raised(yi, y9):
                    if not var_inits:
                        xii, yii = x, y
                        var_inits = True

                    cv2.rectangle(frm, (xii, yii), (x, y), (0, 255, 255), thick)

                else:
                    if var_inits:
                        cv2.rectangle(mask, (xii, yii), (x, y), 0, thick)
                        var_inits = False

            elif curr_tool == "circulo":
                xi, yi = int(i.landmark[12].x * 640), int(i.landmark[12].y * 480)
                y9 = int(i.landmark[9].y * 480)

                if index_raised(yi, y9):
                    if not var_inits:
                        xii, yii = x, y
                        var_inits = True

                    cv2.circle(frm, (xii, yii), int(((xii - x) ** 2 + (yii - y) ** 2) ** 0.5), (255, 255, 0), thick)

                else:
                    if var_inits:
                        cv2.circle(mask, (xii, yii), int(((xii - x) ** 2 + (yii - y) ** 2) ** 0.5), 0, thick)
                        var_inits = False

            elif curr_tool == "borrador":
                xi, yi = int(i.landmark[12].x * 640), int(i.landmark[12].y * 480)
                y9 = int(i.landmark[9].y * 480)

                if index_raised(yi, y9):
                    cv2.circle(frm, (x, y), 30, (0, 0, 0), -1)
                    cv2.circle(mask, (x, y), 30, 255, -1)

            # Añadir imagen
            elif curr_tool == "imagen" and selected_img is None:
                root = tk.Tk()
                root.withdraw()  # Oculta la ventana principal
                img_path = filedialog.askopenfilename(
                    initialdir="imagenes", 
                    title="Seleccionar Imagen", 
                    filetypes=(("Archivos de Imagen", "*.png;*.jpg;*.jpeg"),)
                )
                
                if img_path:
                    selected_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                    if selected_img is None:
                        print("Error: La imagen no se cargó correctamente")
                    else:
                        h, w, _ = selected_img.shape
                        ancho = int(w * img_scale)
                        alto = int(h * img_scale)
                        selected_img = cv2.resize(selected_img, (ancho, alto))
                        
                        if selected_img.shape[-1] == 4:  # Verificar si tiene 4 canales (RGBA)
                            selected_img = cv2.cvtColor(selected_img, cv2.COLOR_BGRA2RGBA)
                            # Separar los canales de la imagen
                            bgr_icon_img = selected_img[:, :, :3]  # Canales RGB
                            alpha_channel_img = selected_img[:, :, 3] / 255.0  # Canal alfa normalizado a [0, 1]		
                        else:
                            roi_img = selected_img
                        img_add = True

            # Captura de pantalla si hay un puño
            if is_fist(i.landmark):
                # Extraer solo el área del lienzo (el área de dibujo)
                canvas = cv2.bitwise_and(frm, frm, mask=cv2.bitwise_not(mask))

                # Guardar la captura del lienzo como una imagen
                screenshot_filename = f"canvas_{screenshot_counter}.png"
                cv2.imwrite(screenshot_filename, canvas)
                print(f"¡Captura del lienzo guardada como {screenshot_filename}!")
                screenshot_counter += 1
                time.sleep(1)

    op = cv2.bitwise_and(frm, frm, mask=mask)
    frm[:, :, 1] = op[:, :, 1]
    frm[:, :, 2] = op[:, :, 2]

    frm[:max_y, ml:max_x] = cv2.addWeighted(tools, 0.7, frm[:max_y, ml:max_x], 0.3, 0)
    
    roi = frm[icon_y:icon_y+50, icon_x:icon_x+50]
    # Mezclar usando el canal alfa
    for c in range(3):  # Iterar sobre B, G, R
        roi[:, :, c] = roi[:, :, c] * (1 - alpha_channel) + bgr_icon[:, :, c] * alpha_channel
    frm[icon_y:icon_y+50, icon_x:icon_x+50] = roi

    if img_add:
        roi_img = frm[50:alto+50, 50:ancho+50]
        for c in range(3):  # Iterar sobre B, G, R
            roi_img[:, :, c] = roi_img[:, :, c] * (1 - alpha_channel_img) + bgr_icon_img[:, :, c] * alpha_channel_img
        frm[50:alto+50, 50:ancho+50] = roi_img
        frm[icon_y+50:icon_y+100, icon_x:icon_x+50] = icon_delete_img

    cv2.putText(frm, curr_tool, (270 + ml, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("paint app", frm)

    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        cap.release()
        break