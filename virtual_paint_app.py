import mediapipe as mp
import cv2
import numpy as np
import time

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

tools = cv2.imread("tools.png")
tools = tools.astype('uint8')
mask = np.ones((480, 640)) * 255
mask = mask.astype('uint8')

cap = cv2.VideoCapture(0)
while True:
    _, frm = cap.read()
    frm = cv2.flip(frm, 1)

    rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
    op = hand_landmark.process(rgb)

    if op.multi_hand_landmarks:
        for hand_landmarks in op.multi_hand_landmarks:
            draw.draw_landmarks(frm, hand_landmarks, hands.HAND_CONNECTIONS)
            x, y = int(hand_landmarks.landmark[8].x * 640), int(hand_landmarks.landmark[8].y * 480)

            # Selección de herramienta
            if x < max_x and y < max_y and x > ml:
                if time_init:
                    ctime = time.time()
                    time_init = False
                ptime = time.time()

                cv2.circle(frm, (x, y), rad, (0, 255, 255), 2)
                rad -= 1

                if (ptime - ctime) > 0.8:
                    curr_tool = getTool(x)
                    print("Estas usando: ", curr_tool)
                    time_init = True
                    rad = 40
            else:
                time_init = True
                rad = 40

            # Dibujar o usar herramienta seleccionada
            if curr_tool == "dibujar":
                xi, yi = int(hand_landmarks.landmark[12].x * 640), int(hand_landmarks.landmark[12].y * 480)
                y9 = int(hand_landmarks.landmark[9].y * 480)

                if index_raised(yi, y9):
                    cv2.line(mask, (prevx, prevy), (x, y), 0, thick)
                prevx, prevy = x, y

            elif curr_tool == "linea":
                if not var_inits:
                    xii, yii = x, y
                    var_inits = True
                cv2.line(frm, (xii, yii), (x, y), (50, 152, 255), thick)

                if index_raised(int(hand_landmarks.landmark[12].y * 480), int(hand_landmarks.landmark[9].y * 480)):
                    cv2.line(mask, (xii, yii), (x, y), 0, thick)
                    var_inits = False

            elif curr_tool == "rectangulo":
                if not var_inits:
                    xii, yii = x, y
                    var_inits = True
                cv2.rectangle(frm, (xii, yii), (x, y), (0, 255, 255), thick)

                if index_raised(int(hand_landmarks.landmark[12].y * 480), int(hand_landmarks.landmark[9].y * 480)):
                    cv2.rectangle(mask, (xii, yii), (x, y), 0, thick)
                    var_inits = False

            elif curr_tool == "circulo":
                if not var_inits:
                    xii, yii = x, y
                    var_inits = True
                cv2.circle(frm, (xii, yii), int(((xii - x) ** 2 + (yii - y) ** 2) ** 0.5), (255, 255, 0), thick)

                if index_raised(int(hand_landmarks.landmark[12].y * 480), int(hand_landmarks.landmark[9].y * 480)):
                    cv2.circle(mask, (xii, yii), int(((xii - x) ** 2 + (yii - y) ** 2) ** 0.5), 0, thick)
                    var_inits = False

            elif curr_tool == "borrador":
                cv2.circle(frm, (x, y), 30, (0, 0, 0), -1)
                cv2.circle(mask, (x, y), 30, 255, -1)

            # Captura de pantalla si hay un puño
            if is_fist(hand_landmarks.landmark):
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
    cv2.putText(frm, curr_tool, (270 + ml, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("paint app", frm)

    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        cap.release()
        break
