import mediapipe as mp
import cv2
import numpy as np
import time

class AplicacionDibujo:
    def __init__(self):
        # Constantes de configuración
        self.ml = 150
        self.max_x, self.max_y = 250 + self.ml, 50
        self.herramienta_actual = "seleccione herramienta"
        self.tiempo_inicializado = True
        self.radio = 40
        self.var_iniciales = False
        self.grosor = 4
        self.prevx, self.prevy = 0, 0

        # Configuración de MediaPipe
        self.manos = mp.solutions.hands
        self.detector_manos = self.manos.Hands(
            min_detection_confidence=0.6, 
            min_tracking_confidence=0.6, 
            max_num_hands=1
        )
        self.dibujante = mp.solutions.drawing_utils

        # Preparación de máscaras y herramientas
        self.herramientas = cv2.imread("tools.png")
        self.herramientas = self.herramientas.astype('uint8')
        self.mascara = np.ones((480, 640)) * 255
        self.mascara = self.mascara.astype('uint8')

    def obtener_herramienta(self, x):
        """Determina la herramienta según la posición x"""
        secciones = [
            (50, "linea"),
            (100, "rectangulo"),
            (150, "dibujar"),
            (200, "circulo"),
            (250, "borrador")
        ]
        
        for limite, herramienta in secciones:
            if x < limite + self.ml:
                return herramienta
        
        return "borrador"

    def dedo_levantado(self, yi, y9):
        """Verifica si el dedo índice está levantado"""
        return (y9 - yi) > 40

    def dibujar_linea(self, frm, x, y, xi, yi, y9):
        """Maneja la herramienta de línea"""
        if self.dedo_levantado(yi, y9):
            if not self.var_iniciales:
                self.xii, self.yii = x, y
                self.var_iniciales = True

            cv2.line(frm, (self.xii, self.yii), (x, y), (50, 152, 255), self.grosor)
        else:
            if self.var_iniciales:
                cv2.line(self.mascara, (self.xii, self.yii), (x, y), 0, self.grosor)
                self.var_iniciales = False

    def dibujar_rectangulo(self, frm, x, y, xi, yi, y9):
        """Maneja la herramienta de rectángulo"""
        if self.dedo_levantado(yi, y9):
            if not self.var_iniciales:
                self.xii, self.yii = x, y
                self.var_iniciales = True

            cv2.rectangle(frm, (self.xii, self.yii), (x, y), (0, 255, 255), self.grosor)
        else:
            if self.var_iniciales:
                cv2.rectangle(self.mascara, (self.xii, self.yii), (x, y), 0, self.grosor)
                self.var_iniciales = False

    def dibujar_circulo(self, frm, x, y, xi, yi, y9):
        """Maneja la herramienta de círculo"""
        if self.dedo_levantado(yi, y9):
            if not self.var_iniciales:
                self.xii, self.yii = x, y
                self.var_iniciales = True

            radio = int(((self.xii - x)**2 + (self.yii - y)**2)**0.5)
            cv2.circle(frm, (self.xii, self.yii), radio, (255, 255, 0), self.grosor)
        else:
            if self.var_iniciales:
                radio = int(((self.xii - x)**2 + (self.yii - y)**2)**0.5)
                cv2.circle(self.mascara, (self.xii, self.yii), radio, (0, 255, 0), self.grosor)
                self.var_iniciales = False

    def dibujar_a_mano_alzada(self, x, y, xi, yi, y9):
        """Maneja el dibujo a mano alzada"""
        if self.dedo_levantado(yi, y9):
            cv2.line(self.mascara, (self.prevx, self.prevy), (x, y), 0, self.grosor)
            self.prevx, self.prevy = x, y
        else:
            self.prevx = x
            self.prevy = y

    def borrar(self, frm, x, y):
        """Maneja la herramienta de borrado"""
        cv2.circle(frm, (x, y), 30, (0, 0, 0), -1)
        cv2.circle(self.mascara, (x, y), 30, 255, -1)

    def seleccionar_herramienta(self, x, y, frm):
        """Selecciona la herramienta basada en la posición de la mano"""
        if x < self.max_x and y < self.max_y and x > self.ml:
            if self.tiempo_inicializado:
                self.ctime = time.time()
                self.tiempo_inicializado = False
            
            ptime = time.time()

            cv2.circle(frm, (x, y), self.radio, (0, 255, 255), 2)
            self.radio -= 1

            if (ptime - self.ctime) > 0.8:
                self.herramienta_actual = self.obtener_herramienta(x)
                print("Estás usando: ", self.herramienta_actual)
                self.tiempo_inicializado = True
                self.radio = 40
        else:
            self.tiempo_inicializado = True
            self.radio = 40

    def ejecutar(self):
        """Método principal que ejecuta la aplicación de dibujo"""
        cap = cv2.VideoCapture(1)
        
        while True:
            _, frm = cap.read()
            frm = cv2.flip(frm, 1)

            rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
            op = self.detector_manos.process(rgb)

            if op.multi_hand_landmarks:
                for i in op.multi_hand_landmarks:
                    self.dibujante.draw_landmarks(frm, i, self.manos.HAND_CONNECTIONS)
                    x, y = int(i.landmark[8].x * 640), int(i.landmark[8].y * 480)

                    self.seleccionar_herramienta(x, y, frm)

                    if self.herramienta_actual == "dibujar":
                        xi = int(i.landmark[12].x * 640)
                        yi = int(i.landmark[12].y * 480)
                        y9 = int(i.landmark[9].y * 480)
                        self.dibujar_a_mano_alzada(x, y, xi, yi, y9)

                    elif self.herramienta_actual == "linea":
                        xi = int(i.landmark[12].x * 640)
                        yi = int(i.landmark[12].y * 480)
                        y9 = int(i.landmark[9].y * 480)
                        self.dibujar_linea(frm, x, y, xi, yi, y9)

                    elif self.herramienta_actual == "rectangulo":
                        xi = int(i.landmark[12].x * 640)
                        yi = int(i.landmark[12].y * 480)
                        y9 = int(i.landmark[9].y * 480)
                        self.dibujar_rectangulo(frm, x, y, xi, yi, y9)

                    elif self.herramienta_actual == "circulo":
                        xi = int(i.landmark[12].x * 640)
                        yi = int(i.landmark[12].y * 480)
                        y9 = int(i.landmark[9].y * 480)
                        self.dibujar_circulo(frm, x, y, xi, yi, y9)

                    elif self.herramienta_actual == "borrador":
                        xi = int(i.landmark[12].x * 640)
                        yi = int(i.landmark[12].y * 480)
                        y9 = int(i.landmark[9].y * 480)
                        if self.dedo_levantado(yi, y9):
                            self.borrar(frm, x, y)

            op = cv2.bitwise_and(frm, frm, mask=self.mascara)
            frm[:, :, 1] = op[:, :, 1]
            frm[:, :, 2] = op[:, :, 2]

            frm[:self.max_y, self.ml:self.max_x] = cv2.addWeighted(self.herramientas, 0.7, frm[:self.max_y, self.ml:self.max_x], 0.3, 0)

            cv2.putText(frm, self.herramienta_actual, (270 + self.ml, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Aplicación de Pintura", frm)

            if cv2.waitKey(1) == 27:
                cv2.destroyAllWindows()
                cap.release()
                break

# Iniciar la aplicación
if __name__ == "__main__":
    app = AplicacionDibujo()
    app.ejecutar()