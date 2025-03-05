# -*- coding: utf-8 -*-
import time
import tkinter as tk
from tkinter import filedialog, Canvas, Label, Frame, Button
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import os
from threading import Thread

import numpy as np
from main_process import process_video, draw_polygon


class VideoProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Metro de Quito + IA")

        # Obtener dimensiones de la pantalla
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # Ajustar la ventana al tamaño de la pantalla
        self.root.geometry(f"{screen_width}x{screen_height}")
        self.root.config(bg="#f4f4f4")  # Fondo gris claro
        self.root.resizable(True, False)  # Desactivar redimensionamiento en alto

        # Variables
        self.video_path = None
        self.first_frame = None
        self.train_zone  = []
        self.current_rectangle = None
        self.cap = None
        self.output_path = None
        self.frame_width = 0
        self.frame_height = 0
        self.safe_zone = []
        self.drawing_enabled = False 

        # Estilos para botones
        self.button_enabled_style = {
            "font": ("Helvetica", 18),
            "bg": "#223e77",
            "fg": "#ffffff",
            "relief": "flat",
            "width": 28,
            "pady": 10,
            "state": tk.ACTIVE,
        }

        self.button_disabled_style = {
            "font": ("Helvetica", 18),
            "bg": "#cccccc",
            "fg": "#666666",
            "relief": "flat",
            "width": 28,
            "pady": 10,
            "state": tk.DISABLED,
        }

        # Contenedor principal
        self.main_frame = Frame(self.root, bg="#f4f4f4")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Usar grid para dividir la ventana en proporciones
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=3)  # Menú (30%)
        self.main_frame.grid_columnconfigure(1, weight=7)  # Video (70%)

        # Columna izquierda: Menú
        self.menu_frame = Frame(self.main_frame, bg="#ffffff", relief="raised", bd=2)
        self.menu_frame.grid(row=0, column=0, sticky="nswe", padx=10, pady=10)

        # Columna derecha: Video
        self.video_frame = Frame(self.main_frame, bg="#000000")
        self.video_frame.grid(row=0, column=1, sticky="nswe", padx=10, pady=10)

        # Menú (columna izquierda)
        self.menu_label = Label(
            self.menu_frame,
            text="Metro de Quito + IA",
            font=("Helvetica", 26, "bold"),
            fg="#223e77",
            bg="#ffffff",
            pady=20,
            padx=20,
        )
        self.menu_label.pack()

        self.info_label = Label(
            self.menu_frame,
            text="Cargue un video del Metro de Quito para empezar.",
            font=("Helvetica", 18),
            fg="#444444",
            bg="#ffffff",
            wraplength=380,
            justify="center",
            pady=20,
        )
        
        self.info_label.pack(pady=40)
        self.progress_bar_frame = Frame(self.menu_frame, bg="#ffffff")  # Contenedor para la barra de progreso
        self.progress_bar_frame.pack(pady=10)
        self.progress_bar = ttk.Progressbar(self.menu_frame, orient="horizontal", length=200, mode="determinate")

        # Botones en el orden correcto
        self.load_button = Button(
            self.menu_frame,
            text="1.- Cargar Video",
            command=self.load_video,
            **self.button_enabled_style
        )
        self.load_button.pack(pady=20)

        self.save_button = Button(
            self.menu_frame,
            text="2.-Guardar ROIs",
            command=self.save_polygon,
            **self.button_disabled_style
        )
        self.save_button.pack(pady=20)
        
        self.reset_polygon_button = Button(
            self.menu_frame,
            text="Reiniciar Polígono",
            command=self.reset_polygon,
            font=("Helvetica", 16),
            bg="#f4a261",
            fg="#ffffff",
            relief="flat",
            width=28,
            pady=10,
        )
        self.reset_polygon_button.pack(pady=20)

        self.choose_output_button = Button(
            self.menu_frame,
            text="3.- Seleccionar Carpeta de Salida",
            command=self.choose_output_folder,
            **self.button_disabled_style
        )
        self.choose_output_button.pack(pady=20)

        self.process_button = Button(
            self.menu_frame,
            text="4.- Procesar Video",
            command=self.process_video,
            **self.button_disabled_style
        )
        self.process_button.pack(pady=20)
        
        # Botón de reinicio
        self.reset_button = Button(
            self.menu_frame,
            text="Reiniciar",
            command=self.reset_app,
            font=("Helvetica", 16),
            bg="#f4a261",
            fg="#ffffff",
            relief="flat",
            width=28,
            pady=10,
        )
        self.reset_button.pack(pady=20)

        self.exit_button = Button(
            self.menu_frame,
            text="Salir",
            command=self.exit_app,
            font=("Helvetica", 16),
            bg="#870000",
            fg="#ffffff",
            relief="flat",
            width=28,
            pady=10,
        )
        self.exit_button.pack(pady=40)

        # Canvas para el video
        self.canvas = Canvas(self.video_frame, bg="#000000", highlightthickness=0)
        self.canvas.pack()

    def set_button_state(self, button, enabled):
        """Aplica estilos habilitados o deshabilitados a un botón."""
        style = self.button_enabled_style if enabled else self.button_disabled_style
        button.config(**style)
        
    def draw_safe_zone(self, event):
        """Captura el color en el punto donde el usuario hace clic."""
        # Aquí ya no reiniciamos self.train_zone    , ya que el polígono se debe dibujar después
        self.info_label.config(text="Dibuje un polígono sobre el vehículo del metro y luego presione 'Guardar'.")
        self.drawing_enabled = True
        # Habilitar eventos para dibujar el polígono
        self.canvas.bind("<ButtonPress-1>", lambda event1: self.start_polygon(self.train_zone, event1))
        self.canvas.bind("<ButtonPress-3>", lambda event1: self.close_polygon(self.train_zone))

            # Activar el botón de guardar si el polígono está completo
        if len(self.train_zone) > 2:  # El polígono debe tener al menos 3 puntos
            self.set_button_state(self.save_button, True)

    def get_color_at_point(self, x, y):
        """Obtiene el color de un pixel en las coordenadas (x, y) en el frame."""
        color_bgr = self.first_frame[y, x]  # Imagen RGB en lugar de BGR
        return tuple(color_bgr)  # Devuelve el color como (R, G, B)

    def load_video(self):
        """Carga un video y muestra el primer frame."""
        self.video_path = filedialog.askopenfilename(filetypes=[("Archivos de video", "*.mp4;*.avi;*.mov")])
        if not self.video_path:
            return

        # Abrir el video y obtener el primer frame
        self.cap = cv2.VideoCapture(self.video_path)
        ret, frame = self.cap.read()

        if not ret:
            self.info_label.config(text="Error al cargar el video. Por favor, intente de nuevo.")
            return

        # Guardar las dimensiones del frame original
        self.frame_height, self.frame_width = frame.shape[:2]

        # Configurar el canvas para que coincida con las dimensiones del video
        self.canvas.config(width=self.frame_width, height=self.frame_height)

        # Convertir el frame a RGB y mostrarlo en el canvas
        self.first_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.display_frame(self.first_frame)

        self.info_label.config(text="Dibuje un poligono que represente la zona segura (franja amarilla) y luego presione 'Guardar'")
        self.set_button_state(self.save_button, True)
        self.drawing_enabled = True
        self.canvas.bind("<ButtonPress-1>", lambda event: self.start_polygon(self.safe_zone, event))
        self.canvas.bind("<ButtonPress-3>", lambda event: self.close_polygon(self.safe_zone))
        
        # Permitir al usuario dibujar en el canvas
        if len(self.safe_zone) > 2:
            self.canvas.bind("<ButtonPress-1>", self.draw_safe_zone)


    def display_frame(self, frame):
        """Muestra el frame sin redimensionar."""
        img = Image.fromarray(frame)
        img_tk = ImageTk.PhotoImage(image=img)
        self.canvas.img_tk = img_tk  # Mantener referencia para evitar recolección de basura
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)

    def choose_output_folder(self):
        """Permite al usuario seleccionar una carpeta para guardar el video procesado."""
        self.output_path = filedialog.askdirectory()
        if self.output_path:
            self.info_label.config(text="Carpeta seleccionada correctamente.")
            self.set_button_state(self.process_button, True)

    def start_polygon(self, zone, event):
        """Inicia el dibujo del polígono y marca los puntos."""
        if not self.drawing_enabled:
            return  # No permitir dibujar si no está habilitado

        if len(zone) == 0:  # Si no hay puntos, iniciar un nuevo polígono
            zone = [(event.x, event.y)]
            # Dibujar un círculo en el primer punto
            self.canvas.create_oval(
                event.x - 5, event.y - 5, event.x + 5, event.y + 5,
                outline="green", fill="green", width=2, tags="polygon_point"
            )
        else:
            # Añadir un punto al polígono
            zone.append((event.x, event.y))
            self.canvas.create_oval(
                event.x - 5, event.y - 5, event.x + 5, event.y + 5,
                outline="blue", fill="blue", width=2, tags="polygon_point"
            )
        
        # Redibujar el polígono para visualizar las líneas
        draw_polygon(self)


    def close_polygon(self,zone):
        """Cierra el polígono al hacer clic derecho, lo pinta por dentro e imprime las coordenadas."""
        if len(zone) > 2:  # Solo cerrar si hay más de dos puntos
            # Añadir el primer punto para cerrar el polígono
            zone.append(zone[0])

            # Pintar el polígono por dentro
            self.canvas.create_polygon(
                zone , fill="lightblue", outline="blue", width=2, tags="polygon_fill"
            )

            # Imprimir las coordenadas del polígono
            print("Coordenadas del polígono:")
            for point in zone    :
                print(point)

            # Bloquear el dibujo de nuevos puntos
            self.drawing_enabled = False

            # Mostrar mensaje de que el polígono está cerrado
            print("Polígono cerrado. Haz clic en 'Resetear' para dibujar uno nuevo.")
                
    def reset_polygon(self):
        """Reinicia el dibujo del polígono actual."""
        self.train_zone  = []
        self.canvas.delete("polygon_point")
        self.canvas.delete("polygon_line")
        self.info_label.config(text="Dibuje un nuevo polígono haciendo clic en el canvas.")
        self.set_button_state(self.save_button, False)

    def save_polygon(self):
        """Guarda el polígono dibujado."""
        if self.train_zone  :
            print("Polígono guardado:", self.train_zone )
            self.info_label.config(text="Polígono guardado correctamente.")
            self.info_label.after(2500, self.show_output_message)
    
    def show_output_message(self):
        """Muestra el mensaje para seleccionar la carpeta de salida."""
        self.info_label.config(text="Seleccione la carpeta de salida.")
        self.set_button_state(self.choose_output_button, True)

    def process_video(self):
        """Procesa el video y guarda el archivo procesado."""
        if self.video_path and len(self.train_zone  ) > 0 and self.output_path and self.safe_zone:
            self.set_button_state(self.process_button, False)
            self.info_label.config(text="Procesando video...")

            # Muestra la barra de progreso
            self.progress_bar["value"] = 0
            self.progress_bar["maximum"] = 100  # Asume que el progreso va de 0 a 100
            self.progress_bar.pack(in_=self.progress_bar_frame, pady=4)

            def processing():
                # Llamada al procesamiento con la barra de progreso
                process_video(self.video_path, self.output_path, self.safe_zone, self.train_zone  , self.display_frame, self.progress_bar)

                self.info_label.config(text="Video procesado exitosamente.")
                self.set_button_state(self.process_button, True)
                self.progress_bar["value"] = 0  # Resetear la barra
                self.progress_bar.pack_forget()  # Ocultar la barra cuando termine

            # Usar un solo hilo para evitar crear múltiples hilos
            processing_thread = Thread(target=processing)
            processing_thread.start()
        else:
            self.info_label.config(text="Complete los pasos anteriores antes de procesar el video.")

    def reset_app(self):
        """Reinicia la aplicación a su estado inicial."""
        # Limpiar variables
        self.video_path = None
        self.first_frame = None
        self.train_zone  = []
        self.current_rectangle = None
        self.cap = None
        self.output_path = None
        self.frame_width = 0
        self.frame_height = 0

        # Resetear interfaz
        self.info_label.config(text="Cargue un video del Metro de Quito para empezar.")
        self.canvas.delete("all")  # Limpiar canvas

        # Restablecer botones
        self.set_button_state(self.load_button, True)
        self.set_button_state(self.save_button, False)
        self.set_button_state(self.choose_output_button, False)
        self.set_button_state(self.process_button, False)

        # Ocultar barra de progreso
        self.progress_bar.pack_forget()

    def exit_app(self):
        """Cierra la aplicación."""
        self.root.destroy()


# Ejecutar la aplicación
root = tk.Tk()
app = VideoProcessorApp(root)
root.mainloop()
