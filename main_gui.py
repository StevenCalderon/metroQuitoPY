# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import filedialog, Canvas, Label, Frame, Button
from PIL import Image, ImageTk
import cv2
import os
from threading import Thread
from main_process import process_video


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
        self.rectangles = []
        self.current_rectangle = None
        self.cap = None
        self.output_path = None
        self.frame_width = 0
        self.frame_height = 0

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
            text="Flujo de Trabajo",
            font=("Helvetica", 26, "bold"),
            fg="#223e77",
            bg="#ffffff",
            pady=20,
            padx=20,
        )
        self.menu_label.pack()

        # Botones en el orden correcto
        self.load_button = Button(
            self.menu_frame,
            text="Cargar Video",
            command=self.load_video,
            font=("Helvetica", 18),
            bg="#223e77",
            fg="#ffffff",
            relief="flat",
            width=28,
            pady=10,
        )
        self.load_button.pack(pady=20)

        self.save_button = Button(
            self.menu_frame,
            text="Guardar ROIs",
            command=self.save_rois,
            font=("Helvetica", 18),
            bg="#4CAF50",
            fg="#ffffff",
            relief="flat",
            width=28,
            pady=10,
            state=tk.DISABLED,
        )
        self.save_button.pack(pady=20)

        self.choose_output_button = Button(
            self.menu_frame,
            text="Seleccionar Carpeta de Salida",
            command=self.choose_output_folder,
            font=("Helvetica", 18),
            bg="#f4f4f4",
            fg="#223e77",
            relief="flat",
            width=28,
            pady=10,
            state=tk.DISABLED,
        )
        self.choose_output_button.pack(pady=20)

        self.process_button = Button(
            self.menu_frame,
            text="Procesar Video",
            command=self.process_video,
            font=("Helvetica", 18),
            bg="#ec253a",
            fg="#ffffff",
            relief="flat",
            width=28,
            pady=10,
            state=tk.DISABLED,
        )
        self.process_button.pack(pady=20)

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

        self.info_label.config(text="Dibuja 6 rectángulos sobre el área del metro.")
        self.save_button.config(state=tk.NORMAL)

        # Permitir al usuario dibujar en el canvas
        self.canvas.bind("<ButtonPress-1>", self.start_rectangle)
        self.canvas.bind("<B1-Motion>", self.draw_rectangle)
        self.canvas.bind("<ButtonRelease-1>", self.finish_rectangle)

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
            self.process_button.config(state=tk.NORMAL)

    def start_rectangle(self, event):
        """Inicia el dibujo de un rectángulo."""
        if len(self.rectangles) < 6:
            self.current_rectangle = (event.x, event.y)

    def draw_rectangle(self, event):
        """Dibuja un rectángulo mientras se arrastra el mouse."""
        if self.current_rectangle:
            self.canvas.delete("preview")
            self.canvas.create_rectangle(
                self.current_rectangle[0], self.current_rectangle[1],
                event.x, event.y, outline="#ec253a", tag="preview"
            )

    def finish_rectangle(self, event):
        """Finaliza el dibujo de un rectángulo."""
        if self.current_rectangle:
            x1, y1 = self.current_rectangle
            x2, y2 = event.x, event.y
            self.rectangles.append((x1, y1, x2, y2))
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="#4CAF50", width=3, tag="roi")
            self.current_rectangle = None

            if len(self.rectangles) == 6:
                self.info_label.config(text="Guarde los ROIs y seleccione la carpeta de salida.")
                self.choose_output_button.config(state=tk.NORMAL)

    def save_rois(self):
        """Guarda las coordenadas de los rectángulos."""
        if len(self.rectangles) == 6:
            print("Rectángulos guardados:", self.rectangles)
            self.info_label.config(text="Rectángulos guardados correctamente.")

    def process_video(self):
        """Procesa el video y guarda el archivo procesado."""
        if self.video_path and len(self.rectangles) == 6 and self.output_path:
            self.process_button.config(state=tk.DISABLED)
            self.info_label.config(text="Procesando video...")

            def processing():
                process_video(self.video_path, self.output_path, self.rectangles)
                self.info_label.config(text="Video procesado exitosamente.")
                self.process_button.config(state=tk.NORMAL)

            Thread(target=processing).start()
        else:
            self.info_label.config(text="Complete los pasos anteriores antes de procesar el video.")

    def exit_app(self):
        """Cierra la aplicación."""
        self.root.destroy()


# Ejecutar la aplicación
root = tk.Tk()
app = VideoProcessorApp(root)
root.mainloop()
