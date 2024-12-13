import tkinter as tk
from tkinter import filedialog
import cv2
import os
from main_process import process_video  # Importa la función de procesamiento

class VideoProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Procesador de Videos")
        self.root.geometry("400x200")

        # Etiqueta de información
        self.label = tk.Label(root, text="Cargar y procesar un video", font=("Helvetica", 14))
        self.label.pack(pady=10)

        # Botón para cargar video
        self.load_button = tk.Button(root, text="Cargar Video", width=20, command=self.load_video)
        self.load_button.pack(pady=10)

        # Botón para procesar el video
        self.process_button = tk.Button(root, text="Procesar Video", width=20, state=tk.DISABLED, command=self.process_video)
        self.process_button.pack(pady=10)

        self.video_path = None

    def load_video(self):
        # Abrir cuadro de diálogo para seleccionar un video
        self.video_path = filedialog.askopenfilename(filetypes=[("Archivos de video", "*.mp4;*.avi;*.mov")])

        if self.video_path:
            self.label.config(text=f"Video cargado: {os.path.basename(self.video_path)}")
            self.process_button.config(state=tk.NORMAL)

    def process_video(self):
        if self.video_path:
            output_path = os.path.splitext(self.video_path)[0] + "_processed.mp4"
            # Llamar a la función de procesamiento
            process_video(self.video_path, output_path)
            self.label.config(text=f"Video procesado y guardado en: {output_path}")
        else:
            self.label.config(text="Por favor, carga un video primero.")

# Crear la ventana principal
root = tk.Tk()
app = VideoProcessorApp(root)
root.mainloop()
