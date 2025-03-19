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

        self._configure_window()
        self._initialize_variables()
        self._initialize_styles()
        self._create_main_layout()
        self._create_menu()
        self._create_video_canvas()

    def _configure_window(self):
        screen_width, screen_height = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry(f"{screen_width}x{screen_height}")
        self.root.config(bg="#f4f4f4")
        self.root.resizable(True, False)

    def _initialize_variables(self):
        self.video_path = None
        self.first_frame = None
        self.current_rectangle = None
        self.cap = None
        self.output_path = None
        self.frame_width, self.frame_height = 0, 0
        self.safe_zone, self.train_zone = [], []
        self.safe_zone_saved, self.train_zone_saved, self.drawing_enabled = False, False, False
        self.current_step = 0  # Nueva variable para rastrear el paso actual
        self.instructions = [
            "1. Cargue un video del Metro de Quito para empezar.",
            "2. Dibuje un polígono sobre la franja amarilla de seguridad.",
            "3. Guarde la zona segura y dibuje un polígono sobre el vehículo del metro.",
            "4. Guarde la zona del tren y seleccione la carpeta de salida.",
            "5. Procese el video para detectar personas cruzando la línea de seguridad."
        ]

    def _initialize_styles(self):
        self.button_enabled_style = {"font": ("Helvetica", 18), "bg": "#223e77", "fg": "#ffffff", "relief": "flat",
                                     "width": 28, "pady": 10, "state": tk.ACTIVE}
        self.button_disabled_style = {"font": ("Helvetica", 18), "bg": "#cccccc", "fg": "#666666", "relief": "flat",
                                      "width": 28, "pady": 10, "state": tk.DISABLED}

    def _create_main_layout(self):
        self.main_frame = Frame(self.root, bg="#f4f4f4")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=3)
        self.main_frame.grid_columnconfigure(1, weight=7)

        self.menu_frame = Frame(self.main_frame, bg="#ffffff", relief="raised", bd=2)
        self.menu_frame.grid(row=0, column=0, sticky="nswe", padx=10, pady=10)

        self.video_frame = Frame(self.main_frame, bg="#000000")
        self.video_frame.grid(row=0, column=1, sticky="nswe", padx=10, pady=10)

    def _create_menu(self):
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
        
        self.instructions_frame = Frame(self.menu_frame, bg="#ffffff")
        self.instructions_frame.pack(pady=10, padx=20, fill="x")
        
        self.instructions_label = Label(
            self.instructions_frame,
            text="Instrucciones:",
            font=("Helvetica", 16, "bold"),
            fg="#223e77",
            bg="#ffffff",
            pady=5,
        )
        self.instructions_label.pack()
        
        self.info_label = Label(
            self.instructions_frame,
            text=self.instructions[0],
            font=("Helvetica", 14),
            fg="#444444",
            bg="#ffffff",
            wraplength=380,
            justify="left",
            pady=10,
        )
        self.info_label.pack()
        
        self.progress_bar_frame = Frame(self.menu_frame, bg="#ffffff")
        self.progress_bar_frame.pack(pady=10)
        
        self.progress_label = Label(
            self.progress_bar_frame,
            text="Progreso:",
            font=("Helvetica", 12),
            fg="#444444",
            bg="#ffffff",
        )
        self.progress_label.pack()
        
        self.progress_bar = ttk.Progressbar(
            self.menu_frame,
            orient="horizontal",
            length=200,
            mode="determinate"
        )

        self.buttons_frame = Frame(self.menu_frame, bg="#ffffff")
        self.buttons_frame.pack(pady=20)

        self.load_button = Button(
            self.buttons_frame,
            text="1. Cargar Video",
            command=self.load_video,
            **self.button_enabled_style
        )
        self.load_button.pack(pady=10)
        self.create_tooltip(self.load_button, "Seleccione un video del Metro de Quito para procesar")

        self.save_button = Button(
            self.buttons_frame,
            text="2. Guardar ROIs",
            command=self.save_polygon,
            **self.button_disabled_style
        )
        self.save_button.pack(pady=10)
        self.create_tooltip(self.save_button, "Guarde las zonas seleccionadas para continuar")

        self.choose_output_button = Button(
            self.buttons_frame,
            text="3. Seleccionar Carpeta de Salida",
            command=self.choose_output_folder,
            **self.button_disabled_style
        )
        self.choose_output_button.pack(pady=10)
        self.create_tooltip(self.choose_output_button, "Seleccione dónde guardar el video procesado")

        self.process_button = Button(
            self.buttons_frame,
            text="4. Procesar Video",
            command=self.process_video,
            **self.button_disabled_style
        )
        self.process_button.pack(pady=10)
        self.create_tooltip(self.process_button, "Inicie el procesamiento del video")

        self.control_frame = Frame(self.menu_frame, bg="#ffffff")
        self.control_frame.pack(pady=20)

        self.reset_polygon_button = Button(
            self.control_frame,
            text="Reiniciar Polígono",
            command=self.reset_polygon,
            font=("Helvetica", 14),
            bg="#f4a261",
            fg="#ffffff",
            relief="flat",
            width=20,
            pady=8,
        )
        self.reset_polygon_button.pack(pady=5)
        self.create_tooltip(self.reset_polygon_button, "Reinicia el polígono actual para dibujarlo de nuevo")

        self.reset_button = Button(
            self.control_frame,
            text="Reiniciar Todo",
            command=self.reset_app,
            font=("Helvetica", 14),
            bg="#f4a261",
            fg="#ffffff",
            relief="flat",
            width=20,
            pady=8,
        )
        self.reset_button.pack(pady=5)
        self.create_tooltip(self.reset_button, "Reinicia toda la aplicación a su estado inicial")

        self.exit_button = Button(
            self.control_frame,
            text="Salir",
            command=self.exit_app,
            font=("Helvetica", 14),
            bg="#870000",
            fg="#ffffff",
            relief="flat",
            width=20,
            pady=8,
        )
        self.exit_button.pack(pady=5)
        self.create_tooltip(self.exit_button, "Cierra la aplicación")

    def _create_video_canvas(self):
        """Crea el canvas para mostrar el video"""
        self.canvas = Canvas(self.video_frame, bg="#000000", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def set_button_state(self, button, enabled):
        """Aplica estilos habilitados o deshabilitados a un botón."""
        style = self.button_enabled_style if enabled else self.button_disabled_style
        button.config(**style)
        
    def draw_train_zone(self):
        """Captura el color en el punto donde el usuario hace clic."""
        self.update_instructions(2)
        self.info_label.config(text="Dibuje un polígono sobre el vehículo del metro:\n1. Haga clic izquierdo para añadir puntos\n2. Haga clic derecho para cerrar el polígono")
        self.drawing_enabled = True
        self.canvas.bind("<ButtonPress-1>", lambda event1: self.start_polygon(self.train_zone, event1))
        self.canvas.bind("<ButtonPress-3>", lambda event1: self.close_polygon(self.train_zone))

        if len(self.train_zone) > 2:
            self.set_button_state(self.save_button, True)


    def get_color_at_point(self, x, y):
        """Obtiene el color de un pixel en las coordenadas (x, y) en el frame."""
        color_bgr = self.first_frame[y, x]
        return tuple(color_bgr)

    def load_video(self):
        """Carga un video y muestra el primer frame."""
        self.video_path = filedialog.askopenfilename(filetypes=[("Archivos de video", "*.mp4;*.avi;*.mov")])
        if not self.video_path:
            return

        self.cap = cv2.VideoCapture(self.video_path)
        ret, frame = self.cap.read()

        if not ret:
            self.info_label.config(text="Error al cargar el video. Por favor, intente de nuevo.")
            return

        self.frame_height, self.frame_width = frame.shape[:2]

        self.canvas.config(width=self.frame_width, height=self.frame_height)

        self.first_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.display_frame(self.first_frame)

        self.update_instructions(1)
        self.info_label.config(text="Dibuje un polígono sobre la franja amarilla de seguridad:\n1. Haga clic izquierdo para añadir puntos\n2. Haga clic derecho para cerrar el polígono")
        self.set_button_state(self.save_button, True)
        self.drawing_enabled = True
        self.canvas.bind("<ButtonPress-1>", lambda event: self.start_polygon(self.safe_zone, event))
        self.canvas.bind("<ButtonPress-3>", lambda event: self.close_polygon(self.safe_zone))



    def display_frame(self, frame):
        img = Image.fromarray(frame)
        img_tk = ImageTk.PhotoImage(image=img)
        self.canvas.img_tk = img_tk
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)

    def choose_output_folder(self):
        """Permite al usuario seleccionar una carpeta para guardar el video procesado."""
        self.output_path = filedialog.askdirectory()
        if self.output_path:
            self.info_label.config(text="Carpeta seleccionada correctamente.")
            self.set_button_state(self.process_button, True)

    def start_polygon(self, zone, event):
        if not self.drawing_enabled:
            return

        if len(zone) == 0:
            zone.append((event.x, event.y))
            self.canvas.create_oval(
                event.x - 5, event.y - 5, event.x + 5, event.y + 5,
                outline="green", fill="green", width=2, tags="polygon_point"
            )
        else:
            zone.append((event.x, event.y))
            self.canvas.create_oval(
                event.x - 5, event.y - 5, event.x + 5, event.y + 5,
                outline="blue", fill="blue", width=2, tags="polygon_point"
            )

        self.draw_polygon(zone)

    def draw_polygon(self, zone):
        if len(zone) > 1:
            x1, y1 = zone[-2]
            x2, y2 = zone[-1]
            self.canvas.create_line(x1, y1, x2, y2, fill="#ec253a", width=2)

    def close_polygon(self,zone):
        print("ZONE CLOSE "+ str(zone))
        if len(zone) > 2:
            zone.append(zone[0])

            self.canvas.create_polygon(
                zone , fill="lightblue", outline="blue", width=2, tags="polygon_fill"
            )

            print("Coordenadas del polígono:")
            for point in zone    :
                print(point)

            self.drawing_enabled = False

            print("Polígono cerrado. Haz clic en 'Resetear' para dibujar uno nuevo.")
                
    def reset_polygon(self):
        """Reinicia el dibujo del polígono actual."""
        self.train_zone  = []
        self.canvas.delete("polygon_point")
        self.canvas.delete("polygon_line")
        self.info_label.config(text="Dibuje un nuevo polígono haciendo clic en el canvas.")
        self.set_button_state(self.save_button, False)

    def save_polygon(self):
        """Guarda el polígono si tiene al menos tres puntos válidos."""

        if len(self.safe_zone) >= 2 and self.safe_zone_saved == False:
            self.safe_zone_saved = True
            self.info_label.config(text="Zona segura guardada correctamente.")
            self.set_button_state(self.save_button, True)
            self.draw_train_zone()

        elif len(self.train_zone) >= 2 and self.train_zone_saved == False:
            self.train_zone_saved = True
            self.info_label.config(text="Zona de tren guardada correctamente.")
            self.update_instructions(3)
            self.info_label.after(2500, self.show_output_message)

        else:
            self.info_label.config(text="El polígono debe tener al menos 2 puntos.")
            print("Error: El polígono debe tener al menos tres puntos para guardarse.")

    def show_output_message(self):
        self.info_label.config(text="Seleccione la carpeta donde desea guardar el video procesado.")
        self.set_button_state(self.choose_output_button, True)

    def process_video(self):
        if self.video_path and len(self.train_zone  ) > 0 and self.output_path and self.safe_zone:
            self.set_button_state(self.process_button, False)
            self.info_label.config(text="Procesando video...")

            self.progress_bar["value"] = 0
            self.progress_bar["maximum"] = 100
            self.progress_bar.pack(in_=self.progress_bar_frame, pady=4)

            def processing():
                process_video(self.video_path, self.output_path, self.safe_zone, self.train_zone  , self.display_frame, self.progress_bar)

                self.info_label.config(text="Video procesado exitosamente.")
                self.set_button_state(self.process_button, True)
                self.progress_bar["value"] = 0
                self.progress_bar.pack_forget()

            processing_thread = Thread(target=processing)
            processing_thread.start()
        else:
            self.info_label.config(text="Complete los pasos anteriores antes de procesar el video.")

    def reset_app(self):
        self.video_path = None
        self.first_frame = None
        self.train_zone  = []
        self.current_rectangle = None
        self.cap = None
        self.output_path = None
        self.frame_width = 0
        self.frame_height = 0
        self.safe_zone_saved = False
        self.train_zone_saved = False
        self.drawing_enabled = False
        self.current_step = 0
    
        self.update_instructions(0)
        self.canvas.delete("all")

        self.set_button_state(self.load_button, True)
        self.set_button_state(self.save_button, False)
        self.set_button_state(self.choose_output_button, False)
        self.set_button_state(self.process_button, False)

        self.progress_bar.pack_forget()

    def exit_app(self):
        self.root.destroy()

    def create_tooltip(self, widget, text):
        def show_tooltip(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            
            label = Label(
                tooltip,
                text=text,
                justify="left",
                background="#ffffe0",
                relief="solid",
                borderwidth=1,
                font=("Helvetica", 10),
                padx=5,
                pady=5
            )
            label.pack()
            
            def hide_tooltip():
                tooltip.destroy()
            
            widget.tooltip = tooltip
            widget.bind('<Leave>', lambda e: hide_tooltip())
        
        widget.bind('<Enter>', show_tooltip)

    def update_instructions(self, step):
        self.current_step = step
        self.info_label.config(text=self.instructions[step])


root = tk.Tk()
app = VideoProcessorApp(root)
root.mainloop()
