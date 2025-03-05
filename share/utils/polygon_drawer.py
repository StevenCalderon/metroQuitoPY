import tkinter as tk
from tkinter import Canvas

class PolygonDrawerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dibujo de Polígono")

        # Dimensiones de la ventana
        self.root.geometry("800x600")

        # Variables
        self.polygon = []
        self.drawing_enabled = True  # Flag para habilitar/deshabilitar el dibujo

        # Frame para contener el canvas y el botón
        self.frame = tk.Frame(self.root)
        self.frame.pack(fill="both", expand=True)

        # Botón para resetear el polígono
        self.reset_button = tk.Button(self.frame, text="Resetear", command=self.reset_polygon)
        self.reset_button.pack(side="top", pady=10)

        # Canvas para dibujar el polígono
        self.canvas = Canvas(self.frame, bg="white", width=800, height=500)
        self.canvas.pack()

        # Iniciar el dibujo del polígono
        self.canvas.bind("<ButtonPress-1>", self.start_polygon)  # Botón izquierdo para añadir puntos
        self.canvas.bind("<ButtonPress-3>", self.close_polygon)  # Botón derecho para cerrar el polígono

    def start_polygon(self, event):
        """Inicia el dibujo del polígono y marca los puntos."""
        if not self.drawing_enabled:
            return  # No permitir dibujar si no está habilitado

        if len(self.polygon) == 0:  # Si no hay puntos, iniciar un nuevo polígono
            self.polygon = [(event.x, event.y)]
            # Dibujar un círculo en el primer punto
            self.canvas.create_oval(
                event.x - 5, event.y - 5, event.x + 5, event.y + 5,
                outline="green", fill="green", width=2, tags="polygon_point"
            )
        else:
            # Añadir un punto al polígono
            self.polygon.append((event.x, event.y))
            # Dibujar el punto
            self.canvas.create_oval(
                event.x - 5, event.y - 5, event.x + 5, event.y + 5,
                outline="blue", fill="blue", width=2, tags="polygon_point"
            )
        
        # Redibujar el polígono para visualizar las líneas
        self.draw_polygon()

    def draw_polygon(self):
        """Dibuja el polígono completo en el lienzo."""
        self.canvas.delete("polygon_line")  # Limpiar las líneas anteriores

        # Dibujar las líneas entre los puntos
        if len(self.polygon) > 1:
            for i in range(len(self.polygon) - 1):
                x1, y1 = self.polygon[i]
                x2, y2 = self.polygon[i + 1]
                self.canvas.create_line(
                    x1, y1, x2, y2,
                    fill="blue", width=2, tags="polygon_line"
                )

    def close_polygon(self, event):
        """Cierra el polígono al hacer clic derecho, lo pinta por dentro e imprime las coordenadas."""
        if len(self.polygon) > 2:  # Solo cerrar si hay más de dos puntos
            # Añadir el primer punto para cerrar el polígono
            self.polygon.append(self.polygon[0])

            # Pintar el polígono por dentro
            self.canvas.create_polygon(
                self.polygon, fill="lightblue", outline="blue", width=2, tags="polygon_fill"
            )

            # Imprimir las coordenadas del polígono
            print("Coordenadas del polígono:")
            for point in self.polygon:
                print(point)

            # Bloquear el dibujo de nuevos puntos
            self.drawing_enabled = False

            # Mostrar mensaje de que el polígono está cerrado
            print("Polígono cerrado. Haz clic en 'Resetear' para dibujar uno nuevo.")

    def reset_polygon(self):
        """Resetea el dibujo, permitiendo comenzar uno nuevo."""
        # Limpiar el canvas
        self.canvas.delete("all")
        
        # Resetear variables
        self.polygon = []
        self.drawing_enabled = True  # Volver a habilitar el dibujo
        
        print("Nuevo polígono listo para dibujar.")

def run_polygon_drawer():
    """Inicia la aplicación y ejecuta el dibujo del polígono."""
    root = tk.Tk()
    app = PolygonDrawerApp(root)
    root.mainloop()

