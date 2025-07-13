import tkinter as tk
from tkinter import messagebox
import requests

def analizar_sentimiento():
    texto = entrada_texto.get("1.0", tk.END).strip()
    if not texto:
        messagebox.showwarning("Campo vacío", "Por favor ingresa una reseña.")
        return

    try:
        response = requests.post(
            "http://127.0.0.1:5000/predict",
            json={"text": texto}
        )
        response.raise_for_status()
        pred = response.json()['prediction']
        if pred == 1:
            resultado.config(text="✅ Sentimiento Positivo 😊", fg="green")
        else:
            resultado.config(text="❌ Sentimiento Negativo 😞", fg="red")
    except requests.exceptions.RequestException as e:
        messagebox.showerror("Error de conexión", f"No se pudo conectar con el API.\n{e}")

# Crear ventana principal
root = tk.Tk()
root.title("Análisis de Sentimientos - Reseñas de Usuarios")
root.geometry("600x400")
root.configure(bg="#F9F9F9")
root.resizable(False, False)

# Título principal
titulo = tk.Label(root, text="🧠 Análisis de Sentimientos", font=("Helvetica", 18, "bold"), bg="#F9F9F9", fg="#333")
titulo.pack(pady=10)

# Subtítulo
subtitulo = tk.Label(root, text="Escribe una reseña de producto y detecta si es positiva o negativa.", font=("Helvetica", 12), bg="#F9F9F9", fg="#666")
subtitulo.pack()

# Cuadro de texto
entrada_texto = tk.Text(root, height=6, width=65, font=("Helvetica", 11))
entrada_texto.pack(pady=15)

# Botón de análisis
boton = tk.Button(root, text="Analizar Sentimiento", command=analizar_sentimiento,
                  bg="#4CAF50", fg="white", font=("Helvetica", 12, "bold"), width=20)
boton.pack(pady=5)

# Resultado
resultado = tk.Label(root, text="", font=("Helvetica", 16), bg="#F9F9F9")
resultado.pack(pady=20)

# Ejecutar loop
root.mainloop()
