import os
import sys
import subprocess
import threading
import tkinter as tk
from tkinter import ttk

try:
    from PIL import Image, ImageTk
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
    from PIL import Image, ImageTk

should_relaunch = True

def center(win):
    win.update_idletasks()
    width = win.winfo_width()
    height = win.winfo_height()
    x = (win.winfo_screenwidth() // 2) - (width // 2)
    y = (win.winfo_screenheight() // 2) - (height // 2)
    win.geometry(f"{width}x{height}+{x}+{y}")

def show_success_window():
    success = tk.Tk()
    success.title("Setup Complete")

    label = ttk.Label(success, text="All dependencies installed!\nYou can now use DeepDisco.")
    label.pack(pady=(15, 5), padx=10)

    try:
        image_path = os.path.join("resources", "deepdisco.png")
        image = Image.open(image_path)
        resized = image.resize((image.width // 2, image.height // 2))
        img_tk = ImageTk.PhotoImage(resized)
        img_label = tk.Label(success, image=img_tk)
        img_label.image = img_tk
        img_label.pack(pady=(0, 10))
    except Exception as e:
        print("Image failed to load:", e)

    def close_and_continue():
        success.destroy()
        os.execv(os.path.join("venv", "Scripts", "python.exe"), ["python.exe", "part.py"])

    ok_button = ttk.Button(success, text="LAUNCH", command=close_and_continue)
    ok_button.pack(pady=(0, 15))

    success.geometry("")
    center(success)
    success.mainloop()

def install():
    global should_relaunch
    try:
        subprocess.check_call([sys.executable, "-m", "venv", "venv"])
        subprocess.check_call(["venv\\Scripts\\pip.exe", "install", "--upgrade", "pip"])
        subprocess.check_call(["venv\\Scripts\\pip.exe", "install", "-r", "requirements.txt"])
    except Exception as e:
        print("Setup error:", e)
        should_relaunch = False
    finally:
        root.quit()

if not os.path.exists("venv\\Scripts\\python.exe"):
    root = tk.Tk()
    root.title("Setting Up Environment")

    label = ttk.Label(root, text="Installing dependencies...\nThis will only happen once.")
    label.pack(pady=(15, 5), padx=10)

    try:
        image_path = os.path.join("resources", "loading.png")
        image = Image.open(image_path)
        img_tk = ImageTk.PhotoImage(image)
        img_label = tk.Label(root, image=img_tk)
        img_label.image = img_tk
        img_label.pack(pady=(0, 10))
    except Exception as e:
        print("Loading image failed:", e)

    progress = ttk.Progressbar(root, mode='indeterminate')
    progress.pack(pady=(0, 15), padx=20, fill='x')
    progress.start()

    root.geometry("")
    center(root)

    threading.Thread(target=install).start()
    root.mainloop()
    root.destroy()

    if should_relaunch:
        show_success_window()
else:
    subprocess.call(["venv\\Scripts\\python.exe", "part.py"])
