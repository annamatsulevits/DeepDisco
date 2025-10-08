#!/bin/bash

cd "$(dirname "$0")"

if [ ! -x "./venv/bin/python" ]; then
    echo "🛠 venv not found, launching setup popup..."

    python3 - <<EOF
import os
import threading
import tkinter as tk
from tkinter import ttk
import subprocess
import sys

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
        original_width, original_height = image.size
        resized_image = image.resize((original_width // 2, original_height // 2), Image.ANTIALIAS)
        img_tk = ImageTk.PhotoImage(resized_image)
        img_label = tk.Label(success, image=img_tk)
        img_label.image = img_tk
        img_label.pack(pady=(0, 10))
    except Exception as e:
        print("DeepDisco image failed to load:", e)


    ok_button = ttk.Button(success, text="LAUNCH", command=success.destroy)
    ok_button.pack(pady=(0, 15))

    success.geometry("")  # Resize to fit image and content
    center(success)
    success.mainloop()

def install():
    global should_relaunch
    try:
        subprocess.check_call([sys.executable, "-m", "venv", "venv"])
        subprocess.check_call(["./venv/bin/pip", "install", "--upgrade", "pip"])
        subprocess.check_call(["./venv/bin/pip", "install", "-r", "requirements.txt"])
    except Exception as e:
        print("❌ Setup error:", e)
        should_relaunch = False
    finally:
        root.quit()

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

root.geometry("")  # Fit to image
center(root)

threading.Thread(target=install).start()
root.mainloop()
root.destroy()

if should_relaunch:
    show_success_window()
    sys.exit(100)
else:
    sys.exit(0)
EOF

    if [ "$?" = "100" ]; then
        bash run_app.sh
    fi

    exit 0
fi

./venv/bin/python part.py
