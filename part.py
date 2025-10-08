import sys
import os
import torch
import nibabel as nib
from models.my_torch_model_fm import UNet3D
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QLabel, QPushButton, QVBoxLayout, QWidget,
    QProgressBar, QHBoxLayout
)
from PySide6.QtGui import QPixmap, QIcon
from monai import transforms
import torch.nn.functional as F
import torchio as tio

# Constants for resources
os.environ["QT_MAC_WANTS_LAYER"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCE_DIR = os.path.join(BASE_DIR, "resources")

AFFINE_FILE = os.path.join(RESOURCE_DIR, "affine_image.nii.gz")
SUCCESS_IMAGE = os.path.join(RESOURCE_DIR, "success_image.png")
ICON_PATH = os.path.join(RESOURCE_DIR, "icon_creation.png")

AVERAGE_TEMPLATE = os.path.join(RESOURCE_DIR, "normalized_average_transformed_new.nii.gz")

brain_mask = nib.load(AVERAGE_TEMPLATE).get_fdata()
brain_mask = (brain_mask > 0).astype(np.float32)  # Convert to binary (1 inside brain, 0 outside)
brain_mask = torch.from_numpy(brain_mask)  # Convert to tensor

LEFT_MASK_PATH = os.path.join(RESOURCE_DIR, "Left_mask.nii.gz")
RIGHT_MASK_PATH = os.path.join(RESOURCE_DIR, "Right_mask.nii.gz")

LEFT_MASK = nib.load(LEFT_MASK_PATH).get_fdata()
RIGHT_MASK = nib.load(RIGHT_MASK_PATH).get_fdata()

# Model definitions
MODELS = {
    "ASSOCIATION FIBERS": {"model": "model_ASS.pt", "average": "normalized_average_ASS.nii.gz", "image": "ASS.png"},
    "PROJECTION FIBERS": {"model": "model_PRO.pt", "average": "normalized_average_PRO.nii.gz", "image": "PRO.png"},
    "COMMISSURAL FIBERS": {"model": "model_COM.pt", "average": "normalized_average_COM.nii.gz", "image": "COM.png"},
    "WHOLE BRAIN": {"model": "trained_model.pt", "average": "normalized_average_transformed_new.nii.gz", "image": "whole.png"},
}

def run_app():
    try:
        app = QApplication(sys.argv)
        app.setApplicationName("Deep-Disconnectome")
        if sys.platform == "darwin":
            app.setApplicationDisplayName("Deep-Disconnectome")
        app.setStyle("Fusion")

        app.setWindowIcon(QIcon(os.path.join(RESOURCE_DIR, "icon_creation.png")))

        window = DisconnectomeApp()
        window.setWindowTitle("Deep-Disconnectome")

        app.processEvents()
        sys.exit(app.exec())
    except Exception as e:
        print(f"Error: {e}")


transform_lesion = transforms.Resize(spatial_size=(96, 96, 96), mode='nearest')
transform_template = transforms.Compose([
    transforms.Resize(spatial_size=(96, 96, 96)),
    transforms.NormalizeIntensity(),
])

brain_mask = nib.load(AVERAGE_TEMPLATE).get_fdata()
brain_mask = (brain_mask > 0).astype(np.float32)  # Convert to binary (1 inside brain, 0 outside)
brain_mask = torch.from_numpy(brain_mask)  # Convert to tensor



def run_inference(model, lesion_data, template_data, transform_back, device):
    try:

        print(f"Initial lesion data shape: {lesion_data.shape}")
        print(f"Initial template data shape: {template_data.shape}")

        # ✅ Resize lesion data to match the target size (96, 96, 96)
        target_shape = (96, 96, 96)
        lesion_data_resized = F.interpolate(lesion_data.unsqueeze(0), size=target_shape, mode='nearest')
        lesion_data_resized = lesion_data_resized.squeeze(0)  # Remove batch dimension after resizing
        print(f"Lesion data shape after resize: {lesion_data_resized.shape}")

        # ✅ Ensure lesion has correct shape [1, 1, 96, 96, 96]
        lesion_data_resized = lesion_data_resized.unsqueeze(0).unsqueeze(0)  # [1, 1, 96, 96, 96]
        print(f"Lesion data shape after unsqueeze: {lesion_data_resized.shape}")

        # ✅ Ensure template has correct shape [1, 1, 96, 96, 96]
        template_data = template_data.unsqueeze(0).unsqueeze(0)  # [1, 1, 96, 96, 96]
        print(f"Template data shape after unsqueeze: {template_data.shape}")

        # ❌ REMOVE EXTRA DIMENSION (Fix the issue)
        if lesion_data_resized.shape[2] == 1:
            lesion_data_resized = lesion_data_resized.squeeze(2)  # Remove dim=2 if it's 1
        if template_data.shape[2] == 1:
            template_data = template_data.squeeze(2)

        print(f"Lesion data shape after squeeze: {lesion_data_resized.shape}")
        print(f"Template data shape after squeeze: {template_data.shape}")

        # ✅ Convert both to float32
        lesion_data_resized = lesion_data_resized.float()
        template_data = template_data.float()

        # ✅ Concatenate along the channel dimension
        input_data = torch.cat((lesion_data_resized, template_data), dim=1)  # Shape: [1, 2, 96, 96, 96]
        print(f"Input data shape after concatenation: {input_data.shape}")
        print(f"Running inference on input data shape: {input_data.shape}")

        # ✅ Run inference
         # Move to GPU/CPU

        input_data = input_data.to(device)
        print(f" input data shape when to device: {input_data.shape}")

        with torch.no_grad():
            model.eval()
            print(f"Input data type: {type(input_data)}, shape: {input_data.shape}")
            print(f"torch.Tensor is: {torch.Tensor}")

            try:
                y_pred = model(input_data)
            except Exception as e:
                print(f"Error when calling model: {e}")

            # ✅ Transform the output back to original dimensions
            y_pred_np = y_pred.squeeze().cpu().detach().numpy()
            y_pred_np = np.expand_dims(y_pred_np, axis=0)
            y_pred_np *= brain_mask.numpy()

            y_pred_np[y_pred_np <= 0.1] = 0

            y_tf = transform_back(y_pred_np)
            y_tf = y_tf.squeeze()
            print(f"Output data shape after transform: {y_tf.shape}")

            return y_tf

    except Exception as e:
        print(f"Error in run_inference: {e}")
        return None

def preprocess_input(input_path):
    lesion_data = nib.load(input_path).get_fdata()
    return torch.from_numpy(lesion_data).unsqueeze(0).float()

def preprocess_template():
    template_data = nib.load(AVERAGE_TEMPLATE).get_fdata()
    return torch.from_numpy(template_data).unsqueeze(0).float()

def preprocess_mask():
    mask_data = nib.load(AFFINE_FILE).get_fdata()
    return torch.from_numpy(mask_data)

def save_output(image, output_path):
    nifti_image = nib.Nifti1Image(image.astype(np.float32), affine=nib.load(AFFINE_FILE).affine)
    nib.save(nifti_image, output_path)
# Set the application icon in the macOS Dock

def load_model(model_path, device):
    """Load a trained PyTorch model from a file."""
    try:
        print(f"Attempting to load model from: {model_path}")  # Debugging

        if not os.path.exists(model_path):
            print(f"Error: Model file does not exist at {model_path}")
            return None

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)

        # Ensure the model architecture matches the trained model
        model = UNet3D(in_channels=2, out_channels=1, num_filters=8).to(device)  # Ensure num_filters=8

        # Print model shape before loading
        for name, param in model.named_parameters():
            print(f"Model layer {name}: {param.shape}")

        # Load state dictionary
        model.load_state_dict(checkpoint)

        model.eval()
        print("Model successfully loaded.")
        return model

    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def detect_lesion_side(lesion_data):
    lesion = lesion_data.squeeze().numpy()
    left_overlap = np.sum(lesion * LEFT_MASK)
    right_overlap = np.sum(lesion * RIGHT_MASK)
    print(f"Lesion left overlap: {left_overlap}, right overlap: {right_overlap}")
    return "LEFT" if left_overlap > right_overlap else "RIGHT"

def apply_hemisphere_mask(output_data, side):
    """Zero out the opposite hemisphere in the output."""
    if side == "LEFT":
        return output_data * LEFT_MASK
    elif side == "RIGHT":
        return output_data * RIGHT_MASK
    else:
        return output_data  # fallback in case of ambiguity



# Model Selection Window
class ModelSelectionWindow(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.setWindowTitle("Select Model")
        self.setGeometry(100, 100, 800, 300)
        layout = QVBoxLayout()

        title_label = QLabel("Select a Model for Reconstruction")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        model_layout = QHBoxLayout()

        for model_name, data in MODELS.items():
            model_widget = QWidget()
            model_vbox = QVBoxLayout()
            model_label = QLabel(model_name)
            model_label.setAlignment(Qt.AlignCenter)

            image_path = os.path.join(RESOURCE_DIR, data["image"])
            pixmap = QPixmap(image_path)
            image_label = QLabel()
            image_label.setPixmap(pixmap.scaled(150, 150, Qt.KeepAspectRatio))
            image_label.setAlignment(Qt.AlignCenter)

            select_button = QPushButton(f"Select {model_name}")
            select_button.clicked.connect(lambda _, m=model_name: self.select_model(m))

            model_vbox.addWidget(model_label)
            model_vbox.addWidget(image_label)
            model_vbox.addWidget(select_button)
            model_widget.setLayout(model_vbox)
            model_layout.addWidget(model_widget)

        layout.addLayout(model_layout)
        self.setLayout(layout)

    def select_model(self, model_name):
        """Set the selected model, close selection, and update the main app."""
        print(f"Model Selected: {model_name}")  # Debugging output
        self.parent.selected_model = model_name  # Store selected model

        self.close()  # Close model selection window

        self.parent.update_ui_after_model_selection()
        self.parent.generate_button.setEnabled(True)
 # Update the UI
        self.parent.show()
        self.parent.activateWindow()  # Bring main window to the front



# DisconnectomeApp (Main Application)
class DisconnectomeApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon(os.path.join(RESOURCE_DIR, "icon_creation.png")))

        self.setGeometry(100, 100, 800, 600)
        self.selected_model = None  # Ensure this is initialized

        # Layout and widgets
        layout = QVBoxLayout()


        self.label = QLabel("Step 1: Select a Model")
        layout.addWidget(self.label)
         # Set spacing between widgets and set margins
        layout.setSpacing(20)  # Add spacing between widgets
        layout.setContentsMargins(20, 20, 20, 20)  # Add margins around the whole layout

        self.label = QLabel("Step 1: Load a lesion mask or directory of lesion masks")
        layout.addWidget(self.label)

        # Load mask button
        self.load_button = QPushButton("Load Lesion Mask")
        self.load_button.clicked.connect(self.load_mask)
        layout.addWidget(self.load_button)

        # Load directory button
        self.load_directory_button = QPushButton("Load Lesion Mask Directory")
        self.load_directory_button.clicked.connect(self.load_mask_directory)
        layout.addWidget(self.load_directory_button)

        # Output directory button
        self.select_output_button = QPushButton("Select Output Directory")
        self.select_output_button.clicked.connect(self.select_output_directory)
        layout.addWidget(self.select_output_button)

        # Button to change model
        self.change_model_button = QPushButton("Change Model")
        #self.change_model_button.setStyleSheet("background-color: blue; color: white; font-weight: bold; font-size: 14px;")
        self.change_model_button.clicked.connect(self.show_model_selection_window)
        layout.addWidget(self.change_model_button)


        # Label to display selected mask files and output folder
        self.mask_label = QLabel("No masks selected.")
        layout.addWidget(self.mask_label)

        self.output_label = QLabel("No output directory selected.")
        layout.addWidget(self.output_label)

        #self.mask_scroll_area = QScrollArea(self)
        #self.mask_widget = QWidget(self)
        #self.mask_layout = QVBoxLayout(self.mask_widget)
        #self.mask_widget.setLayout(self.mask_layout)
        #self.mask_scroll_area.setWidgetResizable(True)
        #self.mask_scroll_area.setWidget(self.mask_widget)
        #layout.addWidget(self.mask_scroll_area)

        # Progress bars layout for each mask
        self.progress_layout = QVBoxLayout()
        layout.addLayout(self.progress_layout)

        # Generate button (disabled initially)
        self.generate_button = QPushButton("Generate Deep-Disconnectomes")
        self.generate_button.setStyleSheet("color: #FFD700; font-size: 14px;")

        self.generate_button.setEnabled(False)
        self.generate_button.clicked.connect(self.generate_disconnectomes)
        layout.addWidget(self.generate_button)

        self.exit_button = QPushButton("Exit", self)
        self.exit_button.setStyleSheet("background-color: red; color: white; font-weight: bold; font-size: 14px;")
        self.exit_button.setVisible(False)  # Hide the button initially
        self.exit_button.clicked.connect(self.close_application)

        # Label for success message (initially hidden)
        self.success_label = QLabel("")
        self.success_label.setStyleSheet("background-color: transparent; color: #50C878; font-weight: bold; font-size: 15px;")
        self.success_label.setAlignment(Qt.AlignCenter)
        self.success_label.setVisible(False)  # Initially hidden
        layout.addWidget(self.success_label)

        self.success_image_label = QLabel()
        #self.success_image_label.setAlignment(Qt.AlignCenter)
        #self.success_image_label.setVisible(False)  # Initially hidden
        layout.addWidget(self.success_image_label)

        exit_layout = QVBoxLayout()
        exit_layout.addStretch()  # Push the button to the bottom
        exit_layout.addWidget(self.exit_button)
        layout.addLayout(exit_layout)

        # Central widget
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
        self.mask_paths = []  # List for storing selected masks
        self.output_path = None
        self.show_model_selection_window()

    def show_model_selection_window(self):
        """Open the model selection window if not already open."""
        if hasattr(self, 'model_selection_window') and self.model_selection_window.isVisible():
            return  # Window already open, don't open another
    
        self.model_selection_window = ModelSelectionWindow(self)
        self.model_selection_window.show()


    def update_ui_after_model_selection(self):
        """Update UI after model is selected."""
        if not self.selected_model:
            return

        # Update label to show selected model
        self.label.setText(f"Selected Model: {self.selected_model}")
        self.label.setStyleSheet("font-size: 14px; color: #FFD700;")

        self.change_model_button.setVisible(True)  # Show "Change Model" button
        self.generate_button.setEnabled(True)  # Enable the generate button


    def load_mask(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog  # Disable native file dialog

        self.mask_path, _ = QFileDialog.getOpenFileName(self, "Load Lesion Mask", "", "NIfTI Files (*.nii.gz)", options=options)

        if self.mask_path:
            self.mask_paths = [self.mask_path]
            self.mask_label.setText(f"Loaded: {os.path.basename(self.mask_path)}")
            self.generate_button.setEnabled(True)
            self.generate_button.setVisible(True)

            self.success_label.setVisible(False)
            # Change the appearance of the button to indicate something is selected
            self.load_button.setStyleSheet("background-color: lightgray; color: black; font-weight: bold;")
            #self.update_mask_scroll_area()




    def load_mask_directory(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog  # Disable native file dialog

        directory = QFileDialog.getExistingDirectory(self, "Select Lesion Mask Directory", options=options)

        if directory:
            self.mask_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.nii.gz')]

            if self.mask_paths:
                mask_names = "\n".join(os.path.basename(mask) for mask in self.mask_paths)
                #self.mask_label.setText(f"Loaded {len(self.mask_paths)} masks from {directory}:\n{mask_names}")
                self.mask_label.setText(f"Loaded {len(self.mask_paths)}")
                self.generate_button.setEnabled(True)
                self.generate_button.setVisible(True)

                #self.update_mask_scroll_area()

                #self.load_button.setStyleSheet("background-color: transparent; color: transparent; font-weight: normal;")
                self.load_directory_button.setStyleSheet("background-color: lightgray; color: black; font-weight: bold;")
            else:
                self.mask_label.setText("No valid lesion mask files found in the selected directory.")
                self.generate_button.setEnabled(False)

    def select_output_directory(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog  # Disable native file dialog

        output_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory", options=options)
        if output_dir:
            self.output_path = output_dir
            self.output_label.setText(f"Output Directory: {output_dir}")
            self.select_output_button.setStyleSheet("background-color: lightgray; color: black; font-weight: bold;")
            self.success_label.setVisible(False)

    def close_application(self):
        self.close()  # Close the window
        QApplication.quit()
###
    def hide_success_message(self):
        """Hides the success message and image, then displays instructions in the same position."""
        self.success_label.setText("To generate another deep-disconnectome, load a new lesion or lesion directory.")
        self.success_label.setStyleSheet("color: #A9A9A9; font-size: 14px;")  # Gray text for subtle effect
        self.success_label.setVisible(True)  # Show new message in the same position
        self.success_image_label.setVisible(False)  # Hide success image

        # ✅ Disable the generate button until a new input is loaded
        self.generate_button.setEnabled(False)


    def generate_disconnectomes(self):
        if not self.mask_paths:
            self.label.setText("No mask(s) selected!")
            self.label.setStyleSheet("color: red; font-weight: bold; font-size: 14px;")
            self.label.setAlignment(Qt.AlignCenter)
            return

        if not self.output_path:
            self.output_path = os.path.dirname(self.mask_paths[0])
            self.output_label.setText(f"Output Directory: {self.output_path}")

        if not self.output_path:
            self.label.setText("No output directory selected!")
            self.label.setStyleSheet("color: red; font-weight: bold; font-size: 14px;")
            self.label.setAlignment(Qt.AlignCenter)
            return

        self.label.setText("Generating deep-disconnectomes...")
        self.generate_button.setEnabled(False)  # Disable button during process

        try:
            # Load the model once
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model_path = os.path.join(RESOURCE_DIR, MODELS[self.selected_model]["model"])

            print(f"Attempting to load model: {self.selected_model} from {model_path}")  # Debugging

            if not os.path.exists(model_path):
                print(f"Error: Model file does not exist at {model_path}")
                self.label.setText(f"Error: Model file not found ({self.selected_model})!")
                self.generate_button.setEnabled(True)
                return  # Exit function if model file is missing

            model = load_model(model_path, device)

            if model is None:
                print("Model loading failed. Check the model path and format.")
                return# Stop execution if model didn't load

            print(f"Loaded model successfully: {type(model)}")  # Debugging

            avg_template = os.path.join(RESOURCE_DIR, MODELS[self.selected_model]["average"])
            print(f"Using template: {avg_template}")  # Debugging

            template_data = nib.load(avg_template).get_fdata()
            template_data = torch.from_numpy(template_data).unsqueeze(0).float()
            print(f"Preprocessed template data")

            transform_back = tio.Resize(target_shape=(91, 109, 91))

            for mask_path in self.mask_paths:
                progress_bar = QProgressBar(self)
                self.progress_layout.addWidget(progress_bar)
                progress_bar.setRange(0, 100)
                progress_bar.setFormat(f"Generating: ({self.selected_model}) {os.path.basename(mask_path)}")
                progress_bar.setValue(0)

                lesion_data = preprocess_input(mask_path)
                lesion_side = detect_lesion_side(lesion_data)
                print(f"Detected lesion is on the {lesion_side} side.")

                progress_bar.setValue(20)
                QApplication.processEvents()

                output_file = os.path.join(
                    self.output_path,
                    os.path.basename(mask_path).replace(".nii.gz", f"_deep_disco_{self.selected_model}.nii.gz")
                )

                print(f"Model type before inference: {type(model)}")  # Debugging
                progress_bar.setValue(40)
                QApplication.processEvents()

                # Run inference
                deep_disco_output = run_inference(
                    model=model,
                    lesion_data=lesion_data,
                    template_data=template_data,
                    transform_back=transform_back,
                    device=device
                )

                progress_bar.setValue(60)
                QApplication.processEvents()
                
                if self.selected_model in ["ASSOCIATION FIBERS", "PROJECTION FIBERS"]:
                    print(f"Applying hemisphere masking for model: {self.selected_model}")
                    deep_disco_output = apply_hemisphere_mask(deep_disco_output, lesion_side)

                if deep_disco_output is not None:
                    save_output(deep_disco_output, output_file)
                    progress_bar.setValue(80)
                    QApplication.processEvents()
                    progress_bar.setValue(100)

                    self.label.setStyleSheet("color: yellow; font-weight: normal; font-size: 14px;")
                    self.label.setText(f"Saved: {output_file}")
                else:
                    self.label.setText(f"Failed processing {mask_path}")
                    progress_bar.setValue(0)

                QApplication.processEvents()  # Update the UI during processing

            self.success_label.setText("Deep-Disconnectomes were successfully generated.")
            self.success_label.setVisible(True)  # Show success message
            self.success_label.setAlignment(Qt.AlignCenter)  # Center the text

            # Load and show the PNG image
            IMAGE = os.path.join(RESOURCE_DIR, "success_image.png")  # Provide the correct image path
            pixmap = QPixmap(IMAGE)
            scaled_pixmap = pixmap.scaled(150, int(pixmap.height() * 150 / pixmap.width()), Qt.KeepAspectRatio)

            self.success_image_label.setPixmap(scaled_pixmap)
            self.success_image_label.setAlignment(Qt.AlignCenter)

            #self.generate_button.setEnabled(True)
            self.exit_button.setVisible(True)

            QTimer.singleShot(5000, self.hide_success_message)

            self.generate_button.setVisible(False)
            # Reset mask selection UI after generation
            self.mask_paths = []
            self.load_button.setStyleSheet("")  # Reset to default
            self.load_directory_button.setStyleSheet("")  # Reset to default
            self.mask_label.setText("No masks selected.")
            self.generate_button.setEnabled(False)  # Ensure the button is disabled again


        except Exception as e:
            self.label.setText(f"Error occurred: {e}")

        self.generate_button.setEnabled(True)



# Run the application
if __name__ == "__main__":
    run_app()
