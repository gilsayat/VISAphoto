# US Visa Photo Automator

A fully self-contained, offline Windows desktop utility that automatically crops, centers, pads, and replaces backgrounds for US Visa 2x2" photo compliance. Created by Gil.

## Prerequisites & Setup

1. Create a folder named `Models` in the project directory (or next to the published `.exe`).
2. **Background Removal Model**: Download `u2net.onnx` and place in `Models/u2net.onnx`.
3. **Facial Landmark Model**: Download `shape_predictor_68_face_landmarks.dat` and place in `Models/shape_predictor_68_face_landmarks.dat`.

## Building the Executable

Run `publish.bat` to generate a single `.exe` file that runs without any .NET installation required.
