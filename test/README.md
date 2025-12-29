# Tangram Piece Detection

This project identifies the coordinates and orientation of tangram pieces from an image.

## Requirements

- Python 3.x
- OpenCV
- NumPy
- Matplotlib

## Installation

1. Create a virtual environment (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place your tangram image in the project directory and name it `tangram.jpg` (or update the path in `main.py`).
2. Run the script:
   ```bash
   python main.py
   ```
3. The result will be saved as `result.jpg` with contours and orientation info drawn.
