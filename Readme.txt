Listener+Facial Recognition
===========================

Overview
--------
**Listener+Facial Recognition** is a Python-based application that integrates facial recognition with Firebase to automatically sort and manage photos uploaded to specific Firebase Storage folders. The application listens for changes in the Firebase Realtime Database and processes photos by matching detected faces with verified participant embeddings. Matched photos are organized accordingly, while unmatched photos are saved for manual review.

Features
--------
- **Real-Time Monitoring**: Listens for `sortPhotoRequest` events in Firebase Realtime Database to trigger photo processing.
- **Facial Recognition**: Utilizes OpenCV and the `face_recognition` library to detect and encode faces.
- **Firebase Integration**: Downloads photos from Firebase Storage, uploads matched/unmatched photos, and manages embeddings in Firebase Realtime Database.
- **Progress Tracking**: Displays a dynamic progress bar with accuracy metrics during photo processing.
- **Error Handling**: Robust error handling to manage file access issues, Firebase errors, and more.
- **Temporary File Cleanup**: Includes a script to clean up temporary files generated during processing.

Table of Contents
-----------------
1. Prerequisites
2. Installation
3. Configuration
4. Usage
5. Project Structure
6. Troubleshooting
7. Contributing
8. License
9. Citation

Prerequisites
-------------
Before setting up the application, ensure you have the following:

- **Python 3.7 or higher**: [Download Python](https://www.python.org/downloads/)
- **Firebase Project**: Set up a Firebase project with Realtime Database and Storage enabled.
- **Firebase Admin SDK**: Obtain the Firebase Admin SDK JSON credentials file.

Installation
------------
1. **Clone the Repository**
git clone https://github.com/MuhammadTahmidurRahman/Face_Recognition_UsingOPENCV_HARCASCADE_FACE_RECOGNITION-pictorabackend-


2. **Create a Virtual Environment (Optional but Recommended)**

python -m venv venv source venv/bin/activate # On Windows: venv\Scripts\activate


3. **Install Dependencies**

pip install -r requirements.txt


*If `requirements.txt` is not provided, install the necessary packages manually:*

pip install opencv-python face_recognition firebase_admin numpy pillow


Configuration
-------------
1. **Firebase Setup**

- **Obtain Firebase Credentials**:
  - Go to your Firebase project.
  - Navigate to **Project Settings** > **Service Accounts**.
  - Click **Generate new private key** and save the JSON file.

- **Place Credentials File**:
  - Save the downloaded JSON file in the project directory.
  - Update the path in `Listener+Facial Recognition.py`:

    ```python
    cred = credentials.Certificate(r"path_to_your_firebase_credentials.json")
    ```

- **Configure Firebase Realtime Database and Storage Rules** as needed for your application.

2. **Directory Setup**

- **Local Image Directory**:
  - The script uses a `local_images` directory to store temporary images.
  - This directory is created automatically if it doesn't exist.

- **Temporary File Path**:
  - The script uses a temporary JSON file to track the last `sortPhotoRequest` per room.
  - Update `TEMP_FILE_PATH` if necessary:

    ```python
    TEMP_FILE_PATH = r'D:\rootpictora\last_sort_photo_request.json'
    ```

3. **Update Script Paths**

- Ensure all file paths in `Listener+Facial Recognition.py` and `temp.py` are correct and accessible.

Usage
-----
1. **Run the Listener Script**

python "Listener+Facial Recognition.py"


The script will initialize Firebase, set up listeners for database changes, and begin monitoring for `sortPhotoRequest` events.

2. **Triggering Photo Sorting**

- To initiate photo sorting for a specific room, update the `sortPhotoRequest` value in your Firebase Realtime Database under the corresponding room code.

3. **Running the Temporary File Cleanup Script**

The `temp.py` script is used to delete a specific temporary file.


python temp.py


*Ensure the `temp_file_path` in `temp.py` is correct before running.*

Project Structure
-----------------


Listener+Facial Recognition/ ├── Listener+Facial Recognition.py ├── temp.py ├── local_images/ # Directory for temporary image storage ├── pictora-7f0ad-firebase-adminsdk-hpzf5-f730a1a51c.json # Firebase credentials ├── last_sort_photo_request.json # Temporary tracking file ├── requirements.txt # Python dependencies ├── README.md # Project documentation └── LICENSE.md # MIT License



Troubleshooting
---------------
- **Firebase Initialization Error**:
  - Ensure the Firebase credentials JSON file path is correct.
  - Verify that the Firebase project has the necessary permissions.

- **Face Detection Issues**:
  - Ensure that the Haar Cascade XML file is correctly loaded.
  - Verify that the images contain clear, front-facing faces.

- **File Access Errors**:
  - Ensure that the script has the necessary permissions to read/write/delete files in the specified directories.

- **Dependency Issues**:
  - Make sure all required Python packages are installed.
  - Use a virtual environment to avoid version conflicts.

- **Progress Bar Not Displaying Correctly**:
  - Ensure the terminal supports carriage return `\r` for dynamic updates.

Contributing
------------
Contributions are welcome! Please follow these steps:

 1. **Fork the Repository**
 2. **Create a Feature Branch**
git checkout -b feature/YourFeature
 3. **Commit Your Changes**
 4. **Push to the Branch**
git push origin feature/YourFeature
 5. **Open a Pull Request**

License
-------
This project is licensed under the MIT License. See the [LICENSE.md](LICENSE.md) file for details.

Citation
--------
If you use this code in your research, please cite it using the following format:


@software{ListenerFacialRecognition2024, author = {Muhamad Tahmidur Rahman and Mohosina Islam Disha, title = {Listener+Facial Recognition}, year = {2024}, publisher = {GitHub}, version = {1.0.0}

https://github.com/MuhammadTahmidurRahman/Face_Recognition_UsingOPENCV_HARCASCADE_FACE_RECOGNITION-pictorabackend-

Developed By:
1. Muhamad Tahmidur Rahman  
2. Mohosina Islam Disha  
North South University, Bangladesh








