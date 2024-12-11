import cv2
import face_recognition
import firebase_admin
from firebase_admin import credentials, storage, db
import numpy as np
import os
import time
from PIL import Image
import json
from pathlib import Path
import sys
import base64

# ============================ #
#       Progress Bar Function  #
# ============================ #

def print_progress_bar(current, total, prefix='Facial Recognition Progress', length=50, fill='█', accuracy=None):
    try:
        percent = 100 * (current / float(total)) if total else 100.0
        percent_str = "{0:.1f}".format(percent)
        filled_length = int(length * current // total) if total else length

        bar = fill * filled_length + '-' * (length - filled_length)
        if accuracy is not None:
            sys.stdout.write(f'\r{prefix} |{bar}| {percent_str}% Accuracy: {accuracy:.2f}%')
        else:
            sys.stdout.write(f'\r{prefix} |{bar}| {percent_str}%')
        
        sys.stdout.flush()
        if current == total:
            print()  # Move to the next line upon completion
    except Exception as e:
        print(f"\nError in print_progress_bar: {e}")

# ============================ #
#       Firebase Setup         #
# ============================ #

# Firebase Initialization
if not firebase_admin._apps:
    try:
        cred = credentials.Certificate(r"pictora-7f0ad-firebase-adminsdk-hpzf5-f730a1a51c.json")  # Update path
        firebase_admin.initialize_app(cred, {
            'storageBucket': 'pictora-7f0ad.appspot.com',
            'databaseURL': 'https://pictora-7f0ad-default-rtdb.asia-southeast1.firebasedatabase.app/'
        })
        print("Firebase initialized successfully.")
    except Exception as e:
        print(f"\nError initializing Firebase: {e}")
        sys.exit(1)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Path for the local directory to store images
LOCAL_IMAGE_DIR = Path(__file__).resolve().parent / "local_images"

# Create the directory if it doesn't exist
if not LOCAL_IMAGE_DIR.exists():
    try:
        LOCAL_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"\nError creating local images directory: {e}")
        sys.exit(1)

# Path for the temporary file to store the last processed sortPhotoRequest values by room
TEMP_FILE_PATH = r'D:\rootpictora\last_sort_photo_request.json'

# ============================ #
#       Helper Functions       #
# ============================ #

def get_last_sort_photo_request(room_code):
    try:
        if os.path.exists(TEMP_FILE_PATH):
            with open(TEMP_FILE_PATH, 'r') as temp_file:
                data = json.load(temp_file)
                return data.get(room_code, -1)  # Default to -1 if no value exists for the room
        return -1  # If the file does not exist, return -1
    except Exception as e:
        print(f"\nError reading temporary file: {e}")
        return -1

def save_last_sort_photo_request(room_code, value):
    try:
        if os.path.exists(TEMP_FILE_PATH):
            with open(TEMP_FILE_PATH, 'r') as temp_file:
                data = json.load(temp_file)
        else:
            data = {}

        data[room_code] = value  # Update the sortPhotoRequest for the room

        with open(TEMP_FILE_PATH, 'w') as temp_file:
            json.dump(data, temp_file)
    except Exception as e:
        print(f"\nError writing to temporary file: {e}")

def crop_faces(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"\nError: Could not load image at path {image_path}")
            return []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        cropped_faces = [image[y:y+h, x:x+w] for (x, y, w, h) in faces]
        return cropped_faces
    except Exception as e:
        print(f"\nError in crop_faces: {e}")
        return []

def get_embedding(image):
    # Revert embedding generation logic to be identical to the first code
    try:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_image)
        if encodings:
            return encodings[0]
        return None
    except Exception as e:
        print(f"\nError in get_embedding: {e}")
        return None

def process_image_with_retry(file_path, func, retries=5, delay=1):
    for attempt in range(retries):
        try:
            return func(file_path)
        except OSError as e:
            if hasattr(e, 'errno') and e.errno == 32:  # File is being used by another process
                print(f"\nFile {file_path} is locked, retrying ({attempt + 1}/{retries})...")
                time.sleep(delay)
            else:
                print(f"\nOS error while processing file {file_path}: {e}")
                break
    print(f"\nFailed to process file {file_path} after {retries} retries.")
    return []

def calculate_accuracy(matched, unmatched, processed):
    total = matched + unmatched
    if total == 0:
        return 0.0
    return (matched / total) * 100

def create_confusion_matrix(matched, unmatched):
    TP = matched
    FN = unmatched
    FP = 0
    TN = 0

    print("\nConfusion Matrix:")
    print(f"{'':20} {'Predicted Matched':20} {'Predicted Unmatched':20}")
    print(f"{'Actual Matched':20} {TP:<20} {FN:<20}")
    print(f"{'Actual Unmatched':20} {FP:<20} {TN:<20}")

def show_accuracy_table(overall_accuracy):
    print("\nAccuracy Summary:")
    print("+----------------------+----------------------+")
    print("|       Metric         |      Percentage      |")
    print("+----------------------+----------------------+")
    print(f"| Overall Accuracy     | {overall_accuracy:>18.2f}% |")
    print("+----------------------+----------------------+")

# ============================ #
#    Facial Recognition Start  #
# ============================ #

def facialrecognitionstart(event_code):
    if not event_code:
        print("Error: Invalid event code.")
        return

    # Initialize variables for accuracy and confusion matrix
    matched_photos = 0
    unmatched_photos = 0

    # First, verify and store profile embeddings (like code 1)
    total_participants, successful_embeddings = verify_and_store_profile_embeddings(event_code)

    # Second, verify and store profile embeddings with progress
    verify_and_store_profile_embeddings_with_progress(event_code, total_participants)
    
    # Identify uploaded photos and get matched/unmatched counts
    total_photos, matched_photos, unmatched_photos = identify_uploaded_photos(event_code)
    
    # Calculate overall accuracy
    overall_accuracy = (matched_photos / total_photos * 100) if total_photos else 0.0

    # Print final summary
    print("\nPhoto identification completed.")
    print(f"Total Photos Processed: {total_photos}")
    print(f"Matched Photos: {matched_photos}")
    print(f"Unmatched Photos: {unmatched_photos}")
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")

    # Create and display confusion matrix
    create_confusion_matrix(matched_photos, unmatched_photos)

    # Show accuracy table
    show_accuracy_table(overall_accuracy)

    # Encrypted Developed By message
    encrypted_message = "RGV2ZWxvcGVkIGJ5IDoKMS4gTXVobWFkIFRhaG1pZHVyIFJhaG1hbgoyLiBNb2hvc2luYSBJc2xhbSBEaXNoYQozLiBBbmlrYSBUYWJhc3N1bSBFbmFuCk5vcnRoIFNvdXRoIFVuaXZlcnNpdHksIEJhbmdsYWRhc2g="
    decoded_message = base64.b64decode(encrypted_message).decode('utf-8')
    print("\n" + decoded_message)

# ============================ #
#   Step 1: Verify & Embed     #
# ============================ #

def verify_and_store_profile_embeddings(event_code):
    try:
        manual_ref = db.reference(f'rooms/{event_code}/manualParticipants')
        participants_ref = db.reference(f'rooms/{event_code}/participants')

        manual_data = manual_ref.get() or {}
        joined_data = participants_ref.get() or {}
        participants_data = {**manual_data, **joined_data}

        verified_embeddings = {}
        total_participants = len(participants_data)
        if total_participants == 0:
            print("No participants found for embedding verification.")
            return 0, 0

        successful_embeddings = 0

        for participant_id, data in participants_data.items():
            profile_url = data.get("photoUrl")
            if profile_url:
                print(f"Processing profile photo for participant: {participant_id}")
                try:
                    path_start = profile_url.find("/o/") + 3
                    path_end = profile_url.find("?alt=media")
                    relative_path = profile_url[path_start:path_end].replace("%2F", "/")

                    blob = storage.bucket().blob(relative_path)
                    local_image_path = LOCAL_IMAGE_DIR / f"{participant_id}_profile.jpg"
                    blob.download_to_filename(str(local_image_path))
                    faces = process_image_with_retry(str(local_image_path), crop_faces)

                    # Fallback: If no faces found, try embedding on the entire image directly
                    if not faces:
                        full_image = cv2.imread(str(local_image_path))
                        if full_image is not None:
                            embedding = get_embedding(full_image)
                            if embedding is not None:
                                verified_embeddings[participant_id] = embedding.tolist()
                                successful_embeddings += 1
                                print(f"Stored embedding for {participant_id}.")
                            else:
                                print(f"Failed to generate embedding for {participant_id}: Embedding is None.")
                        else:
                            print(f"Failed to generate embedding for {participant_id}: No faces detected and no fallback image.")
                    else:
                        embedding = get_embedding(faces[0])
                        if embedding is not None:
                            verified_embeddings[participant_id] = embedding.tolist()
                            successful_embeddings += 1
                            print(f"Stored embedding for {participant_id}.")
                        else:
                            print(f"Failed to generate embedding for {participant_id}: Embedding is None.")
                    
                    os.unlink(local_image_path)  # Delete the image after processing

                except Exception as e:
                    print(f"Error processing profile photo for participant {participant_id}: {e}")
            else:
                print(f"Failed to generate embedding for {participant_id}: No photo URL provided.")

        db.reference(f'verifiedEmbeddings/{event_code}').set(verified_embeddings)
        print(f"Verified embeddings stored for event {event_code}.")

        return total_participants, successful_embeddings

    except Exception as e:
        print(f"Error in verify_and_store_profile_embeddings: {e}")
        return 0, 0

def verify_and_store_profile_embeddings_with_progress(event_code, total_participants):
    try:
        current_embeddings = db.reference(f'verifiedEmbeddings/{event_code}').get() or {}
       
        manual_ref = db.reference(f'rooms/{event_code}/manualParticipants')
        participants_ref = db.reference(f'rooms/{event_code}/participants')

        manual_data = manual_ref.get() or {}
        joined_data = participants_ref.get() or {}
        participants_data = {**manual_data, **joined_data}

        verified_embeddings = {}

        for participant_id, data in participants_data.items():
            # Skip if embedding already exists
            if participant_id in current_embeddings:
                print(f"Embedding already exists for participant {participant_id}. Skipping.")
                continue

            profile_url = data.get("photoUrl")
            if profile_url:
                print(f"Processing profile photo for participant: {participant_id}")
                try:
                    path_start = profile_url.find("/o/") + 3
                    path_end = profile_url.find("?alt=media")
                    relative_path = profile_url[path_start:path_end].replace("%2F", "/")

                    blob = storage.bucket().blob(relative_path)
                    local_image_path = LOCAL_IMAGE_DIR / f"{participant_id}_profile.jpg"
                    blob.download_to_filename(str(local_image_path))
                    faces = process_image_with_retry(str(local_image_path), crop_faces)

                    # Fallback: If no faces found
                    if not faces:
                        full_image = cv2.imread(str(local_image_path))
                        if full_image is not None:
                            embedding = get_embedding(full_image)
                            if embedding is not None:
                                verified_embeddings[participant_id] = embedding.tolist()
                                print(f"Stored embedding for {participant_id}.")
                            else:
                                print(f"Failed to generate embedding for {participant_id}: Embedding is None.")
                        else:
                            print(f"Failed to generate embedding for {participant_id}: No faces detected and no fallback image.")
                    else:
                        embedding = get_embedding(faces[0])
                        if embedding is not None:
                            verified_embeddings[participant_id] = embedding.tolist()
                            print(f"Stored embedding for {participant_id}.")
                        else:
                            print(f"Failed to generate embedding for {participant_id}: Embedding is None.")

                    os.unlink(local_image_path)  # Delete the image after processing

                except Exception as e:
                    print(f"Error processing profile photo for participant {participant_id}: {e}")
            else:
                print(f"Failed to generate embedding for {participant_id}: No photo URL provided.")

        current_embeddings.update(verified_embeddings)
        db.reference(f'verifiedEmbeddings/{event_code}').set(current_embeddings)
        print(f"Verified embeddings stored for event {event_code}.")

    except Exception as e:
        print(f"\nError in verify_and_store_profile_embeddings_with_progress: {e}")

# ============================ #
#   Step 2: Photo Identification #
# ============================ #

def identify_uploaded_photos(event_code):
    matched_photos = 0
    unmatched_photos = 0
    total_photos = 0
    try:
        embeddings_ref = db.reference(f'verifiedEmbeddings/{event_code}')
        verified_embeddings = embeddings_ref.get()
        if not verified_embeddings:
            print(f"No verified embeddings found for event {event_code}.")
            return 0, 0, 0

        verified_embeddings_np = {pid: np.array(emb) for pid, emb in verified_embeddings.items()}

        host_folder_ref = db.reference(f'rooms/{event_code}/hostUploadedPhotoFolderPath')
        participants_ref = db.reference(f'rooms/{event_code}/participants')

        folder_path = host_folder_ref.get()
        participants_data = participants_ref.get() or {}

        participant_folders = [data.get("folderPath") for data in participants_data.values() if "folderPath" in data]

        # Combine and remove duplicates
        all_folder_paths = [folder_path] + participant_folders if folder_path else participant_folders
        all_folder_paths = list(set(all_folder_paths))  # Remove duplicates

        print(f"\nHost folder path: {folder_path}")
        print(f"Regular participant folder paths: {participant_folders}")
        print(f"All folder paths to process: {all_folder_paths}")
        print(f"Processing photos from {len(all_folder_paths)} folders...")

        bucket = storage.bucket()
        blobs = []
        for folder in all_folder_paths:
            if folder:
                blobs.extend(bucket.list_blobs(prefix=folder))
        
        if not blobs:
            print("No photos found in the specified folders.")
            return 0, 0, 0

        unmatched_folder = f"rooms/{event_code}/unmatched/"
        unmatched_counter = 1

        total_photos = len(blobs)
        print(f"Found {total_photos} photos to process.")
        print_progress_bar(0, total_photos, prefix='Facial Recognition Progress', length=50)

        for idx, blob in enumerate(blobs, 1):
            local_image_path = LOCAL_IMAGE_DIR / f"{blob.name.split('/')[-1]}"
            if not blob.exists():
                print(f"\nError: Blob {blob.name} does not exist.")
                unmatched_photos += 1
                accuracy = calculate_accuracy(matched_photos, unmatched_photos, idx)
                print_progress_bar(idx, total_photos, prefix='Facial Recognition Progress', length=50, accuracy=accuracy)
                continue

            try:
                blob.download_to_filename(str(local_image_path))
                print(f"\nDownloaded photo: {blob.name.split('/')[-1]}")
            except Exception as e:
                print(f"\nError downloading {blob.name}: {e}")
                unmatched_photos += 1
                accuracy = calculate_accuracy(matched_photos, unmatched_photos, idx)
                print_progress_bar(idx, total_photos, prefix='Facial Recognition Progress', length=50, accuracy=accuracy)
                continue

            faces = process_image_with_retry(str(local_image_path), crop_faces)

            matched_participants = set()
            for face in faces:
                embedding = get_embedding(face)
                if embedding is not None:
                    for participant_id, verified_embedding in verified_embeddings_np.items():
                        distance = np.linalg.norm(embedding - verified_embedding)
                        if distance < 0.45:  # Match threshold
                            matched_participants.add(participant_id)

            if matched_participants:
                matched_photos += 1
                for participant_id in matched_participants:
                    guest_folder = f"rooms/{event_code}/{participant_id}/photos/"
                    matched_blob = bucket.blob(f"{guest_folder}{os.path.basename(blob.name)}")
                    try:
                        matched_blob.upload_from_filename(str(local_image_path))
                        print(f"Matched photo uploaded to {guest_folder}{os.path.basename(blob.name)}")
                    except Exception as e:
                        print(f"Error uploading to {guest_folder}{os.path.basename(blob.name)}: {e}")
                print(f"Matched photo with participants: {', '.join(matched_participants)}")
            else:
                unmatched_path = f"{unmatched_folder}unmatched_{unmatched_counter}.jpg"
                unmatched_blob = bucket.blob(unmatched_path)
                try:
                    unmatched_blob.upload_from_filename(str(local_image_path))
                    print(f"Unmatched photo saved for review: {unmatched_path}")
                except Exception as e:
                    print(f"Error uploading to {unmatched_path}: {e}")
                unmatched_photos += 1
                unmatched_counter += 1

            try:
                os.unlink(local_image_path)
            except Exception as e:
                print(f"Error deleting local image {local_image_path}: {e}")

            accuracy = calculate_accuracy(matched_photos, unmatched_photos, idx)
            print_progress_bar(idx, total_photos, prefix='Facial Recognition Progress', length=50, accuracy=accuracy)

        return total_photos, matched_photos, unmatched_photos

    except Exception as e:
        print(f"\nError in identify_uploaded_photos: {e}")
        return 0, 0, 0


# ============================ #
#    Monitor Database Changes  #
# ============================ #

def listen_for_sort_photo_requests():
    ref = db.reference('/')

    def listener(event):
        try:
            if isinstance(event.data, dict) and 'sortPhotoRequest' in event.data:
                path_parts = event.path.strip("/").split('/')

                if len(path_parts) >= 2 and path_parts[0] == "rooms":
                    event_code = path_parts[1]
                    print(f"\nRoom code detected: {event_code}")

                    if event_code:
                        last_request = get_last_sort_photo_request(event_code)
                        current_request = event.data.get('sortPhotoRequest', -1)
                        if current_request > last_request:
                            facialrecognitionstart(event_code)
                            save_last_sort_photo_request(event_code, current_request)
                else:
                    print("Error: Room code not found in the path.")
        except Exception as e:
            print(f"\nError in listener: {e}")

    try:
        ref.listen(listener)
    except Exception as e:
        print(f"\nError setting up listener: {e}")

# ============================ #
#          Main Execution      #
# ============================ #

listen_for_sort_photo_requests()

try:
    while True:
        time.sleep(10)
except KeyboardInterrupt:
    print("\nScript terminated by user.")
