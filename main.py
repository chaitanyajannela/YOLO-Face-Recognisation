import cv2
import matplotlib.pyplot as plt
from mtcnn import MTCNN
import os

# Load the MTCNN model
detector = MTCNN()

# Function to detect faces in an image and display with confidence scores
def detect_faces_with_mtcnn(image_path):
    # Check if the file exists
    if not os.path.exists(image_path):
        print(f"Error: The file {image_path} does not exist.")
        return

    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Unable to read the image.")
        return

    # Convert the image from BGR to RGB for MTCNN
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect faces in the image
    faces = detector.detect_faces(img_rgb)

    # Loop through detected faces and draw rectangles with confidence scores
    for face in faces:
        x, y, width, height = face['box']
        confidence = face['confidence']

        # Draw rectangle around the face
        cv2.rectangle(img, (x, y), (x + width, y + height), (255, 0, 0), 2)

        # Display confidence score on the image
        cv2.putText(img, f'Confidence: {confidence*100:.2f}%', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 2)

    # Convert the image from BGR to RGB for matplotlib display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display the image using matplotlib
    plt.imshow(img_rgb)
    plt.axis('off')  # Hide axis
    plt.title("Detected Faces with Confidence Scores")
    plt.show()

    # Optionally, save the result image
    cv2.imwrite('output_with_mtcnn_faces_and_confidence.jpg', img)

# Example usage
detect_faces_with_mtcnn('WhatsApp Image 2024-08-29 at 23.12.55_ca7f7b51.jpg')  # Replace with your image path
