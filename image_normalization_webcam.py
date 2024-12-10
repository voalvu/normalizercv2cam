# image_normalization_webcam.py
# 2024

# Got this quick demo idea while exploring and understanding different layers of a GAN image-image transformation which uses torchvision
# Press Q or do Ctrl+C in the terminal to interrupt the program btw

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

class Options:
    img_height = 256  # Set resize transform height
    img_width = 256   # width

opt = Options()

transforms_ = transforms.Compose([
    transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
    transforms.ToTensor(),
])

# Default Normalization parameters
# You can play with these for colorful effects
mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)
# std = (0.1, 0.2, 0.1)

# capture video from the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

i = 0
held_frame = None
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    if i == 0:
        held_frame = frame
    i += 1
    print(f"Frame count: {i}")

    # Optionally set a frame to hold which gets the normalization effect applied over and over
    if i % 1 != 0:
        frame = held_frame
    else: 
        held_frame = frame

    # Convert the frame from BGR (OpenCV format) to RGB (PIL format)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert the frame to a PIL Image
    pil_image = Image.fromarray(frame_rgb)

    # Apply the transformations
    normalized_tensor = transforms_(pil_image)

    # Normalize the tensor three times consecutively
    # Set this to any number. Above 10 not recommended. Your screen is probably not wide enough
    prevs = []
    for _ in range(4):
        normalized_tensor = (normalized_tensor - torch.tensor(mean).view(3, 1, 1)) / torch.tensor(std).view(3, 1, 1)
        prevs.append(normalized_tensor)
    
    # convert the normalized tensor back to a numpy array for visualization
    normalized_image = normalized_tensor.permute(1, 2, 0).numpy()  # change shape from (C, H, W) to (H, W, C)
    normalized_image = (normalized_image * 0.5 + 0.5).clip(0, 1)  # unnormalize to [0, 1]

    # scale to [0, 255] for display purposes
    normalized_image = (normalized_image * 255).astype(np.uint8)

    # perform the previous two operations to all of the normalization steps recorded in prevs[] for visualization
    prevs = [(p.permute(1, 2, 0).numpy() * 0.5 + 0.5).clip(0, 1) for p in prevs]
    
    # resize the original frame to match the Resize op
    frame_resized = cv2.resize(frame, (opt.img_width, opt.img_height))

    # Display the original and normalized frames
    combined_frame = np.hstack([frame_resized, normalized_image]+prevs) 
    cv2.imshow('Original and Normalized Frames', combined_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
