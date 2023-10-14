import cv2
import os
import numpy as np
import torch
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
from PIL import Image

ControlNetModeluse = "lllyasviel/sd-controlnet-canny"
ControlNetimgtoimgModeluse = "runwayml/stable-diffusion-v1-5"

# Function to process each frame using ControlNet
def process_frame(input_image_path, output_image_path, controlnet, pipe):
    # Load input image
    input_image = load_image(input_image_path)
    np_input_image = np.array(input_image)

    # Get canny image
    np_canny_image = cv2.Canny(np_input_image, 100, 200)
    np_canny_image = np_canny_image[:, :, None]
    np_canny_image = np.concatenate([np_canny_image, np_canny_image, np_canny_image], axis=2)
    canny_image = Image.fromarray(np_canny_image)

    # Process image using ControlNet
    generator = torch.manual_seed(913753751)
    output_image = pipe(
        prompt = "",
        num_inference_steps=20,
        generator=generator,
        image=input_image,
        control_image=canny_image,
    ).images[0]

    # Save the processed image
    output_image.save(output_image_path)

# Path configurations
input_video_path = "/content/4sec.mp4"
output_frames_folder = "frames"
output_processed_frames_folder = "processed_frames"
output_video_path = "output_video.mp4"

# Load ControlNet and Stable Diffusion Pipeline
controlnet = ControlNetModel.from_pretrained(ControlNetModeluse, torch_dtype=torch.float16)

# Mengecek apakah repo_id mengandung ekstensi yang sesuai
if repo_id.endswith((".safetensors", ".ckpt", ".pt")):
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_single_file(ControlNetimgtoimgModeluse,
                                                    torch_dtype=torch.float16,
                                                    use_karras_sigmas=True,
                                                   )
else:
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(ControlNetimgtoimgModeluse,
                                                   torch_dtype=torch.float16,
                                                   use_karras_sigmas=True
                                                  )

pipe.safety_checker = None
pipe.requires_safety_checker = False

# Speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

# Create folders if not exist
os.makedirs(output_frames_folder, exist_ok=True)
os.makedirs(output_processed_frames_folder, exist_ok=True)

# Step 1: Extract frames from video
cap = cv2.VideoCapture(input_video_path)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    frame_path = os.path.join(output_frames_folder, f"frame_{frame_count:04d}.png")
    cv2.imwrite(frame_path, frame)

cap.release()

# Step 2: Process each frame using ControlNet
for i in range(1, frame_count + 1):
    input_frame_path = os.path.join(output_frames_folder, f"frame_{i:04d}.png")
    output_frame_path = os.path.join(output_processed_frames_folder, f"processed_frame_{i:04d}.png")
    process_frame(input_frame_path, output_frame_path, controlnet, pipe)

# Step 3: Combine processed frames into video
img_array = []

for i in range(1, frame_count + 1):
    img = cv2.imread(os.path.join(output_processed_frames_folder, f"processed_frame_{i:04d}.png"))
    img_array.append(img)

height, width, layers = img_array[0].shape
video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

for i in range(len(img_array)):
    video.write(img_array[i])

video.release()
