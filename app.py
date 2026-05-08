import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import matplotlib.pyplot as plt

# Load model
model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
)

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)

print("Device:", device)

# Load image from file
init_image = Image.open("skinny.jpg").convert("RGB")
init_image = init_image.resize((512, 512))

# Prompt (CHANGE THIS)
prompt = "a highly muscular athletic physique, realistic human body, fitness model"

# Generate image
output = pipe(
    prompt=prompt,
    image=init_image,
    strength=0.65,
    guidance_scale=7.5
).images[0]

# Show result
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(init_image)
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Transformed")
plt.imshow(output)
plt.axis("off")

plt.show()

# Save output
output.save("result.png")

print("Done! Image saved as result.png")