from transformers import FuyuProcessor, FuyuForCausalLM
from PIL import Image
import requests

# Load model and processor
model_id = "adept/fuyu-8b"
processor = FuyuProcessor.from_pretrained(model_id)
model = FuyuForCausalLM.from_pretrained(model_id, device_map="cpu")
model.eval()  # Set the model to evaluation mode

def generate_text(image_url):
    # Load and process the image
    response = requests.get(image_url, stream=True)
    image = Image.open(response.raw)

    image = image.resize((256, 256))

    # Process a single image - ensuring it's in the correct format
    text_prompt = "Generate a coco-style caption.\n"
    inputs = processor(text=text_prompt, images=image, return_tensors="pt").to("cpu")

    # Generate text
    generation_output = model.generate(**inputs, max_new_tokens=7)
    generation_text = processor.batch_decode(generation_output[:, -7:], skip_special_tokens=True)

    return generation_text
