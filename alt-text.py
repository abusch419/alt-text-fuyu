import time

from transformers import FuyuProcessor, FuyuForCausalLM
from PIL import Image
import requests

start_time = time.time()

# load model and processor
model_id = "adept/fuyu-8b"
processor = FuyuProcessor.from_pretrained(model_id)
model = FuyuForCausalLM.from_pretrained(model_id, device_map="cpu")

# prepare inputs for the model
text_prompt = "Generate a coco-style caption.\n"
# url = "https://huggingface.co/adept/fuyu-8b/resolve/main/bus.png"
url = "https://as2.ftcdn.net/v2/jpg/01/42/21/91/1000_F_142219194_fwzSiS0dkWeUkmh4uIxA1J1nYoetmayI.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=text_prompt, images=image, return_tensors="pt").to("cpu")

# autoregressively generate text
generation_output = model.generate(**inputs, max_new_tokens=7)
generation_text = processor.batch_decode(generation_output[:, -7:], skip_special_tokens=True)
print("generation_text", generation_text)

end_time = time.time()
print("Execution time:", end_time - start_time, "seconds")