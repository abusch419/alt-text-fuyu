from transformers import FuyuProcessor, FuyuForCausalLM

# Load model and processor
model_id = "adept/fuyu-8b"
processor = FuyuProcessor.from_pretrained(model_id)
model = FuyuForCausalLM.from_pretrained(model_id, device_map="cpu")

def generate_text(image):
    inputs = processor(images=image, return_tensors="pt").to("cpu")
    generation_output = model.generate(**inputs, max_new_tokens=7)
    return processor.batch_decode(generation_output[:, -7:], skip_special_tokens=True)
