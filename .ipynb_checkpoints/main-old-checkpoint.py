import time 
from model import generate_text
from PIL import Image
import requests

start = time.time()

def run_inference(image_url):
    image = Image.open(requests.get(image_url, stream=True).raw)
    return generate_text(image)

# Example usage
url = "https://as2.ftcdn.net/v2/jpg/01/42/21/91/1000_F_142219194_fwzSiS0dkWeUkmh4uIxA1J1nYoetmayI.jpg"
print(run_inference(URL))

end = time.time()
print(end - start)