from gradio_client import Client, file
from PIL import Image
import tempfile

def test_detection_client():
    client = Client("http://0.0.0.0:8081")

    image = Image.open("/home/wangxinran/IFAgent/assets/figs/fried_chicken.png")
    with tempfile.NamedTemporaryFile(delete=True, suffix=".jpg") as tmp_file:
        image.save(tmp_file.name, 'JPEG')
        image = tmp_file.name
        result_image_file, result_json = client.predict(file(image), "food", 0.3, 0.3)
        print(result_image_file)
        print(result_json)

def test_depth_client():
    client = Client("http://127.0.0.1:7860")
    image = Image.open("/home/wangxinran/IFAgent/assets/figs/fried_chicken.png")
    with tempfile.NamedTemporaryFile(delete=True, suffix=".jpg") as tmp_file:
        image.save(tmp_file.name, 'JPEG')
        image = tmp_file.name
        _, grayscale_depth_map, _ = client.predict(file(image), api_name="/on_submit")
        print(grayscale_depth_map)

if __name__ == "__main__":
    test_depth_client()

    
