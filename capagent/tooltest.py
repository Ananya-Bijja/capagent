import PIL
from capagent.tools import *


def test_count_words():
    result = count_words("This close-up photograph captures the intriguing structure of an Echinops Bannaticus 'Blue Glow' flower standing tall against a backdrop of blurred flowers and a clear blue sky. The spiky, almost spherical bloom glows in shades of light purple-blue, with curled tips of deep brown adding contrast. The stem, with its pale tan fuzziness, provides a soft counterpoint to the flower’s sharp extensions.")
    print("The number of words in the sentence is:", result)

def test_shorten_caption():
    result = shorten_caption("A man is playing with a dog in the park. The dog is a golden retriever.", 5)

def test_single_image_visual_question_answering():
    image = PIL.Image.open("assets/trump_vs_harris.png").convert("RGB")
    result = single_image_visual_question_answering("What is this image?", image)
    count = count_words(result)
    print(result)
    print("The number of words in the caption is:", count)

def test_detail_caption():
    image = PIL.Image.open("assets/cat.png").convert("RGB")
    result = detail_caption(image, "The image shows a close-up of an orange tabby cat.", 3)
    print(result)
    count = count_words(result)
    print("The number of words in the detailed caption is:", count)

def test_search_image_on_web():
    image = ImageData(image=None, image_url="http://367469ar22lb.vicp.fun/.tmp/image_1.png")
    result = search_image_on_web(image)
    print(result)


if __name__ == "__main__":
    # test_count_words()
    # test_shorten_caption()
    # test_coarse_caption()s
    # test_single_image_visual_question_answering()
    # test_detail_caption()
    test_search_image_on_web()


