import PIL
from capagent.tools import *
from run import InstructionAugmenter


def test_count_words():
    result = count_words("The image captures the majestic coronation of King Charles III at Westminster Abbey, where he is seated on a grand throne adorned in a splendid gold robe, crowned with the illustrious St Edward's Crown. Holding the Sovereign's Scepter and orb, he embodies traditional royal authority. Surrounding him are clergy and dignitaries in ceremonial robes and military uniforms, underscoring the solemnity of the occasion. The regal ambiance is highlighted by richly decorated surroundings, vibrant colors, and intricate patterns, as distinguished guests in formal attire witness this historic and grandiose ceremony, steeped in tradition and grandeur.")
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


def test_instruction_augmenter():
    instruction_augmenter = InstructionAugmenter()
    image = PIL.Image.open("data/cia_examples/0.png").convert("RGB")
    result = instruction_augmenter.generate_complex_instruction(image, "Please describe this image within 100 words.")
    print(result)


if __name__ == "__main__":
    # test_count_words()
    # test_shorten_caption()
    # test_coarse_caption()s
    # test_single_image_visual_question_answering()
    # test_detail_caption()
    # test_search_image_on_web()
    test_instruction_augmenter()

