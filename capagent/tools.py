import os
import PIL

from nltk.tokenize import word_tokenize
from serpapi import GoogleSearch
from capagent.chat_models.client import llm_client, mllm_client
from capagent.utils import encode_pil_to_base64
from gradio_client import Client, file
from pprint import pprint


try:
    detection_client = Client("http://127.0.0.1:8080")
    print("Detection client is listening on port 8080.")
except Exception as e:
    print("Detection client is not working properly. Tools related to detection will not work.")
    detection_client = None

try:
    depth_client = Client("http://127.0.0.1:8081")
    print("Depth client is listening on port 8081.")
except Exception as e:
    print("Depth client is not working properly. Tools related to depth will not work.")
    depth_client = None


class ImageData:
    """
    A class to store the image and its URL for temporary use.
    """

    def __init__(self, image: PIL.Image.Image, image_url: str, local_path: str):
        """
        Args:
            image (PIL.Image.Image): The image to store
            image_url (str): The image URL to store
        """
        self.image = image
        self.image_url = image_url
        self.local_path = local_path



def visual_question_answering(query: str, image_data: ImageData, show_result: bool = True) -> str:
    """
    Answer a question about the image.
    
    Args:
        query (str): The question to answer
        image_data (ImageData): The image data to answer the question about
        show_result (bool): Whether to print the result
    Returns:
        str: The answer to the question
    """
    messages = [
        {
            "role": "user", 
            "content": [
                {
                    'type': 'image_url', 
                    'image_url': {
                        'url': f"data:image/jpeg;base64,{encode_pil_to_base64(image_data.image)}"
                    }
                },
                {'type': 'text', 'text': query}
            ]
        }
    ]

    result = mllm_client.chat_completion(messages)
    if show_result:
        print(f"Answer to the question: {result}")

    return result
    

def count_words(caption: str, show_result: bool = True) -> int:
    """
    Count the number of words in the input string.
    
    Args:
        caption (str): The input string to count the words
        show_result (bool): Whether to print the result
        
    Returns:
        int: The number of words in the input string
    """
    if show_result:
        print(f"Now the number of words in the caption is: {len(word_tokenize(caption))}.")

    return len(word_tokenize(caption))


def shorten_caption(caption: str, max_len: int, show_result: bool = True) -> str:
    """
    Shorten the caption within the max length while maintaining key information.
    
    Args:
        caption (str): The original caption text to be shortened
        max_len (int): Maximum number of words allowed in the shortened caption
        show_result (bool): Whether to print the result
    
    Returns:
        str: A shortened version of the input caption that respects the word limit
    """

    system_prompt = """You are helpful assistant. You are good at shortening the image caption. Each time the user provides a caption and the max length, you can help to shorten the caption to the max length.

    Note:
    - You should change the length of the caption by first delete unnecessary words or details not mentioned in the user request.
    - You should keep the original sentiment and descriptive perspective of the caption.
    - You should keep the original meaning of the caption.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Caption: {caption}. Max length: {max_len} words. Directly output the shortened caption without any other words."}
    ]
    result = llm_client.chat_completion(messages)
    w_count = count_words(result, show_result=False)

    while w_count > max_len:
        messages += [{"role": "assistant", "content": f"Caption: {result}"}]
        messages += [{"role": "user", "content": f"The length of the caption ({w_count} words) is still longer than the max length ({max_len} words). Please shorten the caption to the max length."}]
        result = llm_client.chat_completion(messages)
        w_count = count_words(result, show_result=False)
    
    if show_result:
        print(f"Shortened caption: {result}")

    return result

def change_caption_sentiment(caption: str, sentiment: str, show_result: bool = True) -> str:
    """
    Transfer the caption to the specified sentiment.
    
    Args:
        caption (str): The original caption text to be transferred
        sentiment (str): The desired sentiment for the caption
        show_result (bool): Whether to print the result
    
    Returns:
        str: The caption with the transferred sentiment

    This function will automatically print the result by setting show_result to True, with the transferred caption and the number of words in the caption.
    """

    user_input = {"role": "user", "content": f"Caption: {caption}. Please change the sentiment of the caption to {sentiment}. Directly output the transferred caption without any other words."}
    messages = [user_input]
    result = llm_client.chat_completion(messages)
    if show_result:
        print(f"Transferred caption: {result}")

    return result

def extend_caption(image_data: ImageData, caption: str, iteration: int, show_result: bool = True) -> str:
    """
    Call this function when you need to extend the caption to include more details. 

    Args:
        image_data (ImageData): The image data to extend the caption
        caption (str): The caption to extend
        iteration (int): The number of iterations to ask and answer questions
        show_result (bool): Whether to print the result
    
    Returns:
        str: The extended caption
        
    This function will automatically print the result by setting show_result to True, with the extended caption and the number of words in the caption.
    """
    user_input = {"role": "user", "content": f"Caption: {caption}."}
    llm_messages = [user_input]
    
    for _ in range(iteration):
        question = llm_client.chat_completion(messages)
        mllm_message = {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_pil_to_base64(image_data.image)}"}},
            {"type": "text", "text": question}
            
        ]}
        answer = mllm_client.chat_completion(mllm_message)
        llm_messages += [
            {"role": "assistant", "content": f"Question: {question}"}, 
            {"role": "user", "content": f"Answer: {answer}. Please generate a new question."}
        ]
    
    messages += [{"role": "user", "content": f"Please extend the caption according to the questions and answers. Caption: {caption}. Directly output the extended caption without any other words."}]
    result = llm_client.chat_completion(messages)
    
    if show_result:
        print(f"Extended caption: {result}.")
        count_words(result, show_result=True)

    return result


def add_keywords_to_caption(caption: str, keywords: list[str], show_result: bool = True) -> str:
    """
    Call this function when you need to add keywords to the caption.

    Args:
        caption (str): The original caption of the image
        keywords (list[str]): The keywords to add to the caption
        show_result (bool): Whether to print the result

    Returns:
        str: The caption with the added keywords
            
    This function will automatically print the result by setting show_result to True, with the added keywords and the number of words in the caption.
    """
    system_prompt = "You are a helpful assistant that can help users to add keywords to the caption. Please ensure the readability of the output caption."
    user_input = {"role": "user", "content": f"Caption: {caption}. Keywords: {keywords}. Directly output the caption without any other words."}
    messages = [{"role": "system", "content": system_prompt}, user_input]
    result = llm_client.chat_completion(messages)

    if show_result:
        print(f"Recaptioned caption: {result}")

    return result

def google_search(query: str, show_result: bool = True, top_k: int = 5) -> str:
    """
    Call this function when you need to search the query on Google.
    
    Args:
        query (str): The query to search
        show_result (bool): Whether to print the result
        top_k (int): The number of results to show
    
    This function will automatically print the search result by setting show_result to True, with the title, snippet, snippet highlighted words, source, and the link of the result.
    """

    params = {
        "q": query,
        "location": "Austin, Texas, United States",
        "hl": "en",
        "gl": "us",
        "google_domain": "google.com",
        "api_key": os.getenv("SERP_API_KEY")
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    organic_results = results.get("organic_results", None)
    
    if results is None and show_result:
        print("No results found")

    print(f"Google Search Result of {query}:")
    for i, result in enumerate(organic_results[:top_k]):
        print("-" * 10 + f"Result {i}" + "-" * 10)
        print(f"Title: {result.get('title', 'N/A')}")
        print(f"Snippet: {result.get('snippet', 'N/A')}")
        print(f"Snippet highlighted words: {result.get('snippet_highlighted_words', 'N/A')}")
        print(f"Source: {result.get('source', 'N/A')}\n")



def google_lens_search(image_data: ImageData, show_result: bool = True, top_k: int = 10) -> str:
    """
    Call this function when you need to search the similar images information on Google Lens. 

    Args:
        image_data (ImageData): The image data to search the similar images on Google Lens
        show_result (bool): Whether to print the result
        top_k (int): The number of results to show

    This function will automatically print the search result by setting show_result to True, with the title of each similar image.
    """

    params = {
        "engine": "google_lens",
        "url": image_data.image_url,
        "api_key": os.getenv("SERP_API_KEY"),
        "hl": "en",
        "country": "US",
        "no_cache": True
    }

    try: 
        search = GoogleSearch(params)
        results = search.get_dict()
        visual_matches = results["visual_matches"]

        titles = [v_match["title"] for v_match in visual_matches[:top_k]]
        if show_result:
            print("Google Lens Image Search Result:")
            for i, title in enumerate(titles):
                print("-" * 10 + f"Result {i}" + "-" * 10)
                print(f"Title: {title}")

    except Exception as e:
        print("This tool is experiencing problems and is not working properly")


def spatial_relation_of_objects(image_data: ImageData, show_result: bool = True, objects: list[str] = None) -> str:
    """
    Call this function when you need to know the spatial relation of the objects in the image.
    """
    
    assert objects is not None, "Objects are not specified."
    objects_caption = ", ".join(objects)
    result_image_file, result_json = detection_client.predict(file(image_data.image), objects_caption, 0.3, 0.3)
    if show_result:
        print(f"Bounding boxes of the objects:\n")
        pprint(result_json)

    llm_messages = [
        {"role": "system", "content": "You are a helpful assistant that can help users to understand the spatial relation of the objects in the image."},
        {"role": "user", "content": f"Bounding boxes of the objects: {result_json}. Please describe the spatial relation of the objects in the image."}
    ]
    result = llm_client.chat_completion(llm_messages)

    if show_result:
        print(f"Spatial relation of the objects: \n{result}")
    

def depth_relation_of_objects(image_data: ImageData, show_result: bool = True, objects: list[str] = None) -> str:
    """
    Call this function when you need to know the depth relation of the objects in the image.
    """

    assert objects is not None, "Objects are not specified."
    objects_caption = ", ".join(objects)
    _, result_json = detection_client.predict(file(image_data.local_path), objects_caption, 0.3, 0.3)
    
    # gain depth map of the image

    _, grayscale_depth_map, _ = depth_client.predict(file(image_data.local_path), api_name="/on_submit")

    # gain depth value of each object, according to the bounding box
    object_depth_values = {
        f"{obj}": grayscale_depth_map[result_json[obj]["y1"]:result_json[obj]["y2"], result_json[obj]["x1"]:result_json[obj]["x2"]].mean() 
        for obj in result_json
    }
    
    llm_messages = [
        {"role": "system", "content": "You are a helpful assistant that can help users to understand the depth relation of the objects in the image."},
        {"role": "user", "content": f"Bounding boxes of the objects: \n{result_json}. Depth values of the objects: \n{object_depth_values}. Please describe the depth relation of the objects in the image."}
    ]
    result = llm_client.chat_completion(llm_messages)

    if show_result:
        print(f"Depth relation of the objects:\n{result}")


def crop_object_region(image_data: ImageData, object: str) -> str:
    """
    Call this function when you need to crop the object region in the image.

    Args:
        image_data (ImageData): The image data to crop the object region
        object (str): The object to crop
    
    Returns:
        PIL.Image.Image: The cropped object region image
    """ 

    _, result_json = detection_client.predict(file(image_data.local_path), object, 0.3, 0.3)  

    result_image = image_data.image.crop(result_json[object])
    
    return result_image
