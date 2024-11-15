import os
import PIL

from nltk.tokenize import word_tokenize
from serpapi import GoogleSearch

from capagent.prompt import (
    shorten_caption_examples, 
    transfer_caption_sentiment_examples,
    asking_a_question_to_detail_caption_examples
)
from capagent.chat_models.client import llm_client, mllm_client
from capagent.utils import encode_pil_to_base64



class ImageData:
    """
    A class to store the image and its URL for temporary use.
    """

    def __init__(self, image: PIL.Image.Image, image_url: str):
        """
        Args:
            image (PIL.Image.Image): The image to store
            image_url (str): The image URL to store
        """
        self.image = image
        self.image_url = image_url



def single_image_visual_question_answering(query: str, image_data: ImageData) -> str:
    """
    Answer a question about the image.
    
    Args:
        query (str): The question to answer
        image_data (ImageData): The image data to answer the question about
        
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
    return result
    

def count_words(sentence: str) -> int:
    """
    Count the number of words in the input string.
    
    Args:
        sentence (str): The input string to count the words
        
    Returns:
        int: The number of words in the input string
    """
    return len(word_tokenize(sentence))


def shorten_caption(caption: str, max_len: int) -> str:
    """
    Shorten the caption within the max length while maintaining key information.
    
    Args:
        caption (str): The original caption text to be shortened
        max_len (int): Maximum number of words allowed in the shortened caption
        
    Returns:
        str: A shortened version of the input caption that respects the word limit
        
    Note:
        This function uses an LLM (Llama3) to intelligently shorten the caption while
        preserving the most important information. It formats the input as a chat message
        and uses predefined examples (shorten_caption_examples) as context for the model.
    """
    user_input = {"role": "user", "content": f"Caption: {caption}. Max length: {max_len} words."}
    messages = shorten_caption_examples + [user_input]
    result  = llm_client.chat_completion(messages)   

    return result

def transfer_caption_sentiment(caption: str, sentiment: str) -> str:
    """
    Transfer the caption to the specified sentiment.
    
    Args:
        caption (str): The original caption text to be transferred
        sentiment (str): The desired sentiment for the caption
        
    Returns:
        str: A caption with the specified sentiment
    """

    user_input = {"role": "user", "content": f"Please transfer the caption to {sentiment} sentiment. Caption: {caption}."}
    messages = transfer_caption_sentiment_examples + [user_input]
    result = llm_client.chat_completion(messages)

    return result

def detail_caption(image_data: ImageData, caption: str, iteration: int) -> str:
    """
    Ask and answer questions to detail the caption. Each iteration using LLM to ask a question and using MLLM to answer it. Finally, LLM according to the questions and answers to detail the caption.

    Args:
        image_data (ImageData): The image data to answer the question about
        caption (str): The caption to detail
        iteration (int): The number of iterations to ask and answer questions
    Returns:
        str: The detailed caption
    """
    user_input = {"role": "user", "content": f"Caption: {caption}."}
    messages = asking_a_question_to_detail_caption_examples + [user_input]
    
    for _ in range(iteration):
        question = llm_client.chat_completion(messages)
        answer = mllm_client.single_image_chat_completion(question, image_data.image)
        messages += [
            {"role": "assistant", "content": f"Question: {question}"}, 
            {"role": "user", "content": f"Answer: {answer}. Please generate a new question."}
        ]
    
    messages += [{"role": "user", "content": f"Please detail the caption according to the questions and answers. Caption: {caption}. Directly output the detailed caption without any other words."}]
    result = llm_client.chat_completion(messages)

    return result

def recaption_image_with_additional_info(caption: str, additional_info: str) -> str:
    """
    Call this function when you need to add additional information to the caption.

    Args:
        caption (str): The original caption of the image
        additional_info (str): The additional information
        
    Returns:
        str: The caption with the additional information
    """
    system_prompt = "You are a helpful assistant that can help users to add additional information to the caption. Please ensure the readability of the output caption."
    user_input = {"role": "user", "content": f"Caption: {caption}. Additional information: {additional_info}. Directly output the caption without any other words."}
    messages = [{"role": "system", "content": system_prompt}, user_input]
    result = llm_client.chat_completion(messages)

    return result

def search_image_on_web(image_data: ImageData) -> str:
    """
    Call this function when you need to search the similar images on the web. 

    Args:
        image_data (ImageData): The image data to search the similar images on the web
        
    Returns:
        str: The search result of the image
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

        top_K = 10
        titles = [v_match["title"] for v_match in visual_matches[:top_K]]
        search_result = "\n".join(titles)

    except Exception as e:
        search_result = "This tool is experiencing problems and is not working properly"

    return search_result
