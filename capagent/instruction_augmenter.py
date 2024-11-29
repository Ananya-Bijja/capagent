from capagent.chat_models.client import mllm_client
from capagent.utils import encode_pil_to_base64
from capagent.tool_prompt import extract_tool_prompt
from capagent.tools import google_search, google_lens_search, ImageData
from PIL import Image



INSTRUCTION_AUGMENTATION_SYSTEM_MESSAGE = f"""You are an intelligent assistant that generates professional caption instructions based on a given image's visual content and a simple user input. Your task is to analyze the visual content of the image and transform the user's simple instruction into a detailed, professional caption instruction. After that you should generate a tool using insruction step by step reasoning for this instruction.

You can add one of the following constrains to the instruction if you want. Here is some constraint dimension you can refer to:
1. Keywords or phrases: You can add suitable keywords or phrases to the instruction according to the image and original description. E.g., please include the words: "Boeing 737", "a long wings" in the description.
2. Sentiment: You can add suitable sentiment constraints to the instruction according to the image and original description. E.g., describe the image with a happy sentiment.
3. Length: You can add length constraints to the instruction and generate the corresponding description. E.g., using 10 words to describe the image.
4. Focus content: You can add focus content constraints to the instruction according to the image and original description. E.g., focus on the material of the vase.
5. Format: You can add format constraints to the instruction according to the image and original description. E.g., describe the image and use bullet points / markdown / html to format the description.
6. Viewpoint: You can add viewpoint constraints to the instruction according to the image and original description. E.g., describe the image from the middle person's perspective.
7. Genre: You can add genre constraints to the instruction according to the image and original description. E.g., describe the image in the style of a children's book; Describe the image in the style of a poem; Describe the image in the style of a news report; Describe the image in the style of a travel blog post; 

After generating the instruction, you should generate a tool using insruction step by step reasoning for this instruction. The planning will help CapAgent to understand the user request and using the tool correctly.
Here are the tool you can incorporate into the planning:
{extract_tool_prompt("capagent/tools.py")}
follow the format:

Step 1: You need to use "<tool_name>" to search image on the web
Step 2: You need to use "<tool_name>" to ...
...


NOTE: 
- Ensure you incorporate essential constraints from the original user instruction. 
- Adapt the instruction to the given visual content, user intent, and image characteristics.
- You should design a suitable format for the caption, according to other constraints and visual content to improve the readability of the caption.
- The professional instruction should be start with "Please describe the image according to the following instructions:", then format each constraint in a new line.
- After generating the instruction, you should generate a tool planning for this instruction.
- Directly output the instruction and tool planning without any other words.

"""


SEARCH_ASSISTANT_SYSTEM_MESSAGE = f"""You are an intelligent assistant that can search on the web. 
The user will provide you an image and image search result on the web.
You need to generate a keywords list for further search on the web. Such information will be used to generate a more accurate instruction to guide the image captioning.
"""


class InstructionAugmenter:

    EXAMPLES = [
        {
            "role": "user", "content": 
            [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_pil_to_base64(Image.open('data/cia_examples/0.png').convert('RGB'))}"
                    }
                },
                {
                    "type": "text",
                    "text": "User instruction: Please describe the image within 100 words. \nPlease generate a professional instruction based on user instruction. Directly output the instruction without any other words."
                }
            ]
        },
        {"role": "assistant", "content": open("data/cia_examples/0.txt", "r").read()},
    ]


    def generate_complex_instruction(self, image, query: str, is_search: bool, timeout=20):

        # TODO: add search information to the instruction

        messages = [
            {"role": "system", "content": INSTRUCTION_AUGMENTATION_SYSTEM_MESSAGE}
        ]

        if is_search:

            search_messages = [
                {"role": "system", "content": SEARCH_ASSISTANT_SYSTEM_MESSAGE}
            ]
            image.save(".tmp/search_image.png")
            image_data = ImageData(image, image_url=f"http://367469ar22lb.vicp.fun/.tmp/search_image.png", local_path=f".tmp/search_image.png")
            
            image_search_result = google_lens_search(image_data)

            search_messages += [
                {
                    "role": "user", "content": [
                        {
                            "type": "image_url",
                            'image_url': {
                                'url': f"data:image/jpeg;base64,{encode_pil_to_base64(image)}"
                            }
                        },
                        {
                            'type': 'text', 
                            'text': f"Here is the image search result:\n{image_search_result}, now please output the no more than 5 keywords or phrases need to be further on google search. Directly output the keywords or phrases without any other words, each keyword or phrase separate by comma."
                        },
                    ]
                }
            ]

            search_keywords = mllm_client.chat_completion(search_messages, timeout=timeout)
            print(f"search_keywords: {search_keywords}")
            keywords_search_result = []
            for keyword in search_keywords.split(","):
                keywords_search_result.append(google_search(keyword))
            keywords_search_result = "\n".join(keywords_search_result)

            
            messages += [
                {
                    "role": "user", "content": [
                        {
                            "type": "text", "text": f"Here is the similar image title search result returned by google lens: {image_search_result}\n\nHere is the search result for the keywords in the similar image titles:\n{keywords_search_result}.\n\nNow please generate a professional instruction based on user instruction and the search result. Directly output the instruction without any other words."
                        }
                    ]
                }
            ]

        else:
        
            messages += [   
                {
                    "role": "user", "content": [
                        {
                            "type": "image_url",
                            'image_url': {
                                'url': f"data:image/jpeg;base64,{encode_pil_to_base64(image)}"
                            }
                        },
                        {
                            'type': 'text', 
                            'text': f"User instruction: {query}. \nPlease generate a professional instruction based on user instruction. Directly output the instruction without any other words."
                        }
                    ]
                }
            ]

        return mllm_client.chat_completion(messages, timeout=timeout)


if __name__ == "__main__":
    ia = InstructionAugmenter()
    print(ia.generate_complex_instruction(Image.open("assets/figs/cybercab.png").convert("RGB"), "Please describe the image.", is_search=True))
