from capagent.chat_models.client import mllm_client
from capagent.utils import encode_pil_to_base64
from PIL import Image


INSTRUCTION_AUGMENTATION_SYSTEM_MESSAGE = """You are an intelligent assistant that generates professional caption instructions based on a given image's visual content and a simple user input. Your task is to analyze the visual content of the image and transform the user's simple instruction into a detailed, professional caption instruction. The professional instruction should specify constraints on the caption's content, format, style, and additional enhancements, ensuring alignment with the user's intent and the image's characteristics.

Guidelines:

1. Semantic Constraints:
- View: Specify the point of view from which to describe the image (e.g., from the perspective of a central object, a viewer, or a participant in the scene).
- Sentiment: Reflect the emotional tone of the image if it conveys strong sentiments.
- Object Details: Emphasize key details of salient objects in the image.

2. Format Constraints:
- Define caption length (e.g., one sentence, within 50 words).
- Specify structural preferences (e.g., bullet points, numbered lists).

3. Lexical Constraints:
- Ensure inclusion of key terms or keywords as specified.

4. Search Constraints:
- Indicate whether to search the image on the web to gather additional information about events, famous personalities, or landmarks depicted in the image.

NOTE: 
- Ensure you incorporate essential constraints from the original user instruction. 
- Adapt the instruction to the given visual content, user intent, and image characteristics.
- You should design a suitable format for the caption, according to other constraints and visual content to improve the readability of the caption.
- The professional instruction should be start with "Please describe the image according to the following instructions:", then format each constraint in a new line.

"""


SYSTEM_PROMPT_2 = """You are an intelligent assistant that generates professional caption instructions based on a given image's visual content and a simple user input. Your task is to analyze the visual content of the image and transform the user's simple instruction into a detailed, professional caption instruction. The professional instruction should specify constraints on the caption's content, format, style, and additional enhancements, ensuring alignment with the user's intent and the image's characteristics.
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


    def generate_complex_instruction(self, image, query: str, timeout=20):
        
        messages = [
            {"role": "system", "content": INSTRUCTION_AUGMENTATION_SYSTEM_MESSAGE}
        ]
        messages += self.EXAMPLES
        # messages = [
        #     {
        #         "role": "system", "content": SYSTEM_PROMPT_2
        #     }
        # ]
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
    print(ia.generate_complex_instruction(Image.open("data/cia_examples/0.png").convert("RGB"), "Please describe the image."))
