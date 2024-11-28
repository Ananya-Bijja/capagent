from capagent.tool_prompt import extract_tool_prompt



ASSISTANT_SYSTEM_MESSAGE = """You are a helpful AI assistant.
Your task is to caption an image according to the users' request. The user may provide an image and a request. The request may contain multiple requirement on the caption, such as the length of caption, the sentiment of the caption. You should meet all requirements in the request. 
There are many tools can assist you to meet the requirements, you can coding to solve the problem. You are coding to use these toolsin a Python jupyter notebook environment.
You can suggest python code (in a python coding block) for the user to execute. In a dialogue, all your codes are executed with the same jupyter kernel, so you can use the variables, working states.. in your earlier code blocks.
Solve the task step by step if you need to. 
The task may be a vision-language task and require several steps. You can write code to process images, text, or other data in a step. Give your code to the user to execute. The user may reply with the text and image outputs of the code execution. You can use the outputs to proceed to the next step, with reasoning, planning, or further coding.
If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your language skill.
When using code, you must indicate the script type in the code block. The user cannot provide any other feedback or perform any other action beyond executing the code you suggest. The user can't modify your code. So do not suggest incomplete code which requires users to modify. Don't use a code block if it's not intended to be executed by the user.
Don't include multiple code blocks in one response. Do not ask users to copy and paste the result. Instead, use 'print' function for the output when relevant. Check the execution result returned by the user.
All images should be stored in PIL Image objects. The notebook has imported 'Image' from 'PIL' package and 'display' from 'IPython.display' package. If you want to read the image outputs of your code, use 'display' function to show the image in the notebook. The user will send the image outputs to you.
If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.

For each turn, you should first do a "THOUGHT", based on the images and text you see.
If you think you get the answer to the intial user request, you can reply with "ANSWER: <your answer>" and ends with "TERMINATE".
"""


class ReActPrompt:
    
    def __init__(self) -> None:
        return

    def initial_prompt(self, query: str, n_images: int, tool_usage_example: str) -> str:

        _init_prompt = f"""Here are some tools that can help you. All are python codes. They are in tools.py and will be imported for you.
        Below are the tools in tools.py:
```python
{extract_tool_prompt("capagent/tools.py")}

```

Below are some examples of how to use the tools to solve the user requests. You can refer to them for help. You can also refer to the tool descriptions for more information.
{tool_usage_example}

"""

        prompt = _init_prompt
        prompt += f"# USER REQUEST #: {query}\n"
        if n_images > 0:
            prompt += f"# USER IMAGE stored in {', '.join([f'image_data_{i}' for i in range(1, n_images+1)])} as ImageData. NOTE: Don't create a new image yourself.\n"
        else:
            prompt += "# USER IMAGE: No image provided.\n"
        prompt += "Now please generate only THOUGHT 0 and ACTION 0 in RESULT. If no action needed, also reply with ANSWER: <your answer> and ends with TERMINATE in the RESULT:\n# RESULT #:\n"
        return prompt
    
    def get_parsing_feedback(self, error_message: str, error_code: str) -> str:
        return f"OBSERVATION: Parsing error. Error code: {error_code}, Error message:\n{error_message}\nPlease fix the error and generate the fixed code, in the next THOUGHT and ACTION."
    
    def get_exec_feedback(self, exit_code: int, output: str) -> str:
        
        # if execution fails
        if exit_code != 0:
           return f"OBSERVATION: Execution error. Exit code: {exit_code}, Output:\n{output}\nPlease fix the error and generate the fixed code, in the next THOUGHT and ACTION."
        else:
            prompt = f"OBSERVATION: Execution success. The output is as follows:\n{output}\n"
            prompt += "Please generate the next THOUGHT and ACTION. If you can get the answer, please also reply with ANSWER: <your answer> and ends with TERMINATE."
            return prompt


transfer_caption_sentiment_system_prompt = """You are a helpful AI assistant. Give you an image caption, your task is to transfer the caption to the specified sentiment.
"""

transfer_caption_sentiment_examples = [
    {"role": "system", "content": transfer_caption_sentiment_system_prompt},
    {"role": "user", "content": "Please transfer the caption to positive sentiment. Caption: A horse with white mane and tail is running in the field."},
    {"role": "assistant", "content": "The majestic horse boasts a stunning white mane and tail as it gallops across the field."}
]

asking_a_question_to_detail_caption_system_prompt = """You are a helpful AI assistant. Give you an image caption, your task is to ask a question to detail the caption.
"""

asking_a_question_to_detail_caption_examples = [
    {"role": "system", "content": asking_a_question_to_detail_caption_system_prompt},
    {"role": "user", "content": "Caption: A beautiful girl is playing with a cat."},
    {"role": "assistant", "content": "What is the girl wearing?"}
]

