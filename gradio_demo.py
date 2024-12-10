import PIL
import gradio as gr
import tempfile
import os

from gradio_toggle import Toggle
from capagent.instruction_augmenter import InstructionAugmenter
from capagent.tools import count_words
from run import run_agent



instruction_augmenter = InstructionAugmenter()

EXAMPLES = [
    # example 1
    [
        "Create a detailed description of the image, focusing on the central figure seated on an ornate throne.",
        "assets/figs/charles_on_the_throne.png"
    ],
    # example 2
    [
        "Captioning this image no more than 10 words.", 
        "assets/figs/cat.png"
    ],

    # example 3
    [
        "Captioning this image in a funny tone.", 
        "assets/figs/funny_cat.png"
    ],

    # example 4
    [
        "Captioning this image with a sad tone and no more than three sentences.", 
        "assets/figs/sad_person.png"
    ],

    # example 5
    [
        "Captioning this news photo.", 
        "assets/figs/trump_assassination.png"
    ],
    
    # example 6
    [
        f"Please describe the image within 30 words.", 
        "assets/figs/statue_of_liberty.png"
    ],

    # example 7
    [
        f"Please describe this cab.", 
        "assets/figs/cybercab.png"
    ],

    # example 8
    [
        "Please describe this image.", 
        "assets/figs/venom.png"
    ],

    # example 9
    [
        "Please describe the spatial relationship in this image.", 
        "assets/figs/living_room.png"
    ]
]



def generate_complex_instruction(query: str, image: PIL.Image.Image, is_search: bool):
    try:
        return instruction_augmenter.generate_complex_instruction(image, query, is_search=is_search, timeout=20)
    except Exception as e:
        return f"Timeout. Please try again."


def process_query(query: str, image: PIL.Image.Image) -> str:
    try:
        # Create temporary directory for image processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded images to temp directory if any
            
            image_path = os.path.join(temp_dir, f"image.png")
            image.save(image_path)
            image_paths = [image_path]
            
            result, messages = run_agent(
                user_query=query, 
                working_dir=temp_dir, 
                image_paths=image_paths
            )

            return result, messages
        
    except Exception as e:
        return f"Error occurred: {str(e)}", []


def launch_gradio_demo():
    # Create the Gradio interface

    with gr.Blocks() as demo:
        gr.Markdown("<h1><a href='https://github.com/xin-ran-w/CapAgent'>CapAgent</a></h1>")
        gr.Markdown("CapAgent is a tool-using agent for image captioning. It can generate professional instructions for image captioning, and use tools to generate more accurate captions.")
        gr.Markdown("## Usage")
        gr.Markdown("1. Enter your simple instruction and upload image to interact with the CapAgent.")
        gr.Markdown("2. Click the button 'Generate Professional Instruction' to generate a professional instruction based on your instruction.")
        gr.Markdown("3. Click the button 'Send' to generate a caption for the image based on your professional instruction.")
        with gr.Row():
            
            with gr.Column():
                image_input = gr.Image(height=256, image_mode="RGB", type="pil", label="Image")
                query_input = gr.Textbox(label="User Instruction", placeholder="e.g., 'Captioning an image with more accurate event information'", lines=2, submit_btn="Send")
                
                with gr.Blocks():
                    pro_instruction_input = gr.Textbox(label="Professional Instruction", submit_btn="Send")

                web_search_toggle = Toggle(
                    label="Use Google Search and Google Lens",
                    value=False,
                    color="green",
                    interactive=True,
                )
            
                with gr.Row():
                    complex_button = gr.Button("Generate Professional Instruction")
                    clear_button = gr.Button("Clear")

                gr.Examples(
                    examples=EXAMPLES,
                    inputs=[query_input, image_input],
                )

            with gr.Column():
                output_textbox = gr.Textbox(label="Agent Response", lines=10)
                cot_textbox = gr.Chatbot(label="Chain of Thought Messages", type='messages', min_height=600)

        gr.Markdown("## Contact")
        gr.Markdown("If you have any questions or suggestions, please contact me at <a href='mailto:wangxr@bupt.edu.cn'>wangxr@bupt.edu.cn</a>.")

        complex_button.click(
            generate_complex_instruction, 
            inputs=[query_input, image_input, web_search_toggle], 
            outputs=pro_instruction_input
        )

        pro_instruction_input.submit(
            process_query, 
            inputs=[pro_instruction_input, image_input], 
            outputs=[output_textbox, cot_textbox]
        )

        query_input.submit(
            process_query, 
            inputs=[query_input, image_input], 
            outputs=[output_textbox, cot_textbox]
        )

        clear_button.click(lambda: [None, None, None, None, None], outputs=[output_textbox, cot_textbox, pro_instruction_input, image_input, query_input])

        output_textbox.change(
            lambda x: gr.update(label=f"Agent Response {count_words(x)} words" if x else "Agent Response"), 
            inputs=output_textbox, 
            outputs=output_textbox
        )
    
    
    # Launch the demo
    demo.launch(
        share=False,                    # Create a public link
        server_name="10.112.104.168",       # Make available on all network interfaces
        server_port=7861,                  # Default Gradio port,
        debug=True
    )

if __name__ == "__main__":
    launch_gradio_demo()