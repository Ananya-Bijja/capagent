import PIL
import gradio as gr
import tempfile
import os

from run import run_agent, InstructionComplexer

instruction_complexer = InstructionComplexer()

EXAMPLES = [
    # example 1
    [
"""Create a detailed description of the image, focusing on the central figure seated on an ornate throne, wearing an elaborate crown and regal robes. Highlight the presence of clergy in ceremonial attire standing nearby, emphasizing their roles in the event. Note the richly decorated surroundings, including the vibrant colors and intricate patterns. Mention the distinguished guests in formal attire seated in the background, adding context to the ceremonial setting. The description should be informative and concise, around 100 words, with a formal and respectful tone.

Constraints:
Semantic Constraints: Convey a sense of grandeur and tradition.
Format Constraints: Provide a single, structured paragraph. Aim for around 100 words.
Content Constraints: Focus on the central figure, clergy, and ceremonial elements.
Avoid unrelated or speculative details.
Search Constraints: This seems to be a special moment in history, please search it on the web.""",
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
        "assets/figs/funny_horse.png"
    ],

    # example 4
    [
        "Captioning this image with a sad tone.", 
        "assets/figs/sad_person.png"
    ],

    # example 5
    [
        "Captioning this news photo.", 
        "assets/figs/trump_assassination.png"
    ],
]



def generate_complex_instruction(query: str, image: PIL.Image.Image):
    return instruction_complexer.generate_complex_instruction(image, query)


def process_query(query: str, image: PIL.Image.Image, complex_instruction: str) -> str:
    try:
        # Create temporary directory for image processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded images to temp directory if any
            
            image_path = os.path.join(temp_dir, f"image.png")
            image.save(image_path)
            image_paths = [image_path]
            
            # Run the agent
            if complex_instruction != "":   
                result, messages = run_agent(
                    user_query=complex_instruction, 
                    working_dir=temp_dir, 
                    image_paths=image_paths
                )
            else:
                result, messages = run_agent(
                    user_query=query, 
                    working_dir=temp_dir, image_paths=image_paths
                )

            # for message in messages:
            #     message['content'] = gr.Markdown(value=message['content'])

            return result, messages
        
    except Exception as e:
        return f"Error occurred: {str(e)}", []


def launch_gradio_demo():
    # Create the Gradio interface

    with gr.Blocks() as demo:
        gr.Markdown("# CapAgent")
        gr.Markdown("CapAgent is a tool-using agent for image captioning. It supports region captioning, captioning with sentiment, lengthening or shortening the caption, and captioning with more informative web entities. ")
        gr.Markdown("## Usage")
        gr.Markdown("1. Enter your query and upload image to interact with the CapAgent.")
        gr.Markdown("2. The agent will generate a caption for the image based on your query.")

        with gr.Row():
            
            with gr.Column():
                image_input = gr.Image(height=256, image_mode="RGB", type="pil", label="Image")
                query_input = gr.Textbox(label="User Query", placeholder="e.g., 'Captioning an image with more accurate event information'", lines=2)
                with gr.Blocks():
                    complex_instruction_input = gr.Textbox(label="Complex Instruction")
                    complex_button = gr.Button("Generate Complex Instruction")
                
                with gr.Row():
                    run_button = gr.Button("Run")
                    clear_button = gr.Button("Clear")

                gr.Examples(
                    examples=EXAMPLES,
                    inputs=[query_input, image_input],
                )

            with gr.Column():
                output_textbox = gr.Textbox(label="Agent Response", lines=10)
                cot_textbox = gr.Chatbot(label="Chain of Thought Messages", type='messages', min_height=600)

        complex_button.click(
            generate_complex_instruction, 
            inputs=[query_input, image_input], 
            outputs=complex_instruction_input
        )

        run_button.click(
            process_query, 
            inputs=[query_input, image_input, complex_instruction_input], 
            outputs=[output_textbox, cot_textbox]
        )

        clear_button.click(lambda: [None, None, None, None, None], outputs=[output_textbox, cot_textbox, complex_instruction_input, image_input, query_input])
        

    
    # Launch the demo
    demo.launch(
        share=False,                    # Create a public link
        server_name="10.112.104.168",       # Make available on all network interfaces
        server_port=7861,                  # Default Gradio port,
        debug=True
    )

if __name__ == "__main__":
    launch_gradio_demo()