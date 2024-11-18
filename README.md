# CapAgent

In this section, we introduce the CapAgent, an agent system with a variety of tools specifically designed to control the image captioning process. As shown in the following figure, like a general agent, CapAgent’s workflow includes three main steps: planning, tool usage, and observation. When the user inputs an image and a caption query, the CapAgent will generate a series of thoughts and corresponding actions to tackle the user request.

<div align="center">
<img src="assets/readme/method.png"/>
</div>

## Prepare environment


### Set API Key
```bash
export SERP_API_KEY=<your-serp-api-key> # for search image on web
export OPENAI_API_KEY=<your-openai-api-key> # for using gpt-4o
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Generate CoT examples embedding
```bash
bash init_rag_database.sh
```

### Launch server
To let local image online for allowing api, e.g., google search, using url access the image.
```bash
python launch_image_server.py
```

## Run CapAgent
Run on a single image
```bash
python run.py
```

Run on gradio demo
```bash
python gradio_demo.py
``` 

## Video Demo

Comming soon ...


## Contact
Contact me if you have any questions. Email: wangxr@bupt.edu.cn
