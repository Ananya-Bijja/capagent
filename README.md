# CapAgent

CapAgent is a tool-using agent for image captioning. It supports region captioning, captioning with sentiment, lengthening or shortening the caption, and captioning with more informative web entities. 

<div align="center">
<img src="assets/readme/method.png"/>
</div>

## Prepare environment

### Set SERP API Key
```bash
export SERP_API_KEY=<your-serp-api-key>
```

### Set OpenAI API Key
```bash
export OPENAI_API_KEY=<your-openai-api-key>
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Generate tool usage examples embedding
```bash
bash init_rag_database.sh
```

### Deploy visual tools
```bash
bash deploy_visual_tools.sh
```

## Run CapAgent

```bash
python run.py
```


## Customize tools in CapAgent
### Add a new tool





