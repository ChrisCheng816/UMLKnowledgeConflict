from openai import OpenAI
import base64
import os
from apis import API_KEYS

api_keys = API_KEYS()
def encode_image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        raise FileNotFoundError(f"Image not found at {image_path}. Please check the path.")

        
def get_info(model):
    if 'claude' in model:
        api_key = api_keys.api_keys['ANTHROPIC_API_KEY']
        base_url = "https://api.anthropic.com/v1/"
    elif 'gpt' in model or 'o' in model[0]:
        api_key = api_keys.api_keys['OPENAI_API_KEY']
        base_url = "https://api.openai.com/v1"  
    else:
        api_key = api_keys.api_keys['GEMINI_API_KEY']
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    return api_key, base_url


def run_vqa(model, prompt, image_path):
    api_key,base_url = get_info(model)
    if 'o3' not in model:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )    
    else:
        client = OpenAI(
            api_key=api_key,
        )    
    base64_image = encode_image_to_base64(image_path)
    
    if 'codex' in model:
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{base64_image}"
                        }
                    ]
                }
            ]
        )

        answer = (response.output_text)
    else:
        response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ]
            )

        answer = (response.choices[0].message.content)
    return answer
