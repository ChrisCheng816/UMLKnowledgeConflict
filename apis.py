import os


class API_KEYS:
    def __init__(self):
        self.api_keys = {}
        self.api_keys["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "<ENTER_API_KEY_HERE>")
        self.api_keys["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY", "<ENTER_API_KEY_HERE>")
        self.api_keys["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY", "<ENTER_API_KEY_HERE>")
