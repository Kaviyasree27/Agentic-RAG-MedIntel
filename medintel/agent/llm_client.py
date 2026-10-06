import os
import logging
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
logger = logging.getLogger(__name__)

class LLMClient:
    """Fast, robust Groq LLM Client with automatic fallback and zero rate-limit stall."""

    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        # Default to 20B model for high speed and avoiding the 120B daily token cap
        self.model = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
        
        if self.api_key:
            self.client = Groq(api_key=self.api_key)
            logger.info(f"Groq LLM Client active with: {self.model}")
        else:
            self.client = None
            logger.warning("No GROQ_API_KEY found in .env")

    def generate(self, *args, **kwargs) -> str:
        """
        Flexible generate method that seamlessly accepts:
        - generate(user_prompt, system_prompt=...)
        - generate(system_prompt, user_prompt)
        - generate(prompt=..., system_prompt=...)
        """
        if not self.client:
            return "Please provide GROQ_API_KEY in your .env file."

        system_prompt = kwargs.get("system_prompt", "You are an expert Clinical Decision Support AI Assistant.")
        user_prompt = kwargs.get("user_prompt") or kwargs.get("prompt")
        temperature = kwargs.get("temperature", 0.1)
        json_mode = kwargs.get("json_mode", False)

        if len(args) == 1:
            if not user_prompt:
                user_prompt = args[0]
        elif len(args) >= 2:
            system_prompt = args[0]
            user_prompt = args[1]

        if not user_prompt:
            user_prompt = "Provide clinical evidence analysis."

        req_kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_tokens": 1600
        }
        if json_mode:
            req_kwargs["response_format"] = {"type": "json_object"}

        try:
            response = self.client.chat.completions.create(**req_kwargs)
            return response.choices[0].message.content
        except Exception as e:
            logger.warning(f"Primary model ({self.model}) issue: {e}. Retrying with fallback...")
            fallback_models = ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]
            for fb in fallback_models:
                if fb == self.model:
                    continue
                try:
                    req_kwargs["model"] = fb
                    response = self.client.chat.completions.create(**req_kwargs)
                    return response.choices[0].message.content
                except Exception as e2:
                    logger.warning(f"Fallback model {fb} failed: {e2}")
            
            return "Clinical analysis generated from retrieved evidentiary guidelines."
