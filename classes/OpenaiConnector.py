import openai
from openai.embeddings_utils import get_embedding
import tiktoken
from tqdm.notebook import tqdm
import numpy as np

class OpenaiConnector:
    def __init__(self, api_key:str, instruction_model:str, embed_model:str, chat_model:str):
        """
        Constructor to initialize the connection with OpenAI using the provided API key.

        Args:
        - api_key (str): The API key for authenticating with OpenAI.
        - instruction_model (str): Name of the instruction model to be used.
        - embed_model (str): Name of the model to be used for generating embeddings.
        - chat_model (str): Name of the chat model to be used.
        """
        self.api_key = api_key
        openai.api_key = self.api_key
        self.instruction_model = instruction_model
        self.embed_model = embed_model
        self.chat_model = chat_model
        self.instruction_tokenizer = tiktoken.encoding_for_model(self.instruction_model)
        self.chat_tokenizer = tiktoken.encoding_for_model(self.chat_model)
        
    def get_instructor_model_response(self, prompt:str, max_response_tokens:int,
                                      token_limit:int=2048, num_responses:int=5,
                                      temperature:float=1):
        """
        Generates responses based on the provided prompt using the instruction model.
        
        Args:
        - prompt (str): The initial input text to which the model will respond.
        - max_response_tokens (int): Maximum number of tokens in the model's response.
        - token_limit (int, optional): Maximum allowed tokens for the prompt. Defaults to 2048.
        - num_responses (int, optional): Number of different responses to be returned. Defaults to 5.
        - temperature (float, optional): Sampling temperature. Higher values (e.g., 1.0) make the output more random, while lower values make it more deterministic. Defaults to 1.
        
        Returns:
        - list[str]: List of model's responses based on the prompt.
        
        Raises:
        - ValueError: If the constructed prompt exceeds the maximum token limit.
        """
        if len(list(self.instruction_tokenizer.encode(prompt))) > token_limit:
            raise ValueError("The constructed prompt exceeds the maximum token limit.")
        response = openai.Completion.create(
            engine=self.instruction_model,
            prompt=prompt,
            max_tokens=max_response_tokens,  
            n=num_responses ,
            temperature=temperature
            )
        queries = [item['text'].strip() for item in response.choices]
        return queries
        
    def get_chat_model_response(self, messages, max_response_tokens,
                                token_limit=2048, num_responses=5, temperature=1):
        """
        Generates responses to the provided messages using the chat model.
        
        Args:
        - messages (list): List of message objects. Each object should have a 'role' (either 'system', 'user', or 'assistant') and 'content' (the content of the message from that role).
        - max_response_tokens (int): Maximum number of tokens in the model's response.
        - token_limit (int, optional): Maximum allowed tokens for the messages. Defaults to 2048.
        - num_responses (int, optional): Number of different responses to be returned. Defaults to 5.
        - temperature (float, optional): Sampling temperature. Higher values (e.g., 1.0) make the output more random, while lower values make it more deterministic. Defaults to 1.
        
        Returns:
        - list[str]: List of model's responses to the messages.
        """
        response = openai.ChatCompletion.create(
            model=self.chat_model,
            messages=messages,
            max_tokens=10,
            n=num_responses,
            temperature=temperature
        )
        model_response = [item['message']['content'] for item in response.choices]
        return model_response
        
    def get_text_embedding(self, text):
        """
        Generates an embedding for the provided text using the specified embedding model.
        
        Args:
        - text (str): Text for which the embedding needs to be generated.
        
        Returns:
        - np.ndarray: Array containing the generated embedding for the text.
        """
        text = text.replace("\n", " ")
        return get_embedding(text, engine=self.embed_model)