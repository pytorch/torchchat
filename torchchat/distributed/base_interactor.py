# main class for handling incoming requests and generating responses

import time

import torch
from request_wrapper import Request
from torchchat.distributed import UniqueCounter
from torchchat.distributed.logging_utils import SingletonLogger

logger = SingletonLogger.get_logger()


class LLMInteractor:
    def __init__(self, tokenizer):
        self.tokenizer = None
        self.model = None
        self.requests_counter = UniqueCounter()
        self.new_requests: deque[Request] = deque()
        self.requests_manager = None
        self.eos_id = self.tokenizer.eos_token_id

    def add_request(self, prompt):
        """Add a new request to the queue.  These are then processed by the requests manager"""
        arrival_time = time.monotonic()
        req_id = next(self.requests_counter)
        prompt_tokens = self.tokenizer.encode(prompt)

        self.requests.append(request)
