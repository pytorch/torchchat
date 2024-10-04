# main class for handling incoming requests and generating responses

import time

import torch
from torchchat.distributed.logging_utils import SingletonLogger
from torchchat.distributed.request_wrapper import Request
from torchchat.distributed.utils import UniqueCounter

logger = SingletonLogger.get_logger()
from collections import deque


class LLMInteractor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.model = None
        self.requests_counter = UniqueCounter()
        self.requests: deque[Request] = deque()
        self.requests_manager = None
        self.eos_id = self.tokenizer.eos_id()

    def add_request(self, request):
        """Add a new request to the queue.  These are then processed by the requests manager"""
        arrival_time = time.monotonic()

        req_id = next(self.requests_counter)
        prompt_tokens = self.tokenizer.encode(request)
        new_request = Request(arrival_time, request, prompt_tokens, req_id)
        self.requests.append(new_request)
        logger.info(
            f"Added request {req_id} with {len(prompt_tokens)} tokens at {arrival_time}"
        )
