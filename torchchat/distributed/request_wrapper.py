# wrap an incoming prompt
import torch


class Request:
    def __init__(self, arrival_time, prompt, prompt_tokens=None, req_id=None):
        self.prompt = prompt
        self.arrival_time = arrival_time
        self.prompt_tokens = None
        self.req_id = req_id
