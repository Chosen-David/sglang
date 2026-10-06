import torch
from typing import List, Optional, Tuple

class BaseGenerator:
    
    def generate(
        self,
        prompts: List[str],
        tokenizer,
        max_new_tokens: Optional[int] = None,
        temperature: float = 1.0,
        do_sample: bool = False
    ) -> List[str]:
        raise NotImplementedError
