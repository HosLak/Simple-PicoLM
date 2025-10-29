from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders
from tqdm import tqdm


def train_tokenizer(data):
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    SPECIAL_TOKENS = [
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "system",
        "user",
        "assistant"
    ]

    trainer = trainers.BpeTrainer(
        vocab_size=2**15,
        min_frequency=1,
        special_tokens=SPECIAL_TOKENS
    )

    tokenizer.train_from_iterator(data['text'], trainer=trainer)

    tokenizer.post_processor = processors.TemplateProcessing(
        single="<|im_start|>$A<|im_end|>", 
        pair="<|im_start|>$A<|im_end|>\n<|im_start|>$B<|im_end|>", 
        special_tokens={
            "<|im_start|>": (
                "<|im_start|>",
                {
                    "type": "im_start",
                    "value": "$A", 
                    "lstrip": False,
                    "add_prefix_space": False
                }
            ),
            "<|im_end|>": (
                "<|im_end|>",
                {
                    "type": "im_end",
                    "value": "",
                    "lstrip": True,
                    "add_prefix_space": False
                }
            ),
        },
    )

    chat_template = """{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}""".strip()
    
    tokenizer.chat_template = chat_template
    tokenizer.eos_token = "<|endoftext|>"

    tokenizer.save("tokenizer.json")


if __name__ == "__main__":
    data = None
    train_tokenizer(data)