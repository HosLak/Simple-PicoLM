from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders
from tqdm import tqdm


def train_tokenizer(data):
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    special_tokens = ["<pad>", "<unk>", "<story_start>", "<story_end>"]

    trainer = trainers.BpeTrainer(
        vocab_size=24576,
        min_frequency=2,
        special_tokens=special_tokens
    )

    tokenizer.train_from_iterator(data['text'], trainer=trainer)

    tokenizer.post_processor = processors.TemplateProcessing(
        single="<story_start> $A <story_end>",
        pair="<story_start> $A <story_end> $B <story_end>",
        special_tokens=[
            ("<pad>", tokenizer.token_to_id("<pad>")),
            ("<unk>", tokenizer.token_to_id("<unk>")),
            ("<story_start>", tokenizer.token_to_id("<story_start>")),
            ("<story_end>", tokenizer.token_to_id("<story_end>")),
        ],
    )

    tokenizer.enable_padding(direction="right", pad_token="<pad>")
    tokenizer.enable_truncation(max_length=2048, direction="right")

    tokenizer.save("tokenizer.json")


if __name__ == "__main__":
    data = None
    train_tokenizer(data)
