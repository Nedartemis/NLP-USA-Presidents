import tiktoken


class MyGPTTokenizer():
    def __init__(self) -> None:
        self.enc = tiktoken.encoding_for_model("gpt-4")

    def tokenize(self, doc: str) -> list[str]:
        """
        tokenize a string using gpt-4 tokenizer
        :param doc: the string to tokenize
        :return: a list of string (tokenized string)
        """
        tokens = self.enc.encode(doc)
        return [str(token) for token in tokens]
    
    def detokenize(self, generated_text: list[str]) -> str:
        """
        detokenize a tokenized string using gpt-4 detokenizer
        :param generated_text: the string to detokenize
        :return: a detokenized string
        """
        result = []
        for el in generated_text:
            try:
                result.append(int(el))
            except ValueError:
                continue
        return self.enc.decode(result)
