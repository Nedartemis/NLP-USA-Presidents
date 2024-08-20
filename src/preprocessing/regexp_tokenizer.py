import nltk

nltk.download('wordnet')

class RegexpTokenizer:
    def __init__(self) -> None:
        self.wnl = nltk.WordNetLemmatizer()
        self.twt = nltk.TweetTokenizer()

        https = r"\bhttps?:\/\/\S+\b"
        words = r"\b(?:[a-zA-Z](?:'?[a-zA-Z])+)\b"
        abbreviations = r"\b(?:[a-zA-Z].)+"
        numbers = r"\b(?:\d{1,3}(?:,\d{3})*\b|\d+(?:st|rd|th))"
        decimal_numbers = r"\b(?:\d+(?:.\d+)?)\b"
        tweet = r"[@#]\b\w+\b"
        signature = r"[-] ?\b\w+\b$"

        regex_strings = [https, words, abbreviations, numbers, decimal_numbers, tweet, signature]
        self.objNlpTokenizer = nltk.RegexpTokenizer(r"""(%s)""" % "|".join(regex_strings))
        pass

    def lemma_tokenize(self, doc: str) ->list[str]:
        """
        lemma_tokenizer: lemmatize the word tokens
        :param doc: the string to tokenize
        :return: the list of tokens
        """ 
        return [self.wnl.lemmatize(t) for t in nltk.word_tokenize(doc)]

    def word_tokenizer(self, doc: str) ->list[str]:
        """
        word_tokenizer: nltk word tokenizer
        :param doc: the string to tokenize
        :return: the list of tokens
        """ 
        return nltk.word_tokenize(doc)

    def sent_tokenizer(self, doc: str) ->list[str]:
        """
        sent_tokenizer: nltk sentence tokenizer
        :param doc: the string to tokenize
        :return: the list of tokens
        """ 
        return nltk.sent_tokenize(doc)
    
    def tweet_tokenizer(self, doc: str) ->list[str]:
        """
        tweet_tokenizer: nltk tweet tokenizer
        :param doc: the string to tokenize
        :return: the list of tokens
        """ 
        return self.twt.tokenize(doc)
    
    def whitespaces_tokenizer(self, doc: str) ->list[str]:
        """
        whitespaces_tokenizer: split the doc using whitespaces, into tokens 
        :param doc: the string to tokenize
        :return: the list of tokens
        """ 
        return doc.split(' ')
    
    def nlpReg_tokenizer(self, doc: str) -> list[str]:
        # Words with optional internal apostrophes
        # Abbreviations like U.S.A.
        # Numbers with or without commas
        # Decimal numbers
         # Non-alphanumeric characters as tokens
         #also remove - in middle of sentences (and - and)
        """
        nlpReg tokenizer: split the doc using regex defined rules 
        :param doc: the string to tokenize
        :return: the list of tokens
        """ 
        return self.objNlpTokenizer.tokenize(doc)
    
if __name__ == '__main__':
    #Sentence example
    sentence = "This is an example sentence, showing how to tokenize text https://regex101.com/r/CKCEuL/1 --Obama 21st of January 2027. It includes numbers like 123, 45.67 and words with apostrophes like don't, abbreviations like U.S.A., and special characters like @Lea #swag. -Obama"
    obj = RegexpTokenizer()
    print(obj.nlpReg_tokenizer(sentence))