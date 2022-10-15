from classes import node
import clip
from libs.spliter.sentence_splitter import SentenceSplitter, split_text_into_sentences

class Text():
    def __init__(self, plain_text) -> None:
        # 从plain text作为输入，调用CLIP进行tokenize，将其tokenized结果保存。
        self.plain_text = plain_text
        self.sentences = self.split_sentence()
        self.tokenized = self.tokenize()

    def split_sentence(self):
        splitter = SentenceSplitter(language='en')
        splited_sentences = split_text_into_sentences(text=self.plain_text,language='en')
        return splited_sentences
    
    def tokenize(self):
        # 将文本转为tokenize
        tokenized_text = clip.tokenize(self.sentences)
        return tokenized_text
