class Tokenizer:

    def __init__(self) -> None:
        self.endb = {}
        self.dedb = {}

    def train(self,x):
        self.endb = { i:j for i,j in enumerate(x)}
        self.dedb = { j:i for i,j in enumerate(x)}

    def encode(self ,x): # text -> num
        return [ self.dedb[i] for i in x]
    
    def decode(self ,x): # num -> text
        return [ self.endb[i] for i in x]
