class Tokenizer:
    def __init__(self):
        self.start = "^"
        self.end = "$"
        self.pad = ' '
        self.int_to_char = None
        self.char_to_int = None
        self.tokenlist = None

    def build_vocab(self):
        chars = []

        chars = chars + ['H', 'B', "C", 'c', 'N', 'n', 'O', 'o', 'P', 'S', 's', 'F', 'I']

        chars = chars + ['Q', 'R', 'V', 'Y', 'Z', 'G', 'T', 'U']

        chars = chars + ['[', ']', '+', 'W', 'X']

        chars = chars + ['-', '=', '#', '.', '/', '@', '\\']

        chars = chars + ['(', ')']

        chars = chars + ['1', '2', '3', '4', '5', '6', '7', '8', '9']

        self.tokenlist = [self.pad, self.start, self.end] + list(chars)

        self.char_to_int = {c: i for i, c in enumerate(self.tokenlist)}

        self.int_to_char = {i: c for c, i in self.char_to_int.items()}

    @property
    def vocab_size(self):
        return len(self.int_to_char)

    def encode(self, smi):
        
        smi = smi.replace('Si', 'Q')
        smi = smi.replace('Cl', 'R')
        smi = smi.replace('Br', 'V')
        smi = smi.replace('Pt', 'Y')
        smi = smi.replace('Se', 'Z')
        smi = smi.replace('se', 'Z')
        smi = smi.replace('Li', 'T')
        smi = smi.replace('As', 'U')
        smi = smi.replace('Hg', 'G')
        smi = smi.replace('H2', 'W')
        smi = smi.replace('H3', 'X')
        return ([self.char_to_int[self.start]] +
                [self.char_to_int[s] for s in smi] +
                [self.char_to_int[self.end]])


    def decode(self, ords):
        smi = ''.join([self.int_to_char[o] for o in ords])
        smi = smi.replace('W', 'H2')
        smi = smi.replace('X', 'H3')
        smi = smi.replace('Q', 'Si')
        smi = smi.replace('R', 'Cl')
        smi = smi.replace('V', 'Br')
        smi = smi.replace('Y', 'Pt')
        smi = smi.replace('Z', 'Se')
        smi = smi.replace('T', 'Li')
        smi = smi.replace('U', 'As')
        smi = smi.replace('G', 'Hg')
        return smi

    @property
    def n_tokens(self):
        return len(self.int_to_char)


def vocabulary(args):
    tokenizer = Tokenizer()
    tokenizer.build_vocab()

    return tokenizer


if __name__ == '__main__':
    tokenize = Tokenizer()
    tokenize.build_vocab()
    print(tokenize.encode('CC(C)C'))
    print('\n')
    print('Vocabulary Information:')
    print('=' * 50)
    print(tokenize.char_to_int)
    print(tokenize.int_to_char)
    print('=' * 50)
