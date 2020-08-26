import regex as re

from nepali_stemmer.stemmer import NepStemmer
from nltk.tokenize import WhitespaceTokenizer

nepstem = NepStemmer()
tk = WhitespaceTokenizer()


def filter(text):
    text = re.sub(r'\([^)]*\)', r'', text)
    text = re.sub(r'\[[^\]]*\]', r'', text)
    text = re.sub(r'<[^>]*>', r'', text)
    text = re.sub(r'[!।,\']', r'', text)
    text = re.sub(r'[०१२३४५६७८९]', r'', text)
    text = text.replace(u'\ufeff', '')
    text = text.replace(u'\xa0', u' ')
    text = re.sub(r'( )+', r' ', text)
    return text


data = set()
with open('test_corpus.txt', 'r', encoding='utf-8') as input_file, open('gold_standard.txt', 'w', encoding='utf-8') as output_file:
    sent = input_file.read()
    for each in sent.split():
        each = filter(each)
        result = nepstem.stem(each)
        # Do not collect no-split results
        # Do not collect same result twice
        if each != result and result not in data:
            data.add(result)
            # Whitespace tokenize
            res = tk.tokenize(result)
            # Get text spans
            span = list(tk.span_tokenize(result))
            span_out = str(span[0][0])+' '+str(span[0][1])+'\t'+str(span[1][0])+' '+str(span[1][1])
            output_file.write(each+'\t'+result+'\t'+span_out+'\n')
            print(each+'\t'+result+'\t'+span_out)
