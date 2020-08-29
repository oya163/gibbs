import regex as re
import grapheme
from nepali_stemmer.stemmer import NepStemmer

nepstem = NepStemmer()


def filter(text):
    text = re.sub(r'\([^)]*\)', r'', text)
    text = re.sub(r'\[[^\]]*\]', r'', text)
    text = re.sub(r'<[^>]*>', r'', text)
    text = re.sub(r'[!।,\'\’\‘\—()?]', r'', text)
    text = re.sub(r'[०१२३४५६७८९]', r'', text)
    text = text.replace(u'\ufeff', '')
    text = text.replace(u'\xa0', u' ')
    text = re.sub(r'( )+', r' ', text)
    text = re.sub(r"^\s+", "", text)
    return text


data = set()
with open('test_corpus.txt', 'r', encoding='utf-8') as input_file, open('gold_standard.txt', 'w', encoding='utf-8') as output_file:
    sent = input_file.read()
    for each in sent.split():
        each = filter(each)
        # Do not collect no-split results
        # Do not collect same result twice
        # if each != result and result not in data:
        if each and each not in data and grapheme.length(each) > 1:
            result = nepstem.stem(each)
            data.add(result)
            if each == result:
                result = result + ' ' + '$'
            output_file.write(each+'\t'+result+'\n')
            print(each+'\t'+result)
