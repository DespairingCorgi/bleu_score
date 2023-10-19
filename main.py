from bpemb import BPEmb
from nltk.translate.bleu_score import sentence_bleu
from time import time
import pandas as pd

def init_model(lang, dim=100, vs=16000):
    
    s_t = time()
    model = BPEmb(lang=lang, dim=dim, vs=vs)
    e_t = time()
    print(f"model loaded: {e_t-s_t} s")
    return model

def bleu_score(model, ref_text, cand_text, use_decimal=True, decimal=3):
    ref_encoded = model.encode(ref_text)
    cand_encoded = model.encode(cand_text)
    score = sentence_bleu([ref_encoded], cand_encoded)
    if use_decimal:
        return round(score, decimal)
    else:
        return score

def get_bleu(path, ref_col, cand_col, \
    lang, dim=100, vs=16000,\
    use_decimal=True, decimal=3):
    '''
        required:
            lang: ko/zh/vi/ru/fr/en
            ref_col: reference column name
            cand_col: candidate column name
        optional(change not recommended):
            initializing model:
                dim: default 100 max 300
                vs: default 16000 max 200000
            bleu_score conventions:
                use_decimal: boolean value - whether use round decimal or not
                decimal: decimal point - only valid when use_decimal=True
    '''
    model = init_model(lang, dim, vs)
    df = pd.read_excel(path)[[ref_col, cand_col]]
    print(df.columns)
    df["blue_score"] = df.apply(lambda x: bleu_score(model, x[ref_col], x[cand_col], use_decimal, decimal), axis=1)
    curpath = path.split('.')
    curfilename = '.'.join(curpath[:-1]) + '_bleu.' + curpath[-1]
    df.to_excel(curfilename, index=False)

if __name__=="__main__":
    
    get_bleu('ROUGE score (한영100문장, 영한100문장).xlsx', 'ko_original', 'ko_matis', 'ko')