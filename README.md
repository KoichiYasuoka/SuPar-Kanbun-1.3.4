[![Current PyPI packages](https://badge.fury.io/py/suparkanbun.svg)](https://pypi.org/project/suparkanbun/)

# SuPar-Kanbun

Tokenizer, POS-Tagger and Dependency-Parser for Classical Chinese Texts (漢文/文言文) with [spaCy](https://spacy.io), [Transformers](https://huggingface.co/transformers/) and [SuPar](https://github.com/yzhangcs/parser).

## Basic usage

```py
>>> import suparkanbun
>>> nlp=suparkanbun.load()
>>> doc=nlp("不入虎穴不得虎子")
>>> print(type(doc))
<class 'spacy.tokens.doc.Doc'>
>>> print(suparkanbun.to_conllu(doc))
# text = 不入虎穴不得虎子
1	不	不	ADV	v,副詞,否定,無界	Polarity=Neg	2	advmod	_	SpaceAfter=No
2	入	入	VERB	v,動詞,行為,移動	_	0	root	_	SpaceAfter=No
3	虎	虎	NOUN	n,名詞,主体,動物	_	4	nmod	_	SpaceAfter=No
4	穴	穴	NOUN	n,名詞,固定物,地形	Case=Loc	2	obj	_	SpaceAfter=No
5	不	不	ADV	v,副詞,否定,無界	Polarity=Neg	6	advmod	_	SpaceAfter=No
6	得	得	VERB	v,動詞,行為,得失	_	2	parataxis	_	SpaceAfter=No
7	虎	虎	NOUN	n,名詞,主体,動物	_	8	nmod	_	SpaceAfter=No
8	子	子	NOUN	n,名詞,人,関係	_	6	obj	_	SpaceAfter=No

>>> import deplacy
>>> deplacy.render(doc)
不 ADV  <════╗   advmod
入 VERB ═══╗═╝═╗ ROOT
虎 NOUN <╗ ║   ║ nmod
穴 NOUN ═╝<╝   ║ obj
不 ADV  <════╗ ║ advmod
得 VERB ═══╗═╝<╝ parataxis
虎 NOUN <╗ ║     nmod
子 NOUN ═╝<╝     obj
```

`suparkanbun.load()` has two options `suparkanbun.load(BERT="guwenbert-base",Danku=False)`. With the option `BERT="guwenbert-large"` the pipeline utilizes [GuwenBERT-large](https://huggingface.co/ethanyt/guwenbert-large). With the option `Danku=True` the pipeline tries to segment sentences automatically.

## Installation for Linux

```sh
pip3 install suparkanbun --user
```

## Installation for Cygwin64

Make sure to get `python37-devel` `python37-pip` `python37-cython` `python37-numpy` `python37-wheel` `gcc-g++` `mingw64-x86_64-gcc-g++` `git` `curl` `make` `cmake` packages, and then:
```sh
curl -L https://raw.githubusercontent.com/KoichiYasuoka/CygTorch/master/installer/supar.sh | sh
pip3.7 install suparkanbun --no-build-isolation
```

## Installation for Jupyter Notebook (Google Colaboratory)

```py
!pip install suparkanbun 
```

Try [notebook](https://colab.research.google.com/github/KoichiYasuoka/SuPar-Kanbun/blob/main/suparkanbun.ipynb) for Google Colaboratory.

