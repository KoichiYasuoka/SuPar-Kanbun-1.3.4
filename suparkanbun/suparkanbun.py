#! /usr/bin/python3 -i
# coding=utf-8

import os
PACKAGE_DIR=os.path.abspath(os.path.dirname(__file__))
DOWNLOAD_DIR=os.path.join(PACKAGE_DIR,"models")
MODEL_URL="https://raw.githubusercontent.com/KoichiYasuoka/SuPar-Kanbun/main/suparkanbun/models/"

import numpy
from spacy.language import Language
from spacy.symbols import LANG,LEMMA,POS,TAG,DEP,HEAD
from spacy.tokens import Doc,Span,Token
from spacy.util import get_lang_class

class SuParKanbunLanguage(Language):
  lang="lzh"
  max_length=10**6
  def __init__(self,BERT,Danku):
    self.Defaults.lex_attr_getters[LANG]=lambda _text:"lzh"
    try:
      self.vocab=self.Defaults.create_vocab()
      self.pipeline=[]
    except:
      from spacy.vocab import create_vocab
      self.vocab=create_vocab("lzh",self.Defaults)
      self._components=[]
      self._disabled=set()
    self.tokenizer=SuParKanbunTokenizer(BERT,Danku,self.vocab)
    self._meta={
      "author":"Koichi Yasuoka",
      "description":"derived from SuParKanbun",
      "lang":"SuParKanbun_lzh",
      "license":"MIT",
      "name":"SuParKanbun_lzh",
      "parent_package":"suparkanbun",
      "pipeline":"Tokenizer, POS-Tagger, Parser",
      "spacy_version":">=2.1.0"
    }
    self._path=None

class SuParKanbunTokenizer(object):
  to_disk=lambda self,*args,**kwargs:None
  from_disk=lambda self,*args,**kwargs:None
  to_bytes=lambda self,*args,**kwargs:None
  from_bytes=lambda self,*args,**kwargs:None
  def __init__(self,bert,danku,vocab):
    from supar import Parser
    self.bert=bert
    self.vocab=vocab
    with open(os.path.join(DOWNLOAD_DIR,"labelPOS.txt"),"r",encoding="utf-8") as f:
      r=f.read()
    d=os.path.join(DOWNLOAD_DIR,bert+".pos")
    self.tagger=AutoModelTagger(d,r.strip().split("\n"))
    f=os.path.join(d,bert+".supar")
    self.supar=Parser.load(f)
    if danku:
      d=os.path.join(DOWNLOAD_DIR,bert+".danku")
      self.danku=AutoModelTagger(d,["B","E","E2","E3","M","S"])
    else:
      self.danku=None
  def __call__(self,text):
    from suparkanbun.simplify import simplify
    t=""
    for c in text:
      if c in simplify:
        t+=simplify[c]
      else:
        t+=c
    if self.danku!=None:
      s=self.danku(t.replace("\n",""))
      t=""
      for c,p in s:
        t+=c
        if p=="S" or p=="E":
          t+="\n"
    p=self.tagger(t.replace("\n",""))
    u=self.supar.predict([[c for c in s] for s in t.strip().split("\n")])
    t=text.replace("\n","")
    i=0
    w=[]
    for s in u.sentences:
      v=[]
      for h,d in zip(s.values[6],s.values[7]):
        v.append({"form":t[i],"lemma":p[i][0],"pos":p[i][1],"head":h,"deprel":d})
        i+=1
      for j in reversed(range(0,len(v)-1)):
        if v[j]["deprel"]=="compound" and v[j]["head"]==j+2 and v[j]["pos"]==v[j+1]["pos"]:
          k=v.pop(j)
          v[j]["form"]=k["form"]+v[j]["form"]
          v[j]["lemma"]=k["lemma"]+v[j]["lemma"]
          for k in range(0,len(v)):
            if v[k]["head"]>j+1:
              v[k]["head"]-=1
      w.append(list(v))
    vs=self.vocab.strings
    r=vs.add("ROOT")
    words=[]
    lemmas=[]
    pos=[]
    tags=[]
    feats=[]
    heads=[]
    deps=[]
    spaces=[]
    for s in w:
      for i,t in enumerate(s):
        words.append(t["form"])
        lemmas.append(vs.add(t["lemma"]))
        p=t["pos"].split(",")
        pos.append(vs.add(p[4]))
        tags.append(vs.add(",".join(p[0:4])))
        feats.append(p[5])
        if t["deprel"]=="root":
          heads.append(0)
          deps.append(r)
        else:
          heads.append(t["head"]-i-1)
          deps.append(vs.add(t["deprel"]))
        spaces.append(False)
    doc=Doc(self.vocab,words=words,spaces=spaces)
    a=numpy.array(list(zip(lemmas,pos,tags,deps,heads)),dtype="uint64")
    doc.from_array([LEMMA,POS,TAG,DEP,HEAD],a)
    try:
      doc.is_tagged=True
      doc.is_parsed=True
    except:
      for i,j in enumerate(feats):
        if j!="_" and j!="":
          doc[i].set_morph(j)
    return doc

class AutoModelTagger(object):
  def __init__(self,dir,label):
    from suparkanbun.download import checkdownload
    from transformers import AutoModelForTokenClassification,AutoTokenizer
    checkdownload(MODEL_URL+os.path.basename(dir)+"/",dir)
    self.model=AutoModelForTokenClassification.from_pretrained(dir)
    self.tokenizer=AutoTokenizer.from_pretrained(dir)
    self.label=label
  def __call__(self,text):
    import torch
    input=self.tokenizer.encode(text,return_tensors="pt")
    output=self.model(input)
    predict=torch.argmax(output[0],dim=2)
    return [(t,self.label[p]) for t,p in zip(text,predict[0].tolist()[1:])]

def load(BERT="guwenbert-base",Danku=False):
  return SuParKanbunLanguage(BERT,Danku)

def to_conllu(item,offset=1):
  if type(item)==Doc:
    return "".join(to_conllu(s)+"\n" for s in item.sents)
  elif type(item)==Span:
    return "# text = "+str(item)+"\n"+"".join(to_conllu(t,1-item.start)+"\n" for t in item)
  elif type(item)==Token:
    m="_" if item.whitespace_ else "SpaceAfter=No"
    if item.norm_!="":
      if item.norm_!=item.orth_:
        m="Gloss="+item.norm_+"|"+m
        m=m.replace("|_","")
    l=item.lemma_
    if l=="":
      l="_"
    t=item.tag_
    if t=="":
      t="_"
    try:
      f=str(item.morph)
      if f.startswith("<spacy") or f=="":
        f="_"
    except:
      f="_"
    return "\t".join([str(item.i+offset),item.orth_,l,item.pos_,t,f,str(0 if item.head==item else item.head.i+offset),item.dep_.lower(),"_",m])
  return "".join(to_conllu(s)+"\n" for s in item)

