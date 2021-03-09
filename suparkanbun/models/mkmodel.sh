#! /bin/sh
# pip3 install transformers seqeval datasets supar
test -f run_ner.py || curl -LO https://raw.githubusercontent.com/huggingface/transformers/v4.0.1/examples/token-classification/run_ner.py
python3 -c '
from suparkanbun.simplify import simplify
c=[]
h=[0]
while True:
  try:
    s=input()
  except:
    quit()
  t=s.strip().split("\t")
  if len(t)==10:
    if t[0]!="#":
      t[0]=str(len(c)+1)
      i=len(t[1])
      if i>1:
        form=t[1]
        lemma=t[2]
        head=t[6]
        deprel=t[7]
        for j in range(0,i-1):
          t[1]=form[j]
          if t[1] in simplify:
            t[1]=simplify[t[1]]
          t[2]=lemma[j]
          t[6]="-1"
          t[7]="compound"
          c.append(list(t))
          t[0]=str(len(c)+1)
        t[1]=form[i-1]
        t[2]=lemma[i-1]
        t[6]=head
        t[7]=deprel
      if t[1] in simplify:
        t[1]=simplify[t[1]]
      c.append(list(t))
      h.append(len(c))
  elif s.strip()=="":
    for t in c:
      t[6]=str(int(t[0])+1 if t[6]=="-1" else h[int(t[6])])
      print("\t".join(t))
    print("")
    c=[]
    h=[0]
' < lzh_kyoto.conllu | tee simplified.conllu | python3 -c '
tokens=[]
tags=[]
while True:
  try:
    s=input()
  except:
    if len(tokens)>0:
      print("{\"tokens\":[\""+"\",\"".join(tokens)+"\"],\"tags\":[\""+"\",\"".join(tags)+"\"]}")
    quit()
  t=s.split("\t")
  if len(t)==10:
    p=t[4]+","+t[3]+","+t[5]
    for c in t[1]:
      tokens.append(c)
      tags.append(p)
  elif len(tokens)>80:
    print("{\"tokens\":[\""+"\",\"".join(tokens)+"\"],\"tags\":[\""+"\",\"".join(tags)+"\"]}")
    tokens=[]
    tags=[]
' | nawk '
{
  if(NR%10>0)
    printf("%s\n",$0)>"trainPOS.json";
  else
    printf("%s\n",$0)>"validPOS.json";
}'
sed 's/^.*"tags":\[//' trainPOS.json | tr '"' '\012' | sort -u | egrep '^[nvps],' > labelPOS.txt
if [ ! -d guwenbert-base.pos ]
then mkdir -p guwenbert-base.pos
     python3 run_ner.py --model_name_or_path ethanyt/guwenbert-base --train_file trainPOS.json --validation_file validPOS.json --output_dir guwenbert-base.pos --do_train --do_eval
fi
if [ ! -d guwenbert-large.pos ]
then mkdir -p guwenbert-large.pos
     python3 run_ner.py --model_name_or_path ethanyt/guwenbert-large --train_file trainPOS.json --validation_file validPOS.json --output_dir guwenbert-large.pos --do_train --do_eval
fi

nawk '
BEGIN{
  f[0]="test.conllu";
  f[1]="dev.conllu";
  for(i=2;i<10;i++)
    f[i]="train.conllu";
}
{
  printf("%s\n",$0)>f[i%10];
  if($0=="")
    i++;
}' simplified.conllu
if [ ! -f guwenbert-base.pos/guwenbert-base.supar ]
then python3 -m supar.cmds.biaffine_dependency train -b -d 0 -p guwenbert-base.pos/guwenbert-base.supar --epochs=1000 -f bert --bert ethanyt/guwenbert-base --train train.conllu --dev dev.conllu --test test.conllu --embed=''
fi
if [ ! -f guwenbert-large.pos/guwenbert-large.supar ]
then python3 -m supar.cmds.biaffine_dependency train -b -d 0 -p guwenbert-large.pos/guwenbert-large.supar --epochs=1000 -f bert --bert ethanyt/guwenbert-large --train train.conllu --dev dev.conllu --test test.conllu --embed=''
fi

python3 -c '
tokens=[]
tags=[]
i=0
while True:
  try:
    s=input()
  except:
    if len(tokens)>0:
      print("{\"tokens\":[\""+"\",\"".join(tokens)+"\"],\"tags\":[\""+"\",\"".join(tags)+"\"]}")
    quit()
  t=s.split("\t")
  if len(t)==10:
    for c in t[1]:
      tokens.append(c)
      i+=1
  else:
    if i==1:
      tags.append("S")
    elif i==2:
      tags+=["B","E"]
    elif i==3:
      tags+=["B","E2","E"]
    else:
      tags+=["B"]+["M"]*(i-4)+["E3","E2","E"]
    i=0
    if len(tokens)>80:
      print("{\"tokens\":[\""+"\",\"".join(tokens)+"\"],\"tags\":[\""+"\",\"".join(tags)+"\"]}")
      tokens=[]
      tags=[]
' < simplified.conllu | nawk '
{
  if(NR%10>0)
    printf("%s\n",$0)>"trainDanku.json";
  else
    printf("%s\n",$0)>"validDanku.json";
}'
sed 's/^.*"tags":\[//' trainDanku.json | tr '"' '\012' | sort -u | egrep '^[A-Z]' > labelDanku.txt
if [ ! -d guwenbert-base.danku ]
then mkdir -p guwenbert-base.danku
     python3 run_ner.py --model_name_or_path ethanyt/guwenbert-base --train_file trainDanku.json --validation_file validDanku.json --output_dir guwenbert-base.danku --do_train --do_eval
fi
if [ ! -d guwenbert-large.danku ]
then mkdir -p guwenbert-large.danku
     python3 run_ner.py --model_name_or_path ethanyt/guwenbert-large --train_file trainDanku.json --validation_file validDanku.json --output_dir guwenbert-large.danku --do_train --do_eval
fi
exit 0
