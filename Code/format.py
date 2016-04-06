import json

f = open("snli_1.0_train.jsonl",'r')
f2 = open("train.txt",'w')

for line in f:
	a = json.loads(line)
	f2.write(a['gold_label'])
	f2.write('\n')
	f2.write(a['sentence1'])
	f2.write('\n')
	f2.write(a['sentence2'])
	f2.write('\n')

f2.close()
f.close()