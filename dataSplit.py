import random
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.Alphabet import generic_dna
filename ='./mtDNA.fasta'# downloaded multi-fasta file from MITOMAP database
reads = []
with open(filename, "rU") as handle:
	for record in SeqIO.parse(handle, "fasta") :
		if(len(record.seq) > 16400):
			reads.append(record)
print(len(reads))

items = range(len(reads))
random.shuffle(items)
train = items[0:700]
valid = items[700:900]
test = items[900:1000]
train_record = []
valid_record = []
test_record = []

def check(read):
	record = read
	my_dna = str(read.seq.upper())
	for i, base in enumerate(my_dna):
		if base not in 'ACGT':
			my_dna = my_dna.replace(base,'A')
	record.seq= Seq(my_dna, generic_dna)
	for i, base in enumerate(record.seq):
		if base not in 'ACGT':
			print(record.seq[i])
	return record
for i in train:
	read = check(reads[i])
	train_record.append(read)
for i in valid:
	read = check(reads[i])
	valid_record.append(read)
for i in test:
	read = check(reads[i])	
	test_record.append(read)

#save the data
SeqIO.write(train_record, "./data/train.fasta", "fasta")
SeqIO.write(valid_record, "./data/valid.fasta", "fasta")
SeqIO.write(test_record, "./data/test.fasta", "fasta")
