from Bio import Entrez
from tqdm import tqdm
import codecs

linesToRead = 1003

genids = open("genbankid.txt", "r")

Entrez.email = "seacheo@nuevaschool.org"
Entrez.api_key = "c268992545656cfcb41217f8583361c99707"

for p in tqdm(range(0, linesToRead)):
    usefulGenesStart = []
    usefulGenesEnd = []
    tempcur = genids.readline()
    if (tempcur[0] == 'C' and tempcur[1] == 'P'):
        tempcur = tempcur[:-1]
        fasta =(Entrez.efetch(db="nucleotide", id=tempcur, rettype="fasta", retmode="text"))
        fast = (fasta.read())
        # try:
        #     fasta[j].read()
        # except:
        #     print("Ruh roh NCBI fasta read didnt work " + str(j) + " " + curid)
        #     fasta[j] = (Entrez.efetch(db="nucleotide", id=curid[j], rettype="fasta", retmode="text"))
        # j+=1
        file = codecs.open('testdata/' + tempcur + '.fna', "w+", "utf-8")
        file.write(fast)
        file.close()
