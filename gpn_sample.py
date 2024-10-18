import sys
from gpn.model import *
from utils import *
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer
# select single sequence gpn model
model_path = "./gpn_model/gpn-brassicales"
# set tokenizer parallelism = false
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
# get a look of how tokenizer convert letters to numeric represents
print(tokenizer.get_vocab())
# sample sequence 
seq = "CGGGTTAAAAATCTAGTTGTTATTATTAAAGGAAATAAAATATCCTCATAAAACAATTTGTTGTAATCTATCTTTGGGCTAATGTTCTTATCCTACAAGACGAACCCTGACCGTATTCGTCGTAGAAAAAAAATTGCTTCGATCCCATCATTGAGTTCAATAATCGGCGCACAAAGGCCGATTCATAAAAACTCTAGGCCCATTAAAGTAAAGCCCATTCTCAACCCTATCCAGTCTCCCTGTATATATATATTTACGACACCAACCCAGCGTTGATATTTAATTTTCTTCAGTCAGAGATTTCGAAACCCTAGTCGATTTCGAGATCCAACTAACTCTGCTCCTTATCTCAGGTAAAATTCTCGCTCGAGAACTCAATTGCTTATCCAAAGTTCCAACTGAAGATGCTTTCCTACTGAATCTTAGGTTAATGTTTTGGATTTGGAATCTTACCCGAAATTTCTCTGCAGCTTGTTGAATTTGCGAAGTATGGGAGACGCTAGAGACAACGAAGCCTACGAGGAGGAGCTCTTGGACTATGAAGAAGAAGACGAGAAGGTCCCAGATTCTGGAAACAAAGTTAACGGCGAAGCTGTGAAAAAGTGAGTTTTATGGTTTCCTCGATATGTTTCATGTATACTACTGTGTGTTTAAATTTGTCGATTCTTAGATTACTACTTGATAACAAGTAGCAGTATGT"
len(seq)
# convert input seq to tokenizered seq
input_ids = tokenizer(seq, return_tensors="pt", return_attention_mask=False, return_token_type_ids=False)["input_ids"]
print(input_ids.shape)

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# load pretrained gpn model
model = AutoModel.from_pretrained(model_path)

#Additional Info when using cuda
if device.type == 'cuda':
    # load the tokenized seq onto gpu
    input_ids = input_ids.cuda()
    # load the model onto gpu
    model.cuda()
    model.eval()
    # get embedding
    with torch.no_grad():
        embedding = model(input_ids=input_ids).last_hidden_state
    print(embedding.shape)
else:
    # use the model on cpu
    model.eval()
    # get embedding
    with torch.no_grad():
        embedding = model(input_ids=input_ids).last_hidden_state
    print(embedding.shape) 