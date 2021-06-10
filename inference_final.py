from torch.utils.data import Dataset, DataLoader
class e2eDataset(Dataset):
    def __init__(self, csv_file, tokenizer):
        """
        Args:
            csv_file (string): csv 파일의 경로
        """
        self.dataset = pd.read_csv(csv_file)
        self.columns = self.dataset.columns
        self.conditions = self.dataset[self.columns[0]]
        self.sentences = self.dataset[self.columns[1]]
        self.tokenizer = tokenizer
        
        self.typ_list = {}
        for k in range(len(self.conditions)):
            cond_set = self.conditions[k].split(',')
            for m in range(len(cond_set)):
                cond_set[m] = cond_set[m].strip()
                pos = cond_set[m].index('[')
                if cond_set[m][:pos] in self.typ_list.keys():
                    self.typ_list[cond_set[m][:pos]].add(cond_set[m][pos+1:-1])
                else:            
                    self.typ_list[cond_set[m][:pos]] = {cond_set[m][pos+1:-1]}        

    def __len__(self):
        return len(self.conditions)

    def __getitem__(self, idx):
        cond = self.conditions[idx]
        cond_set = cond.split(',')
        condition_string = ''
        for m in range(len(cond_set)):
            cond_set[m] = cond_set[m].strip()
            pos = cond_set[m].index('[')
            
            condition_string += '<' + cond_set[m][:pos] + '>' + cond_set[m][pos+1:-1] + ' '
        
        sen = self.sentences[idx]
        input_string = condition_string + '<START>'
        input_ids = torch.tensor(self.tokenizer.encode(input_string, add_special_tokens=True))
        
        input_len = len(input_ids)

        return input_ids, sen, condition_string

    
# from model_large import *
from model import *
from tqdm import tqdm
import time
import nltk
import os
# nltk.download('punkt')
# from nltk import word_tokenize
# from nltk.translate.bleu_score import sentence_bleu
max_len = int(os.environ['GPT_MAX_LEN'])

my_model = mymodel().cuda()
my_model.eval()
    
model_name = f'./gen_model/{os.environ["GPT_GEN_MOD"]}/final/model'
save_path = f'./predictions/{os.environ["GPT_GEN_MOD"]}'
    
my_model.load_state_dict(torch.load(model_name))
     
e2e_dataset = e2eDataset(csv_file='dataset/testset_w_refs.csv', tokenizer=my_model.tokenizer)
dataloader = DataLoader(e2e_dataset, batch_size=int(os.environ['GPT_BATCH_SIZE']), shuffle=False, num_workers=4)
same_condition = []
ref_sentences = []
input_ids_list = []
pre_condition_string = ''

start = 0
for i_batch, sample_batched in tqdm(enumerate(dataloader), desc='Gathering test data:'):
    sen = sample_batched[1][0]
    condition_string = sample_batched[2]  
    input_ids = sample_batched[0].squeeze(0).cuda()

    if start == 0 or condition_string == pre_condition_string:      
        if start == 0:
            input_ids_list.append(input_ids)
        same_condition.append(sen)        
        pre_condition_string = condition_string
        start += 1
    else:   
        input_ids_list.append(input_ids)
        ref_sentences.append(same_condition)
        pre_condition_string = condition_string
        same_condition = [sen]
        start += 1

ref_sentences.append(same_condition)    

print(len(ref_sentences))

bleu_score = 0
bleu_1 = 0

    
if not os.path.exists(save_path):
    os.makedirs(save_path)    
f_pred = open(save_path+'/f_pred.txt', 'w')


for k in tqdm(range(len(ref_sentences)), desc='Generating predictions:'):
    input_ids = input_ids_list[k]
    input_len = len(input_ids)

    for _ in range(max_len):
        model_out = my_model.model_feeding(input_ids) # (batch, seq_len, emb_dim)
        pred_idx = model_out.argmax(1)[-1]        
        if pred_idx == my_model.tokenizer.eos_token_id:
            break            
        input_ids = torch.cat((input_ids, pred_idx.unsqueeze(0)), 0)        

    out_sen = my_model.tokenizer.decode(input_ids[input_len:])
    f_pred.write(out_sen+'\n')


f_pred.close()