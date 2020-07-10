import os
import torch
import pandas as pd
import re
from transformers import BertTokenizer,BertForMaskedLM ,AdamW,get_linear_schedule_with_warmup
import time
from datetime import  timedelta
from torch.utils.data import DataLoader,TensorDataset
from tqdm import tqdm
from apex import amp
import torch.distributed as dist
import torch.multiprocessing as mp
import pickle

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def add_masking(token_id_list, tokenizer, vocab_list):
    """
    Adding masking to each sentence for masked language modelling pretraining
    """
    import random
    tot_len = len(token_id_list)
    mask_list = random.sample(range(tot_len), 19)
    for x in range(0, 15):
        if (token_id_list[mask_list[x]] != tokenizer.sep_token_id) and (
                token_id_list[mask_list[x]] != tokenizer.cls_token_id):
            token_id_list[mask_list[x]] = tokenizer.mask_token_id
    for x in range(15, 17):
        if (token_id_list[mask_list[x]] != tokenizer.sep_token_id) and (
                token_id_list[mask_list[x]] != tokenizer.cls_token_id):
            token_id_list[mask_list[x]] = random.sample(vocab_list, 1)[0]
    return token_id_list

def train(gpu,config):
    def format_time(elapsed):
        '''
        Takes a time in seconds and returns a string hh:mm:ss
        '''
        # Round to the nearest second.
        elapsed_rounded = int(round((elapsed)))

        # Format as hh:mm:ss
        return str(timedelta(seconds=elapsed_rounded))

    #set rank for each GPU 
    rank = gpu
    
    #initiate process group and specify backend configurations
    dist.init_process_group(backend='nccl',init_method='env://',world_size=4,rank=rank)
    
    torch.manual_seed(100)
    #load model and push it into GPU device
    print('model load started')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased', output_hidden_states=True)
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    print('model loaded into device')

    #setup optimizers and other utils
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, eps=config.EPSILON)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[gpu], find_unused_parameters=True)
    train_dataset = TensorDataset(config.masked_sentences, config.original_sentences)

    #setup training sampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,num_replicas=config.world_size)
    
    #setup training data loader with the train sampler setup
    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,sampler=train_sampler, shuffle=False)  # num_workers=4


    total_steps = len(train_dataloader) / config.BATCH_SIZE * config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    tot_loss = 0.0
    steps = 0.0


    model.zero_grad()
    training_stats = []
    for epoch in range(config.EPOCHS):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, config.EPOCHS))
        print('Training...')

        t0 = time.time()

        for step, batch in enumerate(train_dataloader):

            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            batch_input_tensors = batch[0].to('cuda')
            batch_labels = batch[1].to('cuda')

            model.train()
            outputs = model(batch_input_tensors, masked_lm_labels=batch_labels)
            loss = outputs[0]

            #loss.backward()
            tot_loss += loss.item()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1.0)

            optimizer.step()
            scheduler.step()
            model.zero_grad()

            steps += 1.0
        training_time = format_time(time.time() - t0)
        avg_train_loss = tot_loss / steps
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))
        print("Epoch:{} loss:{}".format(epoch + 1, tot_loss / steps))
        training_stats.append(
            {
                'epoch': epoch,
                'training loss': avg_train_loss,
                'training time': training_time
            }
        )
    if gpu == 0:
        import os
        output_dir = './mlm_model_save/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("Saving model to %s" % output_dir)

        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        config.tokenizer.save_pretrained(output_dir)


class Config:
    def __init__(self):
        self.DATA_FILE = None
        self.MAX_SEQ_LENGTH = 128
        self.BATCH_SIZE = 16
        self.EPOCHS = 20
        self.LEARNING_RATE = 5e-5
        self.EPSILON = 1e-8
        self.MAX_GRAD_NORM = 1.0
        self.original_sentences = None
        self.masked_sentences = None
        self.tokenizer = None
        self.world_size = 4


def main():

    #setup multiple configurations for the training process
    config = Config()
    config.DATA_FILE = 'jobs_tsv'
    config.MAX_SEQ_LENGTH = 128
    config.BATCH_SIZE = 16
    config.EPOCHS = 20
    config.LEARNING_RATE = 5e-5
    config.EPSILON= 1e-8
    config.MAX_GRAD_NORM = 1.0

    to_save = {}

    #getting the list of GPUs available
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
        device_ids = list(range(torch.cuda.device_count()))
        gpus = len(device_ids)
        print('GPU detected')
    else:
        DEVICE = torch.device("cpu")
        print('No GPU. switching to CPU')

    #loading the sentences and data pre-processing
    jobs_data = pd.read_csv(config.DATA_FILE,sep='\t')

    jobs_data['Description'] = jobs_data.apply(lambda row: clean_str(row['Description']),axis=1)

    sentences = jobs_data.Description.values
    print('total length of the dataset',len(sentences))

    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    print('tokenizer loaded')

    vocab_list = list(tokenizer.get_vocab().values())

    print('sentences encoding and masking started')

    sentence_tokens = []
    sentence_tokens_with_masks = []

    #encode sentences and add masking
    for sentence in tqdm(sentences):
        token_id = tokenizer.encode(sentence, max_length=config.MAX_SEQ_LENGTH, add_special_tokens=True, pad_to_max_length=True)
        try:
            token_id_with_mask = add_masking(token_id,tokenizer,vocab_list)
        except Exception as e:
            print(e)
            continue
        sentence_tokens.append(token_id)
        sentence_tokens_with_masks.append(token_id_with_mask)

    print('sentence encoding and masking completed')
    to_save['sentence_tokens'] = sentence_tokens
    to_save['masked_sentence_tokens'] = sentence_tokens_with_masks

    original_sentences = torch.tensor(sentence_tokens)
    masked_sentences = torch.tensor(sentence_tokens_with_masks)

    #load original and masked sentences encoded into the config object
    config.original_sentences = original_sentences
    config.masked_sentences = masked_sentences
    config.tokenizer = tokenizer
    config.world_size = 4
    #pickle.dump(to_save,open('mlm_params','wb'))
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '8888'
    
    #initiate the spawning process
    mp.spawn(train,nprocs=config.world_size,args=(config,))

if __name__ == "__main__":
    main()