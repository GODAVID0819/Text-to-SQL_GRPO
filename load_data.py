import os

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import T5TokenizerFast
import torch

PAD_IDX = 0
MODEL_NAME = 'google-t5/t5-small'


class T5Dataset(Dataset):
    def __init__(self, data_folder, split):
        '''
        Dataset for T5 text-to-SQL.
        '''
        self.data_folder = data_folder
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)
        self.decoder_bos_id = self.tokenizer.convert_tokens_to_ids('<extra_id_0>')
        self.data = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        nl_path = os.path.join(data_folder, f'{split}.nl')
        nl_lines = load_lines(nl_path)

        sql_lines = None
        if split != 'test':
            sql_path = os.path.join(data_folder, f'{split}.sql')
            sql_lines = load_lines(sql_path)
            assert len(nl_lines) == len(sql_lines), 'Mismatch between NL and SQL example counts.'

        processed = []
        for idx, nl in enumerate(nl_lines):
            encoder_ids = tokenizer.encode(nl, add_special_tokens=True)
            example = {
                'encoder_ids': torch.tensor(encoder_ids, dtype=torch.long),
                'initial_decoder_input': torch.tensor([self.decoder_bos_id], dtype=torch.long),
            }

            if sql_lines is not None:
                target_ids = tokenizer.encode(sql_lines[idx], add_special_tokens=True)
                decoder_inputs = [self.decoder_bos_id] + target_ids[:-1]
                example['decoder_inputs'] = torch.tensor(decoder_inputs, dtype=torch.long)
                example['decoder_targets'] = torch.tensor(target_ids, dtype=torch.long)
                example['sql_text'] = sql_lines[idx]

            processed.append(example)
        return processed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        if self.split == 'test':
            return ex['encoder_ids'], ex['initial_decoder_input']
        return ex['encoder_ids'], ex['decoder_inputs'], ex['decoder_targets'], ex['initial_decoder_input']


def normal_collate_fn(batch):
    encoder_ids, decoder_inputs, decoder_targets, initial_decoder_inputs = zip(*batch)

    encoder_ids = pad_sequence(encoder_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    decoder_inputs = pad_sequence(decoder_inputs, batch_first=True, padding_value=PAD_IDX)
    decoder_targets = pad_sequence(decoder_targets, batch_first=True, padding_value=PAD_IDX)
    initial_decoder_inputs = torch.stack(initial_decoder_inputs, dim=0)

    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs



def test_collate_fn(batch):
    encoder_ids, initial_decoder_inputs = zip(*batch)

    encoder_ids = pad_sequence(encoder_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    initial_decoder_inputs = torch.stack(initial_decoder_inputs, dim=0)

    return encoder_ids, encoder_mask, initial_decoder_inputs



def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == 'train'
    collate_fn = normal_collate_fn if split != 'test' else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader



def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, 'train')
    dev_loader = get_dataloader(test_batch_size, 'dev')
    test_loader = get_dataloader(test_batch_size, 'test')

    return train_loader, dev_loader, test_loader



def load_lines(path):
    with open(path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    return lines



def load_prompting_data(data_folder):
    train_x = load_lines(os.path.join(data_folder, 'train.nl'))
    train_y = load_lines(os.path.join(data_folder, 'train.sql'))
    dev_x = load_lines(os.path.join(data_folder, 'dev.nl'))
    dev_y = load_lines(os.path.join(data_folder, 'dev.sql'))
    test_x = load_lines(os.path.join(data_folder, 'test.nl'))
    return train_x, train_y, dev_x, dev_y, test_x
