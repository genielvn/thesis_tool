import torch
import wordninja
import torchvision
import os
import logging
import numpy as np
import streamlit as st
from sklearn.metrics import precision_recall_fscore_support
from .models import MsdBERT
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from .resnet_utils import myResnet
from transformers import XLMRobertaTokenizer

model_encoder = "./model/output/pytorch_encoder.bin"
model_model = "./model/output/pytorch_model.bin"
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class InputExample(object):
    def __init__(self, text, img_id, label=None):
        self.text = text
        self.img_id = img_id
        self.label = label

class MMInputFeatures(object):
    def __init__(self, input_ids,
                 input_mask,
                 added_input_mask,
                 img_feat,
                 hashtag_input_ids,
                 hashtag_input_mask,
                 label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.added_input_mask = added_input_mask
        self.img_feat = img_feat
        self.hashtag_input_ids = hashtag_input_ids
        self.hashtag_input_mask = hashtag_input_mask
        self.label_id = label_id


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def macro_f1(y_true, y_pred):
    preds = np.argmax(y_pred, axis=-1)
    true = y_true
    p_macro, r_macro, f_macro, support_macro \
        = precision_recall_fscore_support(true, preds, average='macro')
    return p_macro, r_macro, f_macro

def image_process(image, transform):
    image = transform(image)
    return image

def convert_mm_examples_to_features(examples, label_list, tokenizer, img):
    max_seq_length = 77
    max_hashtag_length = 12
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225))])

    for (ex_index, example) in enumerate(examples):

        hashtags = []
        tokens = []

        sent = example.text.split()
        i = 0
        while i < len(sent):
            if sent[i] == "#" and i < len(sent) - 1:
                while sent[i] == "#" and i < len(sent) - 1:
                    i += 1
                if sent[i] != "#":
                    temp = wordninja.split(sent[i])
                    for _ in temp:
                        hashtags.append(_)
            else:
                if sent[i] != "#":
                    temp = wordninja.split(sent[i])
                    for _ in temp:
                        tokens.append(_)
                i += 1
        tokens = " ".join(tokens)
        hashtags = " ".join(hashtags) if len(hashtags) != 0 else "None"
        tokens = tokenizer.tokenize(tokens)
        hashtags = tokenizer.tokenize(hashtags)

        #####
        # image_text = None
        # image_text_dic = get_image_text()

        # if example.img_id in image_text_dic:
        #     image_text = list(image_text_dic[example.img_id])
        # else:
        #     image_text = ["None"]
        #####

        if len(tokens) > max_seq_length - 2:
            tokens = tokens[:(max_seq_length - 2)]
        if len(hashtags) > max_hashtag_length - 2:
            hashtags = hashtags[:(max_hashtag_length - 2)]

        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        added_input_mask = [1] * (len(input_ids) + 49)
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        added_input_mask += padding

        hashtags = ["[CLS]"] + hashtags + ["[SEP]"]
        hashtag_input_ids = tokenizer.convert_tokens_to_ids(hashtags)
        hashtag_input_mask = [1] * len(hashtag_input_ids)
        hashtag_padding = [0] * (max_hashtag_length - len(hashtag_input_ids))
        hashtag_input_ids += hashtag_padding
        hashtag_input_mask += hashtag_padding
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length

        assert len(hashtag_input_ids) == max_hashtag_length
        assert len(hashtag_input_mask) == max_hashtag_length
        label_id = label_map[example.label]

        image = transform(img)  # 3*224*224
        if ex_index % 1000 == 0:
            st.info("processed image num: " + str(ex_index) + " **********")

        features.append(MMInputFeatures(input_ids=input_ids,
                                        input_mask=input_mask,
                                        added_input_mask=added_input_mask,
                                        img_feat=image,
                                        hashtag_input_ids=hashtag_input_ids,
                                        hashtag_input_mask=hashtag_input_mask,
                                        label_id=label_id))
    
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_added_input_mask = torch.tensor([f.added_input_mask for f in features], dtype=torch.long)
    all_img_feats = torch.stack([f.img_feat for f in features])
    all_hashtag_input_ids = torch.tensor([f.hashtag_input_ids for f in features], dtype=torch.long)
    all_hashtag_input_mask = torch.tensor([f.hashtag_input_mask for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    return all_input_ids, all_input_mask, all_added_input_mask, all_img_feats, all_hashtag_input_ids, all_hashtag_input_mask, all_label_ids


def create_example(text, label):
        """Creates examples for the training and dev sets."""
        examples = []
        inputted_test_sample = [[1, text, label]]
        for line in inputted_test_sample:
            tmpLS = line[1].split()
            if "sarcasm" in tmpLS:
                continue
            if "sarcastic" in tmpLS:
                continue
            if "reposting" in tmpLS:
                continue
            if "<url>" in tmpLS:
                continue
            if "joke" in tmpLS:
                continue
            if "humour" in tmpLS:
                continue
            if "humor" in tmpLS:
                continue
            if "jokes" in tmpLS:
                continue
            if "irony" in tmpLS:
                continue
            if "ironic" in tmpLS:
                continue
            if "exgag" in tmpLS:
                continue
            img_id = 1
            text = line[1]
            examples.append(InputExample(text=text, img_id=img_id, label=label))
        return examples

def test_model(text, image, label):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base", do_lower_case=True)

    if torch.cuda.is_available():
        st.info("************** Using: " + torch.cuda.get_device_name(0) + " ******************")
    else:
        st.info("************** Using CPU ******************")

    model = MsdBERT()
    net = torchvision.models.resnet152(pretrained=True)
    encoder = myResnet(net).to(device)

    model.load_state_dict(torch.load(model_model, map_location=device))
    encoder.load_state_dict(torch.load(model_encoder, map_location=device) )
    model.to(device)
    encoder.to(device)
    model.eval()
    encoder.eval()

    test_examples = create_example(text, label)
    test_features = convert_mm_examples_to_features(test_examples, [0, 1], tokenizer, image)
    test_input_ids, test_input_mask, test_added_input_mask, test_img_feats, \
    test_hashtag_input_ids, test_hashtag_input_mask, test_label_ids = test_features

    test_data = TensorDataset(test_input_ids, test_input_mask, test_added_input_mask, test_img_feats, \
                            test_hashtag_input_ids, test_hashtag_input_mask, test_label_ids)
    test_dataloader = DataLoader(test_data, sampler=SequentialSampler(test_data), batch_size=32)

    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    true_label_list = []
    pred_label_list = []


    for batch in test_dataloader:   
        batch = tuple(t.to(device) for t in batch)
        test_input_ids, test_input_mask, test_added_input_mask, test_img_feats, \
        test_hashtag_input_ids, test_hashtag_input_mask, test_label_ids = batch
        imgs_f, img_mean, test_img_att = encoder(test_img_feats)
        with torch.no_grad():
            tmp_eval_loss = model(test_input_ids,test_img_att, test_input_mask, test_added_input_mask, \
                                    test_hashtag_input_ids, test_hashtag_input_mask, test_label_ids)
            logits = model(test_input_ids, test_img_att, test_input_mask, test_added_input_mask, \
                            test_hashtag_input_ids, test_hashtag_input_mask)
        logits = logits.detach().cpu().numpy()
        label_ids = test_label_ids.to('cpu').numpy()
        true_label_list.append(label_ids)
        pred_label_list.append(logits)
        tmp_eval_accuracy = accuracy(logits, label_ids)

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += test_input_ids.size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples
    loss =  None

    true_label = np.concatenate(true_label_list)
    pred_outputs = np.concatenate(pred_label_list)

    logger.info(f"true labels: {true_label_list}")
    logger.info(f"pred labels: {pred_label_list}")

    precision, recall, F_score = macro_f1(true_label, pred_outputs)
    result = {'test_loss': eval_loss,
                'test_accuracy': eval_accuracy,
                'precision': precision,
                'recall': recall,
                'f_score': F_score,
                'train_loss': loss}
                

    pred_label = np.argmax(pred_outputs, axis=-1)

    st.success(result)
    st.success(pred_label)
    # fout_p = open(os.path.join(args.output_dir, "pred.txt"), 'w')
    # fout_t = open(os.path.join(args.output_dir, "true.txt"), 'w')
    # for i in range(len(pred_label)):
    #     attstr = str(pred_label[i])
    #     fout_p.write(attstr + '\n')
    # for i in range(len(true_label)):
    #     attstr = str(true_label[i])
    #     fout_t.write(attstr + '\n')

    # fout_p.close()
    # fout_t.close()

    # output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    # with open(output_eval_file, "w") as writer:
    #     logger.info("***** Test Eval results *****")
    #     for key in sorted(result.keys()):
    #         logger.info("  %s = %s", key, str(result[key]))
    #         writer.write("%s = %s\n" % (key, str(result[key])))

