
from nltk.tokenize import RegexpTokenizer
from miscc.config import cfg, cfg_from_file
from model import RNN_ENCODER, CNN_ENCODER, G_NET
from miscc.utils import build_super_images, build_super_images2
from miscc.utils import mkdir_p
from torch.autograd import Variable
from PIL import Image

import numpy as np
import torch
import pickle
import argparse
import pprint

def parse_args():
    parser = argparse.ArgumentParser(description='Train a ControlGAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/train_bird.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=-1)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args

def generate(wordtoix, sent):
    data_dic = {}
    sent = sent.replace("\ufffd\ufffd", " ")
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sent.lower())
    if len(tokens) == 0:
        print('sent', sent)
        return
    
    rev = []
    captions =  []
    cap_lens = []
    for t in tokens:
        t = t.encode('ascii','ignore').decode('ascii')
        if len(t) > 0 and t in wordtoix:
            rev.append(wordtoix[t])
            
    captions.append(rev)
    cap_lens.append(len(rev))    
    
    max_len = np.max(cap_lens)   
    sorted_indices = np.argsort(cap_lens)[::-1]
    cap_lens = np.asarray(cap_lens)
    cap_lens = cap_lens[sorted_indices]
    cap_array=np.zeros((len(captions), max_len), dtype = 'int64')
    for i in range(len(captions)):
        idx = sorted_indices[i]
        cap = captions[idx]
        c_len = len(cap)
        cap_array[i, :c_len] = cap
        key = "generated"
        data_dic[key] = [cap_array , cap_lens, sorted_indices]
    # gen_example(data_dic)
    return data_dic


if __name__ == "__main__":
    text = input("Enter What you want to generate : ")
    
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id
    else:
        cfg.CUDA = False

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    print('Using config:')
    pprint.pprint(cfg)
        
    with open('vocabulary.pkl', 'rb') as f:
        vocab = pickle.load(f)
        
    
    
    ixtoword = vocab['ixtoword']
    wordtoix = vocab['wordtoix']
    n_words = len(ixtoword)
    text_encoder = RNN_ENCODER(n_words, nhidden = 256)
    state_dict = torch.load('../DAMSMencoders/bird/text_encoder.pth', map_location=lambda storage, loc: storage)
    text_encoder.load_state_dict(state_dict)
    print('Load text encoder from:', '../DAMSMencoders/bird/text_encoder.pth')
    text_encoder = text_encoder.cuda()
    text_encoder.eval()
    
    netG = G_NET()
    s_tmp = cfg.TRAIN.NET_G[:cfg.TRAIN.NET_G.rfind('.pth')]
    model_dir = cfg.TRAIN.NET_G
    print(model_dir)
    state_dict = \
        torch.load(model_dir, map_location=lambda storage, loc: storage)
    netG.load_state_dict(state_dict)
    print('Load G from: ', model_dir)
    netG.cuda()
    netG.eval()
    
    data_dic = generate(wordtoix, text)
    print(data_dic)
    
    for key in data_dic:
        save_dir = 'output'
        mkdir_p(save_dir)
        captions, cap_lens, sorted_indices = data_dic[key]

        batch_size = captions.shape[0]
        nz = cfg.GAN.Z_DIM
        captions = Variable(torch.from_numpy(captions), volatile=True)
        cap_lens = Variable(torch.from_numpy(cap_lens), volatile=True)

        captions = captions.cuda()
        cap_lens = cap_lens.cuda()
        for i in range(1): 
            noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
            noise = noise.cuda()

            hidden = text_encoder.init_hidden(batch_size)
            words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
            mask = (captions == 0)

            noise.data.normal_(0, 1)
            fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask)

            cap_lens_np = cap_lens.cpu().data.numpy()
            for j in range(batch_size):
                save_name = '%s/%d_s_%d' % (save_dir, i, sorted_indices[j])
                for k in range(len(fake_imgs)):
                    im = fake_imgs[k][j].data.cpu().numpy()
                    im = (im + 1.0) * 127.5
                    im = im.astype(np.uint8)
                    im = np.transpose(im, (1, 2, 0))
                    im = Image.fromarray(im)
                    fullpath = '%s_g%d.png' % (save_name, k)
                    im.save(fullpath)

                for k in range(len(attention_maps)):
                    if len(fake_imgs) > 1:
                        im = fake_imgs[k + 1].detach().cpu()
                    else:
                        im = fake_imgs[0].detach().cpu()
                    attn_maps = attention_maps[k]
                    att_sze = attn_maps.size(2)
                    img_set, sentences = \
                        build_super_images2(im[j].unsqueeze(0),
                                            captions[j].unsqueeze(0),
                                            [cap_lens_np[j]], ixtoword,
                                            [attn_maps[j]], att_sze)
                    if img_set is not None:
                        im = Image.fromarray(img_set)
                        fullpath = '%s_a%d.png' % (save_name, k)
                        im.save(fullpath)