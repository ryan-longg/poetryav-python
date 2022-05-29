# @title Load libraries and variables

import argparse
from cgitb import text
import math
from pathlib import Path
import sys
from os.path import exists
import os
import time
import uuid
import freesound
import random
import requests
import ffmpeg
from pydub import AudioSegment
from moviepy.editor import *

sys.path.insert(1, '/content/taming-transformers')
from IPython import display
from base64 import b64encode
from omegaconf import OmegaConf
from PIL import Image
from taming.models import cond_transformer, vqgan
import taming.modules 
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm.notebook import tqdm
import nltk

from CLIP import clip
import kornia.augmentation as K
import numpy as np
import imageio
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
from urllib.request import urlopen

FRAMES_PER_ANIMATION_SEGMENT = 200

def sinc(x):
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))


def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x/a), x.new_zeros([]))
    return out / out.sum()


def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]


def resample(input, size, align_corners=True):
    n, c, h, w = input.shape
    dh, dw = size

    input = input.view([n * c, 1, h, w])

    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), 'reflect')
        input = F.conv2d(input, kernel_h[None, None, :, None])

    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
        input = F.conv2d(input, kernel_w[None, None, None, :])

    input = input.view([n, c, h, w])
    return F.interpolate(input, size, mode='bicubic', align_corners=align_corners)


class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)


replace_grad = ReplaceGrad.apply


class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_in):
        input, = ctx.saved_tensors
        return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None


clamp_with_grad = ClampWithGrad.apply


def vector_quantize(x, codebook):
    d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
    indices = d.argmin(-1)
    x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
    return replace_grad(x_q, x)


class Prompt(nn.Module):
    def __init__(self, embed, weight=1., stop=float('-inf')):
        super().__init__()
        self.register_buffer('embed', embed)
        self.register_buffer('weight', torch.as_tensor(weight))
        self.register_buffer('stop', torch.as_tensor(stop))

    def forward(self, input):
        input_normed = F.normalize(input.unsqueeze(1), dim=2)
        embed_normed = F.normalize(self.embed.unsqueeze(0), dim=2)
        dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        dists = dists * self.weight.sign()
        return self.weight.abs() * replace_grad(dists, torch.maximum(dists, self.stop)).mean()


def parse_prompt(prompt):
    vals = prompt.rsplit(':', 2)
    vals = vals + ['', '1', '-inf'][len(vals):]
    return vals[0], float(vals[1]), float(vals[2])


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

        self.augs = nn.Sequential(
            # K.RandomHorizontalFlip(p=0.5),
            # K.RandomVerticalFlip(p=0.5),
            # K.RandomSolarize(0.01, 0.01, p=0.7),
            # K.RandomSharpness(0.3,p=0.4),
            # K.RandomResizedCrop(size=(self.cut_size,self.cut_size), scale=(0.1,1),  ratio=(0.75,1.333), cropping_mode='resample', p=0.5),
            # K.RandomCrop(size=(self.cut_size,self.cut_size), p=0.5),
            K.RandomAffine(degrees=15, translate=0.1, p=0.7, padding_mode='border'),
            K.RandomPerspective(0.7,p=0.7),
            K.ColorJitter(hue=0.1, saturation=0.1, p=0.7),
            K.RandomErasing((.1, .4), (.3, 1/.3), same_on_batch=True, p=0.7),
            
)
        self.noise_fac = 0.1
        self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
        self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        
        for _ in range(self.cutn):

            # size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            # offsetx = torch.randint(0, sideX - size + 1, ())
            # offsety = torch.randint(0, sideY - size + 1, ())
            # cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            # cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))

            # cutout = transforms.Resize(size=(self.cut_size, self.cut_size))(input)
            
            cutout = (self.av_pool(input) + self.max_pool(input))/2
            cutouts.append(cutout)
        batch = self.augs(torch.cat(cutouts, dim=0))
        if self.noise_fac:
            facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch


def load_vqgan_model(config_path, checkpoint_path):
    config = OmegaConf.load(config_path)
    if config.model.target == 'taming.models.vqgan.VQModel':
        model = vqgan.VQModel(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == 'taming.models.vqgan.GumbelVQ':
        model = vqgan.GumbelVQ(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
        parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
        parent_model.eval().requires_grad_(False)
        parent_model.init_from_ckpt(checkpoint_path)
        model = parent_model.first_stage_model
    else:
        raise ValueError(f'unknown model type: {config.model.target}')
    del model.loss
    return model


def resize_image(image, out_size):
    ratio = image.size[0] / image.size[1]
    area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
    size = round((area * ratio)**0.5), round((area / ratio)**0.5)
    return image.resize(size, Image.LANCZOS)

def translateInputValues(prompt, init_image):
    texts = prompt
    width =  600#@param {type:"number"}
    height = 600#@param {type:"number"}
    model = "vqgan_imagenet_f16_16384" #@param ["vqgan_imagenet_f16_16384", "vqgan_imagenet_f16_1024", "vqgan_openimages_f16_8192", "wikiart_1024", "wikiart_16384", "coco", "faceshq", "sflckr"]
    images_interval =  50#@param {type:"number"}
    init_image = init_image#@param {type:"string"}
    target_images = ""#@param {type:"string"}
    seed = 42#@param {type:"number"}
    max_iterations = FRAMES_PER_ANIMATION_SEGMENT#@param {type:"number"}

    model_names={"vqgan_imagenet_f16_16384": 'ImageNet 16384',"vqgan_imagenet_f16_1024":"ImageNet 1024", 'vqgan_openimages_f16_8192':'OpenImages 8912',
                    "wikiart_1024":"WikiArt 1024", "wikiart_16384":"WikiArt 16384", "coco":"COCO-Stuff", "faceshq":"FacesHQ", "sflckr":"S-FLCKR"}
    name_model = model_names[model]     

    if seed == -1:
        seed = None
    if init_image == "None":
        init_image = None
    if target_images == "None" or not target_images:
        target_images = []
    else:
        target_images = target_images.split("|")
        target_images = [image.strip() for image in target_images]

    texts = [phrase.strip() for phrase in texts.split("|")]
    if texts == ['']:
        texts = []


    args = argparse.Namespace(
        prompts=texts,
        image_prompts=target_images,
        noise_prompt_seeds=[],
        noise_prompt_weights=[],
        size=[width, height],
        init_image=init_image,
        init_weight=0.,
        clip_model='ViT-B/32',
        vqgan_config=f'{model}.yaml',
        vqgan_checkpoint=f'{model}.ckpt',
        step_size=0.1,
        cutn=32,
        cut_pow=1.,
        display_freq=5,
        seed=seed
    )
    return args

def animateInput(args, imageNumber, randomID):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    texts = args.prompts
    target_images = args.image_prompts
    max_iterations = FRAMES_PER_ANIMATION_SEGMENT

    print('Using device:', device)
    if texts:
        print('Using texts:', texts)
    if target_images:
        print('Using image prompts:', target_images)
    if args.seed is None:
        seed = torch.seed()
    else:
        seed = args.seed
    torch.manual_seed(seed)
    print('Using seed:', seed)

    model = load_vqgan_model(args.vqgan_config, args.vqgan_checkpoint).to(device)
    perceptor = clip.load(args.clip_model, jit=False)[0].eval().requires_grad_(False).to(device)
    # clock=deepcopy(perceptor.visual.positional_embedding.data)
    # perceptor.visual.positional_embedding.data = clock/clock.max()
    # perceptor.visual.positional_embedding.data=clamp_with_grad(clock,0,1)

    cut_size = perceptor.visual.input_resolution

    f = 2**(model.decoder.num_resolutions - 1)
    make_cutouts = MakeCutouts(cut_size, args.cutn, cut_pow=args.cut_pow)

    toksX, toksY = args.size[0] // f, args.size[1] // f
    sideX, sideY = toksX * f, toksY * f

    if args.vqgan_checkpoint == 'vqgan_openimages_f16_8192.ckpt':
        e_dim = 256
        n_toks = model.quantize.n_embed
        z_min = model.quantize.embed.weight.min(dim=0).values[None, :, None, None]
        z_max = model.quantize.embed.weight.max(dim=0).values[None, :, None, None]
    else:
        e_dim = model.quantize.e_dim
        n_toks = model.quantize.n_e
        z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
        z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]
    # z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
    # z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]

    # normalize_imagenet = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                            std=[0.229, 0.224, 0.225])

    if args.init_image:
        if 'http' in args.init_image:
            img = Image.open(urlopen(args.init_image))
        else:
            img = Image.open(args.init_image)
            pil_image = img.convert('RGB')
            pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
            pil_tensor = TF.to_tensor(pil_image)
            z, *_ = model.encode(pil_tensor.to(device).unsqueeze(0) * 2 - 1)
    else:
        one_hot = F.one_hot(torch.randint(n_toks, [toksY * toksX], device=device), n_toks).float()
        # z = one_hot @ model.quantize.embedding.weight
        if args.vqgan_checkpoint == 'vqgan_openimages_f16_8192.ckpt':
            z = one_hot @ model.quantize.embed.weight
        else:
            z = one_hot @ model.quantize.embedding.weight
        z = z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2) 
        z = torch.rand_like(z)*2
    z_orig = z.clone()
    z.requires_grad_(True)
    opt = optim.Adam([z], lr=args.step_size)

    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                    std=[0.26862954, 0.26130258, 0.27577711])



    pMs = []

    for prompt in args.prompts:
        txt, weight, stop = parse_prompt(prompt)
        embed = perceptor.encode_text(clip.tokenize(txt).to(device)).float()
        pMs.append(Prompt(embed, weight, stop).to(device))

    for prompt in args.image_prompts:
        path, weight, stop = parse_prompt(prompt)
        img = Image.open(path)
        pil_image = img.convert('RGB')
        img = resize_image(pil_image, (sideX, sideY))
        batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(device))
        embed = perceptor.encode_image(normalize(batch)).float()
        pMs.append(Prompt(embed, weight, stop).to(device))

    for seed, weight in zip(args.noise_prompt_seeds, args.noise_prompt_weights):
        gen = torch.Generator().manual_seed(seed)
        embed = torch.empty([1, perceptor.visual.output_dim]).normal_(generator=gen)
        pMs.append(Prompt(embed, weight).to(device))

    def synth(z):
        if args.vqgan_checkpoint == 'vqgan_openimages_f16_8192.ckpt':
            z_q = vector_quantize(z.movedim(1, 3), model.quantize.embed.weight).movedim(3, 1)
        else:
            z_q = vector_quantize(z.movedim(1, 3), model.quantize.embedding.weight).movedim(3, 1)
        return clamp_with_grad(model.decode(z_q).add(1).div(2), 0, 1)

    @torch.no_grad()
    def checkin(i, losses):
        losses_str = ', '.join(f'{loss.item():g}' for loss in losses)
        tqdm.write(f'i: {i}, loss: {sum(losses).item():g}, losses: {losses_str}')
        out = synth(z)
        TF.to_pil_image(out[0].cpu()).save("progress" + randomID + ".png")
        display.display(display.Image("progress" + randomID + ".png"))

    def ascend_txt(i, j):
        out = synth(z)
        iii = perceptor.encode_image(normalize(make_cutouts(out))).float()
        
        result = []

        if args.init_weight:
            # result.append(F.mse_loss(z, z_orig) * args.init_weight / 2)
            result.append(F.mse_loss(z, torch.zeros_like(z_orig)) * ((1/torch.tensor(i*2 + 1))*args.init_weight) / 2)
        for prompt in pMs:
            result.append(prompt(iii))
        img = np.array(out.mul(255).clamp(0, 255)[0].cpu().detach().numpy().astype(np.uint8))[:,:,:]
        img = np.transpose(img, (1, 2, 0))
        print("writing image to path: " + './steps/' + str(FRAMES_PER_ANIMATION_SEGMENT*j + i) + '.png')
        imageio.imwrite("./steps/" + str(FRAMES_PER_ANIMATION_SEGMENT*j + i) + randomID + ".png", np.array(img))
        if(i==(FRAMES_PER_ANIMATION_SEGMENT-1)):
            imageio.imwrite("./images/" + str(j) + randomID + ".png", np.array(img))

        return result

    def train(i, j):
        opt.zero_grad()
        lossAll = ascend_txt(i, j)
        if i % args.display_freq == 0:
            checkin(i, lossAll)
        
        loss = sum(lossAll)
        loss.backward()
        opt.step()
        with torch.no_grad():
            z.copy_(z.maximum(z_min).minimum(z_max))

    i = 0
    j = imageNumber
    try:
        with tqdm() as pbar:
            while True:
                train(i, j)
                if i == (max_iterations - 1):
                    print("BREAKING FROM MAX ITER")
                    break
                i += 1
                pbar.update()
    except KeyboardInterrupt:
        pass

def generateAnimation(numImages, duration, randomID):
    init_frame = 1 #This is the frame where the video will start
    last_frame = FRAMES_PER_ANIMATION_SEGMENT*numImages #You can change i to the number of the last frame you want to generate. It will raise an error if that number of frames does not exist.

    min_fps = 10
    max_fps = 60

    total_frames = last_frame-init_frame

    length = duration #Desired time of the video in seconds

    frames = []
    cwd = os.getcwd()
    for i in range(init_frame,last_frame): #
        frames.append(imageio.imread(str(cwd) + "/steps/"+ str(i) + randomID + '.png'))

    fps = np.clip(total_frames/length,min_fps,max_fps)

    movWriter = imageio.get_writer("./output/animation"+ randomID +".mp4", fps=fps)
    for frame in frames:
        movWriter.append_data(frame)
    movWriter.close()
    return fps

def is_noun(pos): return pos[:2] == 'NN' or pos[:2] == 'NNP' or pos[:2] == 'NNPS' or pos[:2] == 'NNS'
def is_adj(pos): return pos[:2] == 'JJ'
def is_verb(pos): return pos[:2] == 'VBP' or pos[:2] == 'VB'
def is_adverb(pos): return pos[:2] == 'VBP' or pos[:2] == 'RB'


def getDescriptiveText(text):
    # do some NLP
    tokenized = nltk.word_tokenize(text)
    descriptive = [(word, is_noun(pos)) for (word, pos) in nltk.pos_tag(tokenized) if is_noun(
        pos) or is_adj(pos)]
    return descriptive

def saveSoundsForPhrases(phrases, randomID):
    freeSoundClient = freesound.FreesoundClient()
    freeSoundClient.set_token("MLl1qjimzI7R9cFMRI1yhnlLefzWJsleHpfRS0S4","token")

    soundURLs = []
    for phrase in phrases:
        phraseWords = phrase.split(" ")
        wordToSonify = phraseWords[len(phraseWords) - 1]
        resultsObj = freeSoundClient.text_search(query=wordToSonify,fields="previews",filter="duration:[20 TO 120]")
        sound = random.choice(resultsObj.results)
        # for demo purposes to get a consistent result
        # sound = resultsObj.results[0]
        hqMP3Preview = sound["previews"]["preview-hq-mp3"]
        soundURLs.append(hqMP3Preview)
        print((hqMP3Preview, wordToSonify))

    for (index, url) in enumerate(soundURLs):
        r = requests.get(url, allow_redirects=True)
        f = open("./sounds/" + str(index) + randomID + ".mp3", 'wb').write(r.content)

    return soundURLs

def mergeSounds(imageCount, randomID):
    merged = AudioSegment.empty()
    for i in range(0,imageCount):
        sound = AudioSegment.from_mp3("./sounds/"+ str(i) + randomID + ".mp3")
        print("./sounds/"+ str(i) + randomID +".mp3")
        # clippedDuration = 20
        # if(i == 0):
        #     clippedDuration = 15
        clipped = sound[0:(15*1000)]

        merged += clipped
        # else:
        #     merged.append(clipped, crossfade=5000)
    merged.export("./output/audio" + randomID + ".mp3", format="mp3")

def combineAnimationAndSound(randomID, numImages):
    my_clip = VideoFileClip("./output/animation"+ randomID +".mp4").subclip(0, numImages*15)
    audio_background = AudioFileClip("./output/audio"+ randomID +".mp3").subclip(0, numImages*15)
    combined = my_clip.set_audio(audio_background)
    combined.write_videofile("./output/combined"+ randomID +".mp4",codec='libx264', audio_codec='aac', temp_audiofile='temp-audio.m4a',remove_temp=True)



def animateText(textIn):
    randomID = str(uuid.uuid1())
    print(textIn)
    print("input text: " + textIn + "\n")
    descriptive = getDescriptiveText(textIn)

    descriptivePhrases = []
    phrase = []
    count = 0
    for (word, isNoun) in descriptive:
        phrase.append(word)
        count += 1
        if((isNoun and len(phrase) > 1)  or count == 10):
            descriptivePhrases.append(" ".join(phrase))
            count = 0
            phrase = []


    print("descriptive phrases:")
    print(descriptivePhrases)
    if(len(descriptivePhrases) == 0):
        print("please input more descriptive text, we could not make an animation from your input")
        return
    saveSoundsForPhrases(descriptivePhrases, randomID)
    
    imageCount = 5
    if (imageCount <= 5):
        imageCount = len(descriptivePhrases)
    
    for i in range(0, imageCount):
        prompt = descriptivePhrases[i]
        print("ANIMATING DESCRIPTIVE PHRASE:")
        print(prompt)
        cwd = os.getcwd()
        pathToLastImage = str(cwd) + "/steps/" + str(i*FRAMES_PER_ANIMATION_SEGMENT) + randomID + ".png"
        init_image = ""
        if(exists(pathToLastImage)):
            init_image = pathToLastImage

        args = translateInputValues(prompt, init_image)
        print("image number " + str(i))
        print(prompt)
        animateInput(args, i, randomID)
    
    print("GENERATING ANIMATION")
    animation = generateAnimation(imageCount, 15*imageCount, randomID)
    # animation = generateAnimation(1, 15, randomID)
    print("MERGING SOUNDS")
    sound = mergeSounds(len(descriptivePhrases), randomID)
    # sound = mergeSounds(4, "cc255d3a-df21-11ec-ad9d-c4b301d7c335")
    print("COMBINING SOUNDS")
    combined = combineAnimationAndSound(randomID, imageCount)
    # do something with combined 

    #remove files
    if(os.path.exists("./output/animation" + randomID + ".mp4")): 
     os.remove("./output/animation" + randomID + ".mp4")
    if(os.path.exists("./output/audio" + randomID + ".mp3")): 
     os.remove("./output/audio" + randomID + ".mp3")
    if(os.path.exists("./progress" + randomID + ".png")): 
     os.remove("./progress" + randomID + ".png")
    for i in range(0, imageCount): 
        if(os.path.exists("./images/" + str(i) + randomID + ".png")): 
            os.remove("./images/" + str(i) + randomID + ".png")
        if(os.path.exists("./sounds/" + str(i) + randomID + ".mp3")): 
            os.remove("./sounds/" + str(i) + randomID + ".mp3")
    for i in range(0, imageCount*FRAMES_PER_ANIMATION_SEGMENT):
        if(os.path.exists("./steps/" + str(i) + randomID + ".png")): 
            os.remove("./steps/" + str(i) + randomID + ".png")
    
