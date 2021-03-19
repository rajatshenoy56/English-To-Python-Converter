from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from langdetect import detect
import speech_recognition as sr
import io
# Create your views here.
# import collections
# import os
# import pathlib
# import re
# import string
# import sys
# import tempfile
# import time
# import json

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# !pip install -q tensorflow-text-nightly
# !pip install -q tensorflow_datasets
# !pip install -q tf-nightly

# !pip install tensorflow
# import tensorflow_text as text
# import tensorflow as tf
def detect_en(text):
    if detect(text) == "en":
        return True
    else:
        return False

@csrf_exempt
def text(request):
    if request.method == "POST":
        if detect_en(str(request.body)) == True:
            return JsonResponse({'lang':"English"})
        else:
            return JsonResponse({'lang':'Non English'})
    else:
        return render(request, 'input/text_input.html')

@csrf_exempt
def file(request):
    if request.method == "POST":
        print(request.body)
        print(request.FILES['File'])
        str_file = ""
        for chunk in request.FILES['File'].chunks():
            str_file += str(chunk)
        if detect_en(str_file) == True:
            return JsonResponse({'lang':"English"})
        else:
            return JsonResponse({'lang':'Non English'})


# tf.keras.utils.get_file(
#     f"en_py_converter.zip",
#     f"https://storage.googleapis.com/download.tensorflow.org/models/en_py_converter.zip",
#     cache_dir='.', cache_subdir='', extract=True
# )
# model_name="en_py_converter"
# tokenizers = tf.saved_model.load(model_name)

# def get_angles(pos, i, d_model):
#   angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
#   return pos * angle_rates

# def positional_encoding(position, d_model):
#   angle_rads = get_angles(np.arange(position)[:, np.newaxis],
#                           np.arange(d_model)[np.newaxis, :],
#                           d_model)
  
#   # apply sin to even indices in the array; 2i
#   angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  
#   # apply cos to odd indices in the array; 2i+1
#   angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
#   pos_encoding = angle_rads[np.newaxis, ...]
    
#   return tf.cast(pos_encoding, dtype=tf.float32)


# def create_padding_mask(seq):
#   seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
  
#   # add extra dimensions to add the padding
#   # to the attention logits.
#   return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


# def create_look_ahead_mask(size):
#   mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
#   return mask  # (seq_len, seq_len)


# def scaled_dot_product_attention(q, k, v, mask):
#   """Calculate the attention weights.
#   q, k, v must have matching leading dimensions.
#   k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
#   The mask has different shapes depending on its type(padding or look ahead) 
#   but it must be broadcastable for addition.
  
#   Args:
#     q: query shape == (..., seq_len_q, depth)
#     k: key shape == (..., seq_len_k, depth)
#     v: value shape == (..., seq_len_v, depth_v)
#     mask: Float tensor with shape broadcastable 
#           to (..., seq_len_q, seq_len_k). Defaults to None.
    
#   Returns:
#     output, attention_weights
#   """

#   matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
  
#   # scale matmul_qk
#   dk = tf.cast(tf.shape(k)[-1], tf.float32)
#   scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

#   # add the mask to the scaled tensor.
#   if mask is not None:
#     scaled_attention_logits += (mask * -1e9)  

#   # softmax is normalized on the last axis (seq_len_k) so that the scores
#   # add up to 1.
#   attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

#   output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

#   return output, attention_weights

# def print_out(q, k, v):
#   temp_out, temp_attn = scaled_dot_product_attention(
#       q, k, v, None)
#   print ('Attention weights are:')
#   print (temp_attn)
#   print ('Output is:')
#   print (temp_out)


# class MultiHeadAttention(tf.keras.layers.Layer):
#   def __init__(self, d_model, num_heads):
#     super(MultiHeadAttention, self).__init__()
#     self.num_heads = num_heads
#     self.d_model = d_model
    
#     assert d_model % self.num_heads == 0
    
#     self.depth = d_model // self.num_heads
    
#     self.wq = tf.keras.layers.Dense(d_model)
#     self.wk = tf.keras.layers.Dense(d_model)
#     self.wv = tf.keras.layers.Dense(d_model)
    
#     self.dense = tf.keras.layers.Dense(d_model)
        
#   def split_heads(self, x, batch_size):
#     """Split the last dimension into (num_heads, depth).
#     Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
#     """
#     x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
#     return tf.transpose(x, perm=[0, 2, 1, 3])
    
#   def call(self, v, k, q, mask):
#     batch_size = tf.shape(q)[0]
    
#     q = self.wq(q)  # (batch_size, seq_len, d_model)
#     k = self.wk(k)  # (batch_size, seq_len, d_model)
#     v = self.wv(v)  # (batch_size, seq_len, d_model)
    
#     q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
#     k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
#     v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
    
#     # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
#     # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
#     scaled_attention, attention_weights = scaled_dot_product_attention(
#         q, k, v, mask)
    
#     scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

#     concat_attention = tf.reshape(scaled_attention, 
#                                   (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

#     output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        
#     return output, attention_weights

# def point_wise_feed_forward_network(d_model, dff):
#   return tf.keras.Sequential([
#       tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
#       tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
#   ])


# class EncoderLayer(tf.keras.layers.Layer):
#   def __init__(self, d_model, num_heads, dff, rate=0.1):
#     super(EncoderLayer, self).__init__()

#     self.mha = MultiHeadAttention(d_model, num_heads)
#     self.ffn = point_wise_feed_forward_network(d_model, dff)

#     self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
#     self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
#     self.dropout1 = tf.keras.layers.Dropout(rate)
#     self.dropout2 = tf.keras.layers.Dropout(rate)
    
#   def __call__(self, x, training, mask):

#     attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
#     attn_output = self.dropout1(attn_output, training=training)
#     out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
    
#     ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
#     ffn_output = self.dropout2(ffn_output, training=training)
#     out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
    
#     return out2


# class DecoderLayer(tf.keras.layers.Layer):
#   def __init__(self, d_model, num_heads, dff, rate=0.1):
#     super(DecoderLayer, self).__init__()

#     self.mha1 = MultiHeadAttention(d_model, num_heads)
#     self.mha2 = MultiHeadAttention(d_model, num_heads)

#     self.ffn = point_wise_feed_forward_network(d_model, dff)
 
#     self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
#     self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
#     self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
#     self.dropout1 = tf.keras.layers.Dropout(rate)
#     self.dropout2 = tf.keras.layers.Dropout(rate)
#     self.dropout3 = tf.keras.layers.Dropout(rate)
    
    
#   def call(self, x, enc_output, training, 
#            look_ahead_mask, padding_mask):
#     # enc_output.shape == (batch_size, input_seq_len, d_model)

#     attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
#     attn1 = self.dropout1(attn1, training=training)
#     out1 = self.layernorm1(attn1 + x)
    
#     attn2, attn_weights_block2 = self.mha2(
#         enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
#     attn2 = self.dropout2(attn2, training=training)
#     out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)
    
#     ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
#     ffn_output = self.dropout3(ffn_output, training=training)
#     out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)
    
#     return out3, attn_weights_block1, attn_weights_block2


# class Encoder(tf.keras.layers.Layer):
#   def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
#                maximum_position_encoding, rate=0.1):
#     super(Encoder, self).__init__()

#     self.d_model = d_model
#     self.num_layers = num_layers
    
#     self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
#     self.pos_encoding = positional_encoding(maximum_position_encoding, 
#                                             self.d_model)
    
    
#     self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) 
#                        for _ in range(num_layers)]
  
#     self.dropout = tf.keras.layers.Dropout(rate)
        
#   def call(self, x, training, mask):

#     seq_len = tf.shape(x)[1]
    
#     # adding embedding and position encoding.
#     x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
#     x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
#     x += self.pos_encoding[:, :seq_len, :]

#     x = self.dropout(x, training=training)
    
#     for i in range(self.num_layers):
#       x = self.enc_layers[i](x, training, mask)
    
#     return x  # (batch_size, input_seq_len, d_model)

# class Decoder(tf.keras.layers.Layer):
#   def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
#                maximum_position_encoding, rate=0.1):
#     super(Decoder, self).__init__()

#     self.d_model = d_model
#     self.num_layers = num_layers
    
#     self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
#     self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
    
#     self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) 
#                        for _ in range(num_layers)]
#     self.dropout = tf.keras.layers.Dropout(rate)
    
#   def call(self, x, enc_output, training, 
#            look_ahead_mask, padding_mask):

#     seq_len = tf.shape(x)[1]
#     attention_weights = {}
    
#     x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
#     x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
#     x += self.pos_encoding[:, :seq_len, :]
    
#     x = self.dropout(x, training=training)

#     for i in range(self.num_layers):
#       x, block1, block2 = self.dec_layers[i](x, enc_output, training,
#                                              look_ahead_mask, padding_mask)
      
#       attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
#       attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
    
#     # x.shape == (batch_size, target_seq_len, d_model)
#     return x, attention_weights

# class Transformer(tf.keras.Model):
#   def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
#                target_vocab_size, pe_input, pe_target, rate=0.1):
#     super(Transformer, self).__init__()

#     self.tokenizer = Encoder(num_layers, d_model, num_heads, dff, 
#                            input_vocab_size, pe_input, rate)

#     self.decoder = Decoder(num_layers, d_model, num_heads, dff, 
#                            target_vocab_size, pe_target, rate)

#     self.final_layer = tf.keras.layers.Dense(target_vocab_size)
    
#   def call(self, inp, tar, training, enc_padding_mask, 
#            look_ahead_mask, dec_padding_mask):

#     enc_output = self.tokenizer(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
    
#     # dec_output.shape == (batch_size, tar_seq_len, d_model)
#     dec_output, attention_weights = self.decoder(
#         tar, enc_output, training, look_ahead_mask, dec_padding_mask)
    
#     final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
    
#     return final_output, attention_weights

# num_layers = 4
# d_model = 128
# dff = 512
# num_heads = 8
# dropout_rate = 0.1

# transformer = Transformer(
#     num_layers=num_layers,
#     d_model=d_model,
#     num_heads=num_heads,
#     dff=dff,
#     input_vocab_size=tokenizers.en.get_vocab_size(),
#     target_vocab_size=tokenizers.py.get_vocab_size(), 
#     pe_input=1000, 
#     pe_target=1000,
#     rate=dropout_rate)

# latest = tf.train.latest_checkpoint("/content/drive/MyDrive/checkpoints")
# print(latest)
# temp_input = tf.random.uniform((64, 38), dtype=tf.int64, minval=0, maxval=200)
# temp_target = tf.random.uniform((64, 36), dtype=tf.int64, minval=0, maxval=200)

# fn_out, _ = transformer(temp_input, temp_target, training=False, 
#                                enc_padding_mask=None, 
#                                look_ahead_mask=None,
#                                dec_padding_mask=None)

# fn_out.shape  # (batch_size, tar_seq_len, target_vocab_size)
# transformer.load_weights(latest)

# def create_masks(inp, tar):
#   # Encoder padding mask
#   enc_padding_mask = create_padding_mask(inp)
  
#   # Used in the 2nd attention block in the decoder.
#   # This padding mask is used to mask the encoder outputs.
#   dec_padding_mask = create_padding_mask(inp)
  
#   # Used in the 1st attention block in the decoder.
#   # It is used to pad and mask future tokens in the input received by 
#   # the decoder.
#   look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
#   dec_target_padding_mask = create_padding_mask(tar)
#   combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
  
#   return enc_padding_mask, combined_mask, dec_padding_mask


# def evaluate(sentence, max_length=60):
#   # inp sentence is portuguese, hence adding the start and end token
#   sentence = tf.convert_to_tensor([sentence])
#   sentence = tokenizers.en.tokenize(sentence).to_tensor()

#   encoder_input = sentence
  
#   # as the target is english, the first word to the transformer should be the
#   # english start token.
#   start, end = tokenizers.py.tokenize([''])[0]
#   output = tf.convert_to_tensor([start])
#   output = tf.expand_dims(output, 0)
    
#   for i in range(max_length):
#     enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
#         encoder_input, output)
  
#     # predictions.shape == (batch_size, seq_len, vocab_size)
#     predictions, attention_weights = transformer(encoder_input, 
#                                                  output,
#                                                  False,
#                                                  enc_padding_mask,
#                                                  combined_mask,
#                                                  dec_padding_mask)
    
#     # select the last word from the seq_len dimension
#     predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

#     predicted_id = tf.argmax(predictions, axis=-1)

#     # concatentate the predicted_id to the output which is given to the decoder
#     # as its input.
#     output = tf.concat([output, predicted_id], axis=-1)
    
#     # return the result if the predicted_id is equal to the end token
#     if predicted_id == end:
#       break

#   # output.shape (1, tokens)
#   text = tokenizers.py.detokenize(output)[0] # shape: ()
  
#   tokens = tokenizers.py.lookup(output)[0]
#   return text, tokens, attention_weights

# def print_translation(sentence, tokens, ground_truth):
#   print(f'{"Input:":15s}: {sentence}')
#   print(f'{"Prediction":15s}: {tokens.numpy().decode("utf-8")}')
#   print(f'{"Ground truth":15s}: {ground_truth}')
# sentence = "for every a in lista,"
# ground_truth = "for a in lista:"

# translated_text, translated_tokens, attention_weights = evaluate(sentence)
# print_translation(sentence, translated_text, ground_truth)
# # print(attention_weights)