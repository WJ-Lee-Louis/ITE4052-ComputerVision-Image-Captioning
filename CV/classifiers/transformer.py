import numpy as np
import copy

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from ..coco_utils import decode_captions
from ..image_utils import image_from_url

from ..transformer_layers import *


class CaptioningTransformer(nn.Module):
    """
    A CaptioningTransformer produces captions from image features using a
    Transformer decoder.

    The Transformer receives input vectors of size D, has a vocab size of V,
    works on sequences of length T, uses word vectors of dimension W, and
    operates on minibatches of size N.
    """
    def __init__(self, word_to_idx, input_dim, wordvec_dim, num_heads=4,
                 num_layers=2, max_length=50):
        """
        Construct a new CaptioningTransformer instance.

        Inputs:
        - word_to_idx: A dictionary giving the vocabulary. It contains V entries.
          and maps each string to a unique integer in the range [0, V).
        - input_dim: Dimension D of input image feature vectors.
        - wordvec_dim: Dimension W of word vectors.
        - num_heads: Number of attention heads.
        - num_layers: Number of transformer layers.
        - max_length: Max possible sequence length.
        """
        super().__init__()

        vocab_size = len(word_to_idx)
        self.vocab_size = vocab_size
        self._null = word_to_idx["<NULL>"]
        self._start = word_to_idx.get("<START>", None)
        self._end = word_to_idx.get("<END>", None)

        # Setup device - use CUDA if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.visual_projection = nn.Linear(input_dim, wordvec_dim)
        self.embedding = nn.Embedding(vocab_size, wordvec_dim, padding_idx=self._null)
        self.positional_encoding = PositionalEncoding(wordvec_dim, max_len=max_length)

        decoder_layer = TransformerDecoderLayer(input_dim=wordvec_dim, num_heads=num_heads)
        self.transformer = TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.apply(self._init_weights)

        self.output = nn.Linear(wordvec_dim, vocab_size)

        # Move the entire model to the appropriate device
        self.to(self.device)


    def _init_weights(self, module):
        """
        Initialize the weights of the network.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, features, captions):
        """
        Given image features and caption tokens, return a distribution over the
        possible tokens for each timestep. Note that since the entire sequence
        of captions is provided all at once, we mask out future timesteps.

        Inputs:
         - features: image features, of shape (N, D)
         - captions: ground truth captions, of shape (N, T)

        Returns:
         - scores: score for each token at each timestep, of shape (N, T, V)
        """
        N, T = captions.shape
        # Create a placeholder, to be overwritten by your code below.
        scores = torch.empty((N, T, self.vocab_size))
        ############################################################################
        # TODO: Implement the forward function for CaptionTransformer.             #
        # A few hints:                                                             #
        #  1) You first have to embed your caption and add positional              #
        #     encoding. You then have to project the image features into the same  #
        #     dimensions.                                                          #
        #  2) You have to prepare a mask (tgt_mask) for masking out the future     #
        #     timesteps in captions. torch.tril() function might help in preparing #
        #     this mask.                                                           #
        #  3) Finally, apply the decoder features on the text & image embeddings   #
        #     along with the tgt_mask. Project the output to scores per token      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        # 1. 캡션 임베딩과 위치 인코딩, 이미지 피처 투영
        # (N, T) -> (N, T, W)
        caption_embedding = self.embedding(captions)
        caption_embedding = self.positional_encoding(caption_embedding)
        
        # (N, D) -> (N, W) -> (N, 1, W)
        # 이미지 피처를 decoder의 memory로 사용하기 위해 차원 추가
        image_features = self.visual_projection(features).unsqueeze(1)
        
        # 2. 미래 타임스텝을 가리기 위한 마스크 생성
        # (T, T)
        tgt_mask = torch.tril(torch.ones(T, T, device=caption_embedding.device, dtype=caption_embedding.dtype))
        
        # 3. Transformer Decoder에 입력 전달
        # tgt: (N, T, W), memory: (N, 1, W) -> output: (N, T, W)
        transformer_output, attention_weights = self.transformer(tgt=caption_embedding, memory=image_features, tgt_mask=tgt_mask)
        
        # 4. 최종 점수 계산
        # (N, T, W) -> (N, T, V)
        scores = self.output(transformer_output)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return scores, attention_weights

    def sample(self, features, max_length=30):
        """
        Given image features, use greedy decoding to predict the image caption.

        Inputs:
         - features: image features, of shape (N, D)
         - max_length: maximum possible caption length

        Returns:
         - captions: captions for each example, of shape (N, max_length)
        """
        with torch.no_grad():
            features = torch.Tensor(features).to(self.device)
            N = features.shape[0]

            # Create an empty captions tensor (where all tokens are NULL).
            captions = self._null * np.ones((N, max_length), dtype=np.int32)

            # Create a partial caption, with only the start token.
            partial_caption = self._start * np.ones(N, dtype=np.int32)
            partial_caption = torch.LongTensor(partial_caption).to(self.device)
            # [N] -> [N, 1]
            partial_caption = partial_caption.unsqueeze(1)

            for t in range(max_length):

                # Predict the next token (ignoring all other time steps).
                output_logits, _ = self.forward(features, partial_caption)
                output_logits = output_logits[:, -1, :]

                # Choose the most likely word ID from the vocabulary.
                # [N, V] -> [N]
                word = torch.argmax(output_logits, axis=1)

                # Update our overall caption and our current partial caption.
                captions[:, t] = word.cpu().numpy()
                word = word.unsqueeze(1)
                partial_caption = torch.cat([partial_caption, word], dim=1)

            return captions

    def visualize_self_attention_heatmap(self, image_feature, image_url, data):
        print("Generating caption and self-attention heatmap...")

        self.eval()
        
        idx_to_word = data['idx_to_word']

        # 1. 캡션 전체를 먼저 생성
        captions_ids = self.sample(image_feature)
        
        # <NULL> 토큰 제거 및 <START> 토큰 추가
        first_caption_ids = captions_ids[0]
        first_null = np.where(first_caption_ids == self._null)[0]
        if len(first_null) > 0:
            final_len = first_null[0]
            final_ids = first_caption_ids[:final_len]
        else:
            final_ids = first_caption_ids
            
        final_ids_with_start = np.insert(final_ids, 0, self._start)

        # 2. 완성된 캡션을 모델에 한번 더 통과시켜 전체 어텐션 가중치를 얻음
        with torch.no_grad():
            features = torch.Tensor(image_feature).to(self.device)
            captions_tensor = torch.LongTensor(final_ids_with_start).to(self.device).unsqueeze(0)
            
            _, attention_weights_list = self.forward(features, captions_tensor)
            
            # 3. 마지막 레이어의 Self-Attention 가중치 선택 및 처리
            last_layer_attention = attention_weights_list[-1].cpu()
            heatmap_data = last_layer_attention.mean(dim=1).squeeze(0)

        # 4. Matplotlib으로 히트맵 시각화
        img = image_from_url(image_url)
        if img is None:
            print(f"Could not load image from URL: {image_url}")
            return

        decoded_words = decode_captions(final_ids_with_start.reshape(1,-1), idx_to_word)[0].split()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={'width_ratios': [1, 1.2]})

        ax1.imshow(img)
        ax1.axis('off')
        ax1.set_title(f"Generated Caption:\n{' '.join(decoded_words[1:])}") # <START> 제외하고 표시

        im = ax2.imshow(heatmap_data, cmap='viridis')
        
        ax2.set_xticks(np.arange(len(decoded_words)))
        ax2.set_yticks(np.arange(len(decoded_words)))
        ax2.set_xticklabels(decoded_words, rotation=90)
        ax2.set_yticklabels(decoded_words)
        ax2.set_xlabel("Key (Attended-to Words)")
        ax2.set_ylabel("Query (Generated Word)")
        
        fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        ax2.set_title("Self-Attention Heatmap (Last Layer)")
        
        plt.tight_layout()
        plt.show()


class TransformerDecoderLayer(nn.Module):
    """
    A single layer of a Transformer decoder, to be used with TransformerDecoder.
    """
    def __init__(self, input_dim, num_heads, dim_feedforward=2048, dropout=0.1):
        """
        Construct a TransformerDecoderLayer instance.

        Inputs:
         - input_dim: Number of expected features in the input.
         - num_heads: Number of attention heads
         - dim_feedforward: Dimension of the feedforward network model.
         - dropout: The dropout value.
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(input_dim, num_heads, dropout)
        self.multihead_attn = MultiHeadAttention(input_dim, num_heads, dropout)
        self.linear1 = nn.Linear(input_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, input_dim)

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.norm3 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()


    def forward(self, tgt, memory, tgt_mask=None):
        """
        Pass the inputs (and mask) through the decoder layer.

        Inputs:
        - tgt: the sequence to the decoder layer, of shape (N, T, W)
        - memory: the sequence from the last layer of the encoder, of shape (N, S, D)
        - tgt_mask: the parts of the target sequence to mask, of shape (T, T)

        Returns:
        - out: the Transformer features, of shape (N, T, W)
        """
        # Perform self-attention on the target sequence (along with dropout and
        # layer norm).
        tgt2, self_attn_weights = self.self_attn(query=tgt, key=tgt, value=tgt, attn_mask=tgt_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Attend to both the target sequence and the sequence from the last
        # encoder layer.
        tgt2, _ = self.multihead_attn(query=tgt, key=memory, value=memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Pass
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, self_attn_weights # return self attention weight for visualization

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, memory, tgt_mask=None):
        output = tgt
        attention_weights_list = []

        for mod in self.layers:
            output, attn_weights = mod(output, memory, tgt_mask=tgt_mask)
            attention_weights_list.append(attn_weights)

        return output, attention_weights_list
