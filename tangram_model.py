import torch
import torch.nn as nn
import torchvision.models as models

class TangramAssemblyTransformer(nn.Module):
    def __init__(self, 
                 d_model=512, 
                 nhead=8, 
                 num_layers=6, 
                 vocab_size=1000, # Assuming simple text vocab
                 max_seq_len=7,
                 image_size=224):
        super().__init__()
        
        self.d_model = d_model
        
        # 1. Image Encoder (extracts features from current assembly state)
        # Using a lightweight ResNet or similar
        resnet = models.resnet18(pretrained=True)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1]) # Output: [B, 512, 1, 1]
        self.img_proj = nn.Linear(512, d_model)
        
        # 2. Text Encoder (encodes the instruction/label)
        self.text_embedding = nn.Embedding(vocab_size, d_model)
        # Simple average pooling for text or use a CLS token if using a full transformer for text
        # Here we assume a single vector summary of the text is needed for s_0
        
        # 3. Positional Embedding for the 7 steps
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len + 1, d_model))
        
        # 4. Decoder-only Transformer (Causal)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # 5. Output Head
        # Option A: Generate the next image directly (Pixel-wise)
        # Option B: Predict parameters of the next piece (x, y, rotation, type_id)
        # Here we implement a feature decoder that could be upsampled to an image
        self.image_decoder = nn.Sequential(
            nn.Linear(d_model, 512 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (512, 7, 7)),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), # 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 28x28
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 56x56
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 112x112
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),    # 224x224
            nn.Sigmoid()
        )
        
        # Special encoder for s_0 (Text + Empty Image/Initial State)
        self.text_img_fusion = nn.Linear(d_model * 2, d_model)

    def forward(self, images, text_indices, causal_mask=None):
        """
        Args:
            images: [B, Steps, C, H, W] - Sequence of partial assembly images.
                    Step 0 is usually empty canvas.
            text_indices: [B, TextLen] - Text description tokens.
        Returns:
            predicted_images: [B, Steps, C, H, W] - Predicted next steps.
        """
        B, Steps, C, H, W = images.shape
        
        # 1. Encode Images
        # Flatten steps into batch dimension for efficient encoding
        imgs_flat = images.view(B * Steps, C, H, W)
        img_feats = self.image_encoder(imgs_flat).view(B * Steps, -1) # [B*Steps, 512]
        img_embs = self.img_proj(img_feats).view(B, Steps, -1) # [B, Steps, d_model]
        
        # 2. Encode Text
        text_emb = self.text_embedding(text_indices).mean(dim=1) # [B, d_model]
        
        # 3. Construct Input Sequence s_0, s_1, ...
        # s_0 combines Text and Image_0 (Empty)
        # According to diagram: s_0 = Encoder(Text, Current_Img)
        # We fuse text_emb and img_embs[:, 0]
        s_0 = self.text_img_fusion(torch.cat([text_emb, img_embs[:, 0]], dim=-1)) # [B, d_model]
        
        # The sequence input to transformer is [s_0, s_1, s_2, ..., s_{T-1}]
        # Where s_i (i>0) are just the image embeddings of the steps
        # We replace the first image embedding with the fused s_0
        
        seq_input = img_embs.clone()
        seq_input[:, 0] = s_0
        
        # Add positional embeddings
        seq_input = seq_input + self.pos_embedding[:, :Steps, :]
        
        # 4. Causal Masking
        # Ensure step t can only see 0...t
        if causal_mask is None:
            causal_mask = nn.Transformer.generate_square_subsequent_mask(Steps).to(images.device)
            
        # 5. Transformer Pass
        # Output h_t predicts state t+1
        # TransformerDecoder usually takes (tgt, memory). Here we are using it as a Decoder-only (GPT style).
        # In PyTorch, TransformerDecoderLayer self-attention is masked if we provide tgt_mask.
        # We don't have a separate encoder memory, so we might just use the self-attention part.
        # Or use GPT2Model. But nn.TransformerDecoder with memory=None or dummy might work if configured right,
        # but standard usage expects memory.
        # Better to use a stack of TransformerEncoderLayers with is_causal=True (PyTorch > 1.9) 
        # or manually mask.
        # Let's use TransformerEncoder for GPT-style decoder-only.
        
        # Re-defining transformer as Encoder for GPT-style usage
        # (PyTorch naming is confusing: Encoder = Self-Attention blocks, Decoder = Cross-Attention blocks)
        # For GPT (Decoder-only), we actually use TransformerEncoderLayer with a causal mask.
        pass 
        # (I will fix this in the class definition below by swapping to TransformerEncoder)

        return seq_input

class TangramGPT(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=6, vocab_size=1000, max_seq_len=7):
        super().__init__()
        self.d_model = d_model
        
        # Encoders
        resnet = models.resnet18(pretrained=True)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.img_proj = nn.Linear(512, d_model)
        self.text_embedding = nn.Embedding(vocab_size, d_model)
        self.text_img_fusion = nn.Linear(d_model * 2, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        
        # GPT Backbone (Decoder-only)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Decoder Head (Image Generation)
        self.image_decoder = nn.Sequential(
            nn.Linear(d_model, 512 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (512, 7, 7)),
            nn.ConvTranspose2d(512, 256, 4, 2, 1), # 14
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # 28
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 56
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # 112
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),    # 224
            nn.Sigmoid()
        )

    def forward(self, images, text_indices):
        # images: [B, 7, 3, 224, 224]
        # text_indices: [B, L]
        
        B, Steps, C, H, W = images.shape
        
        # Encode Images
        imgs_flat = images.view(B * Steps, C, H, W)
        img_feats = self.image_encoder(imgs_flat).view(B * Steps, -1)
        img_embs = self.img_proj(img_feats).view(B, Steps, -1)
        
        # Encode Text
        text_emb = self.text_embedding(text_indices).mean(dim=1)
        
        # Prepare Input Sequence
        # s_0 = Fusion(Text, Image_0)
        s_0 = self.text_img_fusion(torch.cat([text_emb, img_embs[:, 0]], dim=-1))
        
        # Sequence: [s_0, s_1, ..., s_6]
        # Note: If we want to predict Image_1 from s_0, Image_2 from s_1...
        # The input to the transformer should be the sequence of "current states".
        # If we have 7 steps, inputs are indices 0 to 6.
        
        seq_input = img_embs.clone()
        seq_input[:, 0] = s_0
        seq_input = seq_input + self.pos_embedding[:, :Steps, :]
        
        # Causal Mask
        mask = nn.Transformer.generate_square_subsequent_mask(Steps).to(images.device)
        
        # Transformer
        # Output [B, Steps, d_model]
        hidden_states = self.transformer(seq_input, mask=mask, is_causal=True)
        
        # Decode to Images
        # h_0 predicts Image_1
        # h_1 predicts Image_2
        # ...
        # h_6 predicts Image_7 (if exists) or we stop at 6 to predict 7th image?
        # Tangram has 7 pieces.
        # Start: Empty (0 pieces).
        # Step 1: 1 piece.
        # ...
        # Step 7: 7 pieces.
        # Input sequence length 7: [Empty, 1pc, ..., 6pcs]
        # Output sequence length 7: [1pc, 2pcs, ..., 7pcs]
        
        # Flatten for decoder
        hidden_flat = hidden_states.view(B * Steps, -1)
        pred_images = self.image_decoder(hidden_flat)
        pred_images = pred_images.view(B, Steps, C, H, W)
        
        return pred_images
