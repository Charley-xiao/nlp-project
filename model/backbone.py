import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# ------------------------------
# Transformer Fusion Block
# ------------------------------
class TransformerFusionBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        """
        Residual block with self-attention and a feed-forward network.
        
        Args:
            d_model (int): Dimensionality of the input features.
            nhead (int): Number of attention heads.
            dim_feedforward (int): Dimensionality of the feed-forward network.
            dropout (float): Dropout probability.
        """
        super(TransformerFusionBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()  # Alternatively, GELU

    def forward(self, src):
        """
        Args:
            src (torch.Tensor): Input tensor of shape (batch, seq_len, d_model).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch, seq_len, d_model).
        """
        # Rearrange for multi-head attention (seq_len, batch, d_model)
        src_t = src.transpose(0, 1)
        attn_output, _ = self.self_attn(src_t, src_t, src_t)
        src_t = src_t + self.dropout1(attn_output)
        src_t = self.norm1(src_t)
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(src_t))))
        src_t = src_t + self.dropout2(ff_output)
        src_t = self.norm2(src_t)
        return src_t.transpose(0, 1)  # Back to (batch, seq_len, d_model)

# ------------------------------
# Combined Classifier with Attention & Residual Blocks
# ------------------------------
class CombinedClassifier(nn.Module):
    def __init__(self, hidden_dim, nhead=2, num_layers=1, dim_feedforward=128, output_dim=2, dropout=0.1):
        """
        Combines handcrafted and latent feature branches using transformer fusion blocks.
        
        Args:
            hidden_dim (int): Dimension of each branch's feature output.
            nhead (int): Number of attention heads.
            num_layers (int): Number of transformer fusion blocks.
            dim_feedforward (int): Dimensionality of the feed-forward network.
            output_dim (int): Number of output classes.
            dropout (float): Dropout probability.
        """
        super(CombinedClassifier, self).__init__()
        self.transformer_layers = nn.ModuleList([
            TransformerFusionBlock(hidden_dim, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        # After transformer fusion, we pool over the sequence (of length 2) and classify.
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, handcrafted, latent):
        """
        Args:
            handcrafted (torch.Tensor): Features from handcrafted branch, shape (batch, hidden_dim).
            latent (torch.Tensor): Features from the text encoder branch, shape (batch, hidden_dim).
        
        Returns:
            torch.Tensor: Logits of shape (batch, output_dim).
        """
        # Stack the two branches to form a sequence of two tokens.
        x = torch.stack([handcrafted, latent], dim=1)  # Shape: (batch, 2, hidden_dim)
        for layer in self.transformer_layers:
            x = layer(x)
        # Aggregate via mean pooling over the sequence dimension.
        x = x.mean(dim=1)
        logits = self.classifier(x)
        return logits

# ------------------------------
# Classifier Backbone with Trainable Encoder
# ------------------------------
class ClassifierBackbone(nn.Module):
    def __init__(self, handcrafted_dim, latent_dim, hidden_dim=128, output_dim=2, dropout=0.1,
                 nhead=2, num_layers=1, dim_feedforward=128):
        """
        Classifier backbone.
        
        Args:
            handcrafted_dim (int): Dimensionality of the handcrafted feature vector.
            latent_dim (int): Dimensionality of the latent feature vector from the text encoder.
            hidden_dim (int): Internal hidden dimension for processing each branch.
            output_dim (int): Number of output classes.
            dropout (float): Dropout probability.
            nhead (int): Number of attention heads in the fusion blocks.
            num_layers (int): Number of transformer fusion blocks.
            dim_feedforward (int): Dimensionality of the feed-forward network in transformer blocks.
        """
        super(ClassifierBackbone, self).__init__()

        # Branch for handcrafted features.
        self.handcrafted_fc = nn.Sequential(
            nn.Linear(handcrafted_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # Branch for latent features from the trainable text encoder.
        self.latent_fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # Combined classifier that fuses both branches.
        self.combined_classifier = CombinedClassifier(hidden_dim, nhead, num_layers, dim_feedforward, output_dim, dropout)

    def forward(self, handcrafted_features, latent_features):
        """
        Args:
            handcrafted_features (torch.Tensor): Tensor of handcrafted features, shape (batch, handcrafted_dim).
            latent_features (torch.Tensor): Tensor of latent features from text encoder, shape (batch, latent_dim).
        
        Returns:
            torch.Tensor: Logits of shape (batch, output_dim).
        """
        h_handcrafted = self.handcrafted_fc(handcrafted_features)  # (batch, hidden_dim)
        h_latent = self.latent_fc(latent_features)  # (batch, hidden_dim)
        
        # Fuse the two branches using the combined classifier.
        logits = self.combined_classifier(h_handcrafted, h_latent)
        return logits

# ------------------------------
# Example Usage
# ------------------------------
if __name__ == "__main__":
    handcrafted_dim = 20
    encoder_model_name = "roberta-base"
    hidden_dim = 128
    output_dim = 2
    batch_size = 8

    handcrafted_features = torch.randn(batch_size, handcrafted_dim)
    raw_texts = [
        "This is a sample sentence for testing.",
        "Another example text goes here.",
        "AI-generated content detection is an interesting problem.",
        "Machine learning has many applications.",
        "Natural language processing is fascinating.",
        "This is text number six.",
        "Sample text for demonstration purposes.",
        "Final example sentence in the batch."
    ]

    tokenizer = AutoTokenizer.from_pretrained(encoder_model_name)
    text_encoder = AutoModel.from_pretrained(encoder_model_name)
    latent_features = text_encoder(**tokenizer(raw_texts, return_tensors="pt", padding=True, truncation=True)).last_hidden_state.mean(dim=1)
    latent_dim = text_encoder.config.hidden_size

    model = ClassifierBackbone(
        handcrafted_dim, latent_dim, hidden_dim=hidden_dim,
        output_dim=output_dim, dropout=0.1, nhead=2, num_layers=2, dim_feedforward=256
    )
    print(model)
    logits = model(handcrafted_features, latent_features)
    print("Logits shape:", logits.shape)  # Expected: (batch_size, output_dim)
