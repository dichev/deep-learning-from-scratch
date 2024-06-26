from models.transformer_networks import TransformerEncoder, TransformerDecoder, Transformer
from lib.optimizers import Adam

from examples.recurrent.reused.translate_en_to_fr import fit, diagnostics, vocab_en, vocab_fr, PAD_IDX
from utils.rng import seed_global
seed_global(1)

# training settings
EPOCHS = 400
BATCH_SIZE = 1024
LEARN_RATE = 0.001
DEVICE = 'cuda'


# Model
model = Transformer(
    encoder=TransformerEncoder(vocab_en.size, 256, 512, n_layers=4, padding_idx=PAD_IDX, attn_heads=4, dropout=.1, norm_first=True, gelu_activation=True, scale_up_embeddings=False),
    decoder=TransformerDecoder(vocab_fr.size, 256, 512, n_layers=4, padding_idx=PAD_IDX, attn_heads=4, dropout=.1, norm_first=True, gelu_activation=True, scale_up_embeddings=False, tied_embeddings=False),
    sos_token=vocab_fr.to_idx['<SOS>'], eos_token=vocab_fr.to_idx['<EOS>']
)
model.to(DEVICE)
model.summary()
optimizer = Adam(model.parameters(), lr=LEARN_RATE)
diagnostics()

# Training
fit(model, optimizer, epochs=EPOCHS, batch_size=BATCH_SIZE, device=DEVICE, title='Transformer (Norm First) Label Smooth 0.01', visualize_fn=model.visualize_attn_weights)
