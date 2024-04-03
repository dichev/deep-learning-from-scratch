from models.attention_networks import Seq2Seq, AdditiveAttentionDecoder, AttentionEncoder
from lib.optimizers import Adam

from examples.recurrent.reused.translate_en_to_fr import fit, diagnostics, vocab_en, vocab_fr, PAD_IDX
from utils.rng import seed_global
seed_global(1)

# training settings
EPOCHS = 100
BATCH_SIZE = 1024
LEARN_RATE = 0.02
DEVICE = 'cuda'

# hyperparams
ENCODER_EMBED_SIZE = 128
ENCODER_HIDDEN_SIZE = 256
DECODER_EMBED_SIZE = 128
DECODER_HIDDEN_SIZE = 256

# Model
model = Seq2Seq(
    encoder=AttentionEncoder(vocab_en.size, ENCODER_EMBED_SIZE, ENCODER_HIDDEN_SIZE, cell='gru', n_layers=1, padding_idx=PAD_IDX),
    decoder=AdditiveAttentionDecoder(vocab_fr.size, DECODER_EMBED_SIZE, DECODER_HIDDEN_SIZE, enc_hidden_size=ENCODER_HIDDEN_SIZE, attn_hidden_size=ENCODER_HIDDEN_SIZE//2, attn_dropout=0, cell='gru', n_layers=1, padding_idx=PAD_IDX),
    sos_token=vocab_fr.to_idx['<SOS>'], eos_token=vocab_fr.to_idx['<EOS>']
)
model.to(DEVICE)
model.summary()
optimizer = Adam(model.parameters(), lr=LEARN_RATE)
diagnostics()

# Training
fit(model, optimizer, epochs=EPOCHS, batch_size=BATCH_SIZE, device=DEVICE)
