from models.recurrent_networks import Seq2Seq, Encoder, Decoder
from lib.optimizers import Adam

from examples.recurrent.reused.translate_en_to_fr import fit, diagnostics, vocab_en, vocab_fr, PAD_IDX
from utils.rng import seed_global
seed_global(1)

# training settings
EPOCHS = 100
BATCH_SIZE = 1024
LEARN_RATE = 0.005
DEVICE = 'cuda'

# hyperparams
ENCODER_EMBED_SIZE = 128
ENCODER_HIDDEN_SIZE = 256
DECODER_EMBED_SIZE = 128
DECODER_HIDDEN_SIZE = 256
N_LAYERS = 1


# Model
model = Seq2Seq(
    encoder=Encoder(vocab_en.size, ENCODER_EMBED_SIZE, ENCODER_HIDDEN_SIZE, cell='lstm', n_layers=N_LAYERS, direction='backward', padding_idx=PAD_IDX),
    decoder=Decoder(vocab_fr.size, DECODER_EMBED_SIZE, DECODER_HIDDEN_SIZE, cell='lstm', n_layers=N_LAYERS, direction='forward', padding_idx=PAD_IDX),
    sos_token=vocab_fr.to_idx['<SOS>'], eos_token=vocab_fr.to_idx['<EOS>']
)
model.to(DEVICE)
model.summary()
optimizer = Adam(model.parameters(), lr=LEARN_RATE)
diagnostics()

# Training
fit(model, optimizer, epochs=EPOCHS, batch_size=BATCH_SIZE, device=DEVICE)



