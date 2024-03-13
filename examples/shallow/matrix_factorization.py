import torch
import pandas as pd
from tqdm import trange
from lib import optimizers
from matplotlib import pyplot as plt
from lib.autoencoders import MatrixFactorization
from preprocessing.integer import index_encoder
from scipy import sparse
# todo one-hot + linear = indices +embedding
# Hyperparams & settings
LEARN_RATE = 0.5
EPOCHS = 50
BATCH_SIZE = 10000
MATRIX_RANK = 10
DEVICE = 'cuda'


# Data preprocessing
print('Data preprocessing..')
df_meta = pd.read_csv('./data/anime-metadata.csv')
df = pd.read_csv('./data/anime-users-ratings-filtered.csv')
df['user_idx'], user_to_idx, idx_to_user = index_encoder(df['user_id'])
df['anime_idx'], anime_to_idx, idx_to_anime = index_encoder(df['anime_id'])
N = df.shape[0]
# print(df_meta.head())
# print(df)


# Model
model = MatrixFactorization(n_users=len(user_to_idx), n_animes=len(anime_to_idx), rank=MATRIX_RANK).to(DEVICE)
optimizer = optimizers.SGD(model.parameters(), lr=LEARN_RATE)

# Fit the data
history = []
print('Fitting data to MatrixFactorization..')
print(f' - training data: {N} samples ({N//BATCH_SIZE} batches with batch size {BATCH_SIZE})')

pbar = trange(1, EPOCHS+1, desc='EPOCH')
for epoch in pbar:
    indices = torch.randperm(N)
    for i in range(0, N, BATCH_SIZE):

        # batch
        batch = df.iloc[indices[i:i+BATCH_SIZE]]
        ratings, users, animes = batch[['rating', 'user_idx', 'anime_idx']].values.T
        ratings = torch.tensor(ratings, device=DEVICE)

        # optimize
        optimizer.zero_grad()
        predictions = model.forward(users, animes)
        cost = ((ratings - predictions)**2).mean()
        cost.backward()
        optimizer.step()

        # collect metrics
        history.append((cost.item(),))
        pbar.set_postfix(cost=cost.item())


# Plot the loss function
loss, = zip(*history)
plt.plot(range(len(loss)), loss); plt.title('Loss'); plt.xlabel('iterations'); plt.show()


# Visualize a slice of the real and approximated rating matrices
users = slice(0, 100)
animes = slice(0, 50)

anime_ids = [idx_to_anime[idx] for idx in range(animes.stop)]
anime_names = df_meta.loc[df_meta['anime_id'].isin(anime_ids), 'name']
df['predicted'] = model.forward(df['user_idx'].values, df['anime_idx'].values).detach().cpu().numpy(); optimizer.step().zero_grad()
R = sparse.csc_matrix((df['rating'].values, (df['user_idx'].values, df['anime_idx'].values)))  # using sparse matrix to fit in memory
R_hat = sparse.csc_matrix((df['predicted'].values, (df['user_idx'].values, df['anime_idx'].values)))
R_hat_full = model.predict(torch.arange(users.stop), torch.arange(animes.stop))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
ax1.matshow(R[users, animes].todense().T, aspect='auto', cmap='Greens'); ax1.set_title('Ratings (real data)'); ax1.set_xlabel('Users'); ax1.set_yticks(range(animes.stop), anime_names.str.slice(0, 20))
ax2.matshow(R_hat[users, animes].todense().T, aspect='auto', cmap='Greens'); ax2.set_title('Ratings (predicted)'); ax2.set_xlabel('Users'); ax2.set_yticks([])
ax3.matshow(R_hat_full.T, aspect='auto', cmap='Greens');  ax3.set_title('Ratings (predicted & extrapolated)'); ax3.set_xlabel('Users'); ax3.set_yticks([])
plt.suptitle(f'User ratings per anime (slice users[{users.start}:{users.stop}] x anime[{animes.start}:{animes.stop}])')
plt.tight_layout()
plt.show()
