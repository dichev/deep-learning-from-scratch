import torch
import pandas as pd
from models import optimizers
from matplotlib import pyplot as plt
from models.autoencoders import MatrixFactorization
from functions.preprocessing import integer_encoder


# Hyperparams & settings
LEARN_RATE = 0.5
EPOCHS = 50
BATCH_SIZE = 10000
MATRIX_RANK = 10
DEVICE = 'cuda'


# Data preprocessing
anime_meta = pd.read_csv('./data/anime-metadata.csv')
print(anime_meta.head())
df = pd.read_csv('./data/anime-users-ratings-filtered.csv')
print(df)
df['user_idx'], user_dict = integer_encoder(df['user_id'])
df['anime_idx'], anime_dict = integer_encoder(df['anime_id'])
N = df.shape[0]


# Model
model = MatrixFactorization(n_users=len(user_dict), n_animes=len(user_dict), rank=MATRIX_RANK, device=DEVICE)


# Fit the data
history = []
optimizer = optimizers.Optimizer(model.params, lr=LEARN_RATE)
for i in range(1, EPOCHS * N//BATCH_SIZE):
    # batch
    batch = df.iloc[torch.randint(0, N, (BATCH_SIZE,))]
    ratings, users, animes = batch[['rating', 'user_idx', 'anime_idx']].values.T
    ratings = torch.tensor(ratings, device=DEVICE)

    # optimize
    predictions = model.forward(users, animes)
    cost = ((ratings - predictions)**2).mean()
    cost.backward()
    optimizer.step().zero_grad()

    # collect metrics
    history.append((cost.item(),))
    if i <= 10 or i % 100 == 0 or i == EPOCHS:
        print(f'#{i}/{EPOCHS * N//BATCH_SIZE} cost={cost.item()} ')


# Plot the results
loss, = zip(*history)
plt.plot(range(len(loss)), loss); plt.title('Loss'); plt.xlabel('iterations'); plt.show()


predicted = (model.U[df['user_idx'].values] * model.V[df['anime_idx'].values]).sum(axis=1).detach().cpu().numpy()
df['predicted'] = predicted
print(df)
