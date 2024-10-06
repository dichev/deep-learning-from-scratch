import torch
import matplotlib.pyplot as plt
import torchvision
from PIL import Image
from torch.nn.functional import relu
from preprocessing.floats import normalizeMinMax

def inject_backward_relu(model, mode):
    def relu_gradient_hook(m, grad_input, grad_output):
        if mode == 'deconvnet':
            grad_new = relu(grad_output[0])
        elif mode == 'guided_backprop':
            grad_new = relu(grad_output[0]) * (grad_input[0] != 0)
        else:
            raise NotImplementedError
        m.__handle.remove()  # clean up the hook
        return (grad_new,)

    for module in model.modules():
        if isinstance(module, torch.nn.ReLU):
            if module.inplace:  # inplace operations cannot be overridden by hooks
                module.inplace = False
            module.__handle = module.register_full_backward_hook(relu_gradient_hook)

def generate_saliency_map(model, image, mode='guided_backprop', normalize=True):
    print(f'Generate saliency map: mode={mode}')
    batch = image.clone()
    batch.requires_grad = True
    if mode != 'backprop':
        inject_backward_relu(model, mode)
    model.zero_grad()
    prediction = model(batch)
    score, _ = torch.max(prediction, 1)
    score.backward()
    saliency_map = batch.grad.abs().squeeze(0).permute(1, 2, 0)
    if normalize:
        saliency_map = normalizeMinMax(saliency_map)
    return saliency_map


# Get the image
image = Image.open('./data/images/cat-tiger.jpg')  # 360x480x3


# Model
print('Loading pretrained ResNet50 model..')
weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
model.eval()  # very important (e.g. BatchNorm is used)
preprocess = weights.transforms()
image_normalized = preprocess(image).unsqueeze(0)


# Get probability and class name
prediction = model(image_normalized)
probs = prediction.squeeze(0).softmax(0)
label = probs.argmax().item()
score = probs.squeeze(0)[label].item()
category = weights.meta['categories'][label]
print(f"Forward a single image input:")
print(f"{category}: {100 * score:.5}%")


# Extract saliency maps
saliency_map_1 = generate_saliency_map(model, image_normalized, mode='backprop')
saliency_map_2 = generate_saliency_map(model, image_normalized, mode='deconvnet')
saliency_map_3 = generate_saliency_map(model, image_normalized, mode='guided_backprop')


# Plot the input image and activations
fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(8, 4))
fig.suptitle(f'Saliency maps for\n"{category}" score ({100 * score:.3}%)')
ax0.imshow(image); ax0.axis(False); ax0.set_title('Input image', fontsize=9)
ax1.imshow(saliency_map_1, cmap='gray'); ax1.axis(False); ax1.set_title(f'\nbackpropagation', fontsize=9)
ax2.imshow(saliency_map_2, cmap='gray'); ax2.axis(False); ax2.set_title(f'\ndeconvolution', fontsize=9)
ax3.imshow(saliency_map_3, cmap='gray'); ax3.axis(False); ax3.set_title(f'\nguided backpropagation', fontsize=9)
plt.tight_layout()
plt.show()


