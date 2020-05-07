import torch
import torchvision.transforms as transforms

import constants

import matplotlib.pyplot

def transform_image(screen):
    return transforms.Compose([
        transforms.ToPILImage(mode="RGB"),
        transforms.Resize((110, 84)),
        transforms.CenterCrop(92),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.4161, ], [0.1688, ]),
    ])(screen)


def conv2d_size_out(size, kernel_size=5, stride=2):
    return (size - (kernel_size - 1) - 1) // stride + 1


def process_state(cumulative_screenshot):
    last_four_images = cumulative_screenshot[-constants.N_IMAGES_PER_STATE:]
    return torch.cat(last_four_images, dim=0).unsqueeze(0)


def state_to_image(state, identifier):
    i = 0
    for channel in state[0]:
        matplotlib.pyplot.imsave("state-" + identifier + "-" + str(i) + ".png", channel)
        i += 1