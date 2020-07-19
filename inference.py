import PIL.Image as Image
import torch
import torch.nn as nn
from torchvision import models, transforms

MODEL_PATH = './checkpoints/best_r18.pth'
TEST_BG_IMAGE_PATH = './data/test/0.jpg'
TEST_FG_IMAGE_PATH = './data/test/1.jpg'

# define model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = models.resnet18(pretrained=True)
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 2)
net = net.to(device)
net.load_state_dict(torch.load(MODEL_PATH))

# prepare data
test_transform = transforms.Compose([
                    transforms.Resize(96),
                    transforms.CenterCrop(80),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

# background image and foreground
bg_image = Image.open(TEST_BG_IMAGE_PATH)
fg_image = Image.open(TEST_FG_IMAGE_PATH)

bg_image_tensor = test_transform(bg_image).float()
bg_image_tensor = bg_image_tensor.unsqueeze_(0)
fg_image_tensor = test_transform(fg_image).float()
fg_image_tensor = fg_image_tensor.unsqueeze_(0)

bg_image_tensor = bg_image_tensor.cuda()
fg_image_tensor = fg_image_tensor.cuda()

with torch.no_grad():
    net.eval()
    bg_out = net(bg_image_tensor)
    _, bg_preds = torch.max(bg_out, 1)
    print('background prediction: '+str(bg_preds[0].item()))

    fg_out = net(fg_image_tensor)
    _, fg_preds = torch.max(fg_out, 1)
    print('foreground prediction: '+str(fg_preds[0].item()))

