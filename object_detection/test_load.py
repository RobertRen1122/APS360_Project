import torch
from model import YOLOv1
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from utils import convert_cellboxes, plot_pred, cellboxes_to_boxes

MODEL_PATH = 'model.pth.tar'
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_DATA_DIR = 'data/images/4920.jpg'

if __name__ == '__main__':
    model = YOLOv1(split_size=7, num_boxes=2, num_classes=1).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    optimizer.load_state_dict(checkpoint['optimizer'])
    to_tensor = transforms.ToTensor()
    reshape = transforms.Resize((448, 448))
    test_img = Image.open(TEST_DATA_DIR)
    raw_img = Image.open(TEST_DATA_DIR)
    test_img = to_tensor(test_img)
    test_img = reshape(test_img)
    test_img = torch.unsqueeze(test_img, 0)
    print(test_img.shape)
    output = model(test_img)
    cellboxes = convert_cellboxes(output)
    plot_pred(TEST_DATA_DIR, cellboxes)

