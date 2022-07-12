from helper_functions import load_checkpoint

LOAD_MODEL_FILE = "overfit.pth.tar"

if __name__ == '__main__':
    model = load_checkpoint(LOAD_MODEL_FILE)
    model