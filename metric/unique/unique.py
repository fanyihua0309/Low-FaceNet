import torch
from torchvision import transforms
from metric.unique.BaseCNN import BaseCNN
from metric.unique.Transformers import AdaptiveResize
from PIL import Image
import argparse


def parse_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", type=bool, default=False)
    parser.add_argument('--get_scores', type=bool, default=True)
    parser.add_argument("--use_cuda", type=bool, default=True)
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=19901116)

    parser.add_argument("--backbone", type=str, default='resnet34')
    parser.add_argument("--fc", type=bool, default=True)
    parser.add_argument('--scnn_root', type=str, default='saved_weights/scnn.pkl')

    parser.add_argument("--network", type=str, default="basecnn") #basecnn or dbcnn
    parser.add_argument("--representation", type=str, default="BCNN")

    parser.add_argument("--ranking", type=bool, default=True)  # True for learning-to-rank False for regular regression
    parser.add_argument("--fidelity", type=bool, default=True)  # True for fidelity loss False for regular ranknet with CE loss
    parser.add_argument("--std_modeling", type=bool,
                        default=True)  # True for modeling std False for not
    parser.add_argument("--std_loss", type=bool, default=True)
    parser.add_argument("--margin", type=float, default=0.025)

    parser.add_argument("--split", type=int, default=1)
    parser.add_argument("--trainset", type=str, default="./IQA_database/")
    parser.add_argument("--live_set", type=str, default="./IQA_database/databaserelease2/")
    parser.add_argument("--csiq_set", type=str, default="./IQA_database/CSIQ/")
    parser.add_argument("--tid2013_set", type=str, default="./IQA_database/TID2013/")
    parser.add_argument("--bid_set", type=str, default="./IQA_database/BID/")
    #parser.add_argument("--cid_set", type=str, default="./IQA_database/CID2013_camera/")
    parser.add_argument("--clive_set", type=str, default="./IQA_database/ChallengeDB_release/")
    parser.add_argument("--koniq10k_set", type=str, default="./IQA_database/koniq-10k/")
    parser.add_argument("--kadid10k_set", type=str, default="./IQA_database/kadid10k/")

    parser.add_argument("--eval_live", type=bool, default=True)
    parser.add_argument("--eval_csiq", type=bool, default=True)
    parser.add_argument("--eval_tid2013", type=bool, default=True)
    parser.add_argument("--eval_kadid10k", type=bool, default=True)
    parser.add_argument("--eval_bid", type=bool, default=True)
    parser.add_argument("--eval_clive", type=bool, default=True)
    parser.add_argument("--eval_koniq10k", type=bool, default=True)

    parser.add_argument("--split_modeling", type=bool, default=False)

    parser.add_argument('--ckpt_path', default='./checkpoint', type=str,
                        metavar='PATH', help='path to checkpoints')
    parser.add_argument('--ckpt', default=None, type=str, help='name of the checkpoint to load')

    parser.add_argument("--train_txt", type=str, default='train.txt') # train.txt | train_synthetic.txt | train_authentic.txt | train_sub2.txt | train_score.txt

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--batch_size2", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=384, help='None means random resolution')
    parser.add_argument("--max_epochs", type=int, default=3)
    parser.add_argument("--max_epochs2", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--decay_interval", type=int, default=3)
    parser.add_argument("--decay_ratio", type=float, default=0.1)
    parser.add_argument("--epochs_per_eval", type=int, default=1)
    parser.add_argument("--epochs_per_save", type=int, default=1)

    parser.add_argument('--dataset_name', default='CelebA-Test', type=str)
    parser.add_argument('--dataset_root', default='dataset', type=str)
    parser.add_argument('--method_name', default='no_contrast', type=str)

    return parser.parse_args()


# source: https://github.com/zwx8981/UNIQUE
def unique_score(img_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_transform = transforms.Compose([
        AdaptiveResize(768),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])

    config = parse_config()
    config.backbone = 'resnet34'
    config.representation = 'BCNN'

    model = BaseCNN(config)
    model = torch.nn.DataParallel(model).cuda()

    # module_path = os.path.dirname(__file__)
    # ckpt = os.path.join(module_path, 'model.pt')

    ckpt = 'checkpoint/unique.pt'

    checkpoint = torch.load(ckpt)
    model.load_state_dict(checkpoint)

    model.eval()

    image = Image.open(img_path)
    image = test_transform(image)
    image = torch.unsqueeze(image, dim=0)
    image = image.to(device)
    with torch.no_grad():
        score1, std1 = model(image)

    score = score1.cpu().item()
    return score
    # std = std1.cpu().item()
    # print('The predicted quality of image1 is {}, with an estimated std of {}'.format(score, std))

