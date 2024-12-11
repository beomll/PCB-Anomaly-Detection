import argparse
from scripts.main import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('-m', '--model_type', type=str, default='resnet50', help='model type (resnet50 or resnext)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-1, help='learning rate')
    parser.add_argument('-e', '--num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('-sch', '--scheduler', type=str, default='reduce', help='scheduler')
    parser.add_argument('-opt', '--optimizer', type=str, default='sgd', help='optimizer')
    parser.add_argument('-w', '--warmup_steps', type=int, default=100, help='warmup steps')
    parser.add_argument('-g', '--gamma', type=float, default=0.1, help='gamma')
    parser.add_argument('-d', '--drop_prob', type=float, default=0.1, help='dropout probability')
    parser.add_argument('-s', '--step_size', type=int, default=50, help='step size')
    parser.add_argument('-trn', '--train_data', type=str, default='data/train', help='train data path')
    parser.add_argument('-val', '--val_data', type=str, default='data/val', help='validation data path')
    parser.add_argument('-tst', '--test_data', type=str, default='data/test', help='test data path')
    parser.add_argument('-p', '--pretrained', type=bool, default=False, help='pretrained model')
    parser.add_argument('-mp', '--model_path', type=str, default=None, help='model path')
    parser.add_argument('-sp', '--save_path', type=str, default='checkpoints', help='save path')

    args = parser.parse_args()
    main(args)
