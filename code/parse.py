import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Go RecModel")
    
    parser.add_argument('--bpr_batch', type=int, default=1024,
                        help="the batch size for bpr loss training procedure")  # 512 1024 2048 4096

    parser.add_argument('--epochs', type=int, default=1000) 

    parser.add_argument('--testbatch', type=int, default=512,
                        help="the batch size of users for testing")

    parser.add_argument('--seed', type=int, default=0 ,help='random seed')

    parser.add_argument('--K', type=int, default=3)

    parser.add_argument('--lr', type=float, default=0.001,
                        help="the learning rate:0.001")  # 0.001
    
    parser.add_argument('--dataset', type=str, default='yelp2018')

    parser.add_argument('--patience', type=int, default=50)

    parser.add_argument('--dropout', type=float, default=1e-1)

    parser.add_argument('--num_neg', type=int, default=64)

    parser.add_argument('--init_weight', type=float, default=1.0)

    parser.add_argument('--decay', type=float, default=1e-4)

    parser.add_argument('--tau', type=float, default=0.1,
                        help="the temperature for softmax in loss function")  # 0.1
    parser.add_argument('--lambda_', type=float, default=1,
                        help="the lambda for cross norm in loss function")  # 0.5

    return parser.parse_args()
