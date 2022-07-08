import argparse


def init_opts():
    parser = argparse.ArgumentParser(description="Make simulated time series dataset")

    # Distribution type
    parser.add_argument('--distribution_type', type=str,
                        help='Type of distribution: gaussian or uniform',
                        default="gaussian")
    parser.add_argument('--ds_size', type=int,
                        help='Change dataset size. Default is set to 60000',
                        default=60000)

    # Data split
    parser.add_argument('--train_split_perc', type=int,
                        help='Train-Split value in %. Default is set to 80 (%)',
                        default=80)
    parser.add_argument('--test_split_perc', type=int,
                        help='Test-Split value in %. Default is set to 20 (%)',
                        default=20)
    parser.add_argument('--val_split_perc', type=int,
                        help='Validation-Split value in %. Default is set to 20 (%)',
                        default=20)
    parser.add_argument('--val_split_after_test', type=bool,
                        help='Splits validation dataset from training dataset after test split if value is true. Default is set to true.',
                        default=True)

    # Time Series opts                    
    parser.add_argument('--ts_len', type=int,
                        help='Length of time series. Default is set to 28.',
                        default=28)
    parser.add_argument('--min_start', type=int,
                        help='Minimum (inclusive) step start positon. Default is set to 3.',
                        default=3)
    parser.add_argument('--max_start', type=int,
                        help='Maximum (exclusive) step start positon. Default is set to 15.',
                        default=15)
    parser.add_argument('--min_width', type=int,
                        help='Minimum (exclusive) step width. Default is set to 3.',
                        default=3)
    parser.add_argument('--max_width', type=int,
                        help='Maximum (exclusive) step width. Default is set to 10.',
                        default=10)
    parser.add_argument('--min_amp', type=float,
                        help='Minimum (exclusive) step amp. Default is set to 0.3.',
                        default=0.3)
    parser.add_argument('--max_amp', type=float,
                        help='Maximum (exclusive) step amp. Default is set to 1.0.',
                        default=1.0)
    parser.add_argument('--batch_size', type=int,
                        help='Batch size to compute gafs. Default is set to 128',
                        default=128)

    # Return path
    parser.add_argument('--p_out', type=str,
                        help='path where the data should be saved.',
                        default="data")


    return parser.parse_args()