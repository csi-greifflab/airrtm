from pathlib import Path

from argparse import ArgumentParser

from utils import load_data, create_input_tensors, alphabet


def main():
    parser = ArgumentParser()
    parser.add_argument('-w', '--witness_rate', type=float, required=True)
    parser.add_argument('-l', '--max_len', type=int, default=20)
    parser.add_argument('--n_samples', type=int, required=True)
    parser.add_argument('--n_seqs', type=int, default=None)
    parser.add_argument('--input_data_dir', required=True)
    args = parser.parse_args()

    truncated_samples, repertoires_aa = load_data(
        witness_rate=witness_rate,
        input_data_dir=input_data_dir,
        max_len=args.max_len,
        alphabet=alphabet,
        n_samples=args.n_samples,
        n_seq=args.n_seq,
    )
    input_data = create_input_tensors(truncated_samples)

    repertoires_dir = Path(f'{input_data_dir}/repertoires_aa/{witness_rate}.pickle')
    repertoires_dir.mkdir(parents=True, exist_ok=True)
    with open(repertoires_dir, 'wb') as otp:
        pickle.dump(repertoires_aa, otp)
    input_tensors_dir = Path(f'{input_data_dir}/input_tensors/{witness_rate}.pickle')
    input_tensors_dir.mkdir(parents=True, exist_ok=True)
    with open(input_tensors_dir, 'wb') as otp:
        pickle.dump(input_data, otp)


if __name__ == '__main__':
    main()
