import pickle

from pathlib import Path

from argparse import ArgumentParser

from utils import load_data, create_input_tensors, alphabet



def main():
    parser = ArgumentParser()
    parser.add_argument('-w', '--witness_rate', type=float, required=True)
    parser.add_argument('-l', '--max_len', type=int, default=20)
    parser.add_argument('--min_len', type=int, default=0)
    # parser.add_argument('--n_samples', type=int, required=True)
    parser.add_argument('--n_seq', type=int, default=None)
    parser.add_argument('--input_data_dir', required=True)
    parser.add_argument('--translate', action='store_true')
    parser.add_argument('--use_vj', action='store_true')
    args = parser.parse_args()

    loaded_data = load_data(
        witness_rate=args.witness_rate,
        input_data_dir=args.input_data_dir,
        max_len=args.max_len,
        min_len=args.min_len,
        alphabet=alphabet,
        # n_samples=args.n_samples,
        n_seq=args.n_seq,
        translate=args.translate,
        use_vj=args.use_vj,
    )
    if args.use_vj:
        samples, repertoires_aa, sample_labels, sample_vj_genes, v_size, j_size = loaded_data
        input_data = create_input_tensors(samples, sample_labels, sample_vj_genes, v_size, j_size)
    else:
        samples, repertoires_aa, sample_labels = loaded_data
        input_data = create_input_tensors(samples, sample_labels)

    repertoires_dir = Path(f'{args.input_data_dir}/repertoires_aa/{args.witness_rate}.pickle')
    repertoires_dir.parent.mkdir(parents=True, exist_ok=True)
    with open(repertoires_dir, 'wb') as otp:
        pickle.dump(repertoires_aa, otp)
    input_tensors_dir = Path(f'{args.input_data_dir}/input_tensors/{args.witness_rate}.pickle')
    input_tensors_dir.parent.mkdir(parents=True, exist_ok=True)
    with open(input_tensors_dir, 'wb') as otp:
        pickle.dump(input_data, otp)


if __name__ == '__main__':
    main()
