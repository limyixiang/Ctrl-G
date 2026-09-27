import argparse

import numpy
import torch

from tqdm import tqdm
from ctrlg import HMM


def init():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--sequences_file', required=True)
    arg_parser.add_argument('--embeddings_file', required=True)
    arg_parser.add_argument('--hidden_states', required=True, type=int)
    arg_parser.add_argument('--vocab_size', required=True, type=int)
    arg_parser.add_argument('--eos_token_id', required=True, type=int)
    arg_parser.add_argument('--kmeans_iterations', default=100, type=int)
    arg_parser.add_argument('--kmeans_sample_size', default=262144, type=int,
        help='maximum number of non-initial token embeddings used to fit centroids')
    arg_parser.add_argument('--assignment_batch_size', default=8192, type=int,
        help='number of embeddings processed in each centroid fitting or assignment batch')
    arg_parser.add_argument('--pseudocount', default=0.001, type=float)
    arg_parser.add_argument('--seed', default=42, type=int)
    arg_parser.add_argument('--output_file', required=True)
    args = arg_parser.parse_args()

    if args.hidden_states < 2:
        arg_parser.error('--hidden_states must be at least 2')
    if args.kmeans_sample_size <= 0 or args.assignment_batch_size <= 0:
        arg_parser.error('sample and assignment batch sizes must be positive')
    if args.kmeans_iterations <= 0:
        arg_parser.error('--kmeans_iterations must be positive')
    return args


def load_examples(sequences_file, embeddings_file, eos_token_id):
    # mmap keeps the saved (usually bf16) tensor file-backed. Converting the
    # entire padded tensor to float32 would defeat the memory limit below.
    seqs = torch.load(sequences_file, map_location='cpu', weights_only=True)
    embeddings = torch.load(embeddings_file, map_location='cpu',
        mmap=True, weights_only=True)
    if seqs.ndim != 2 or embeddings.ndim != 3 or seqs.shape != embeddings.shape[:2]:
        raise ValueError('sequences and embeddings must have matching first two dimensions')
    if seqs.shape[0] == 0 or seqs.shape[1] == 0:
        raise ValueError('LVD sequences are empty')

    eos = seqs.eq(eos_token_id).numpy()
    lengths = numpy.where(eos.any(axis=1), eos.argmax(axis=1),
        seqs.shape[1]).astype(numpy.int64)
    offsets = numpy.zeros(len(lengths) + 1, dtype=numpy.int64)
    numpy.cumsum(lengths, out=offsets[1:])
    print(f'seqs num: {len(seqs)}', flush=True)
    print(f'embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}', flush=True)
    print(f'non-EOS embeddings: {offsets[-1]}', flush=True)
    return seqs, embeddings, lengths, offsets


def gather_vectors(embeddings, rows, positions, batch_size):
    """Gather only requested embeddings, with a bounded float32 conversion."""
    vectors = numpy.empty((len(rows), embeddings.shape[-1]), dtype=numpy.float32)
    for start in tqdm(range(0, len(rows), batch_size), desc='loading K-means sample'):
        stop = min(start + batch_size, len(rows))
        vectors[start:stop] = embeddings[rows[start:stop],
            positions[start:stop]].float().numpy()
    return vectors


def sample_training_vectors(embeddings, lengths, sample_size, batch_size, seed):
    # The original initializer trained on every position except position zero.
    # Index this population compactly without materializing its embeddings.
    eligible = numpy.maximum(lengths - 1, 0)
    offsets = numpy.zeros(len(lengths) + 1, dtype=numpy.int64)
    numpy.cumsum(eligible, out=offsets[1:])
    population = int(offsets[-1])
    count = min(population, sample_size)
    if count == population:
        chosen = numpy.arange(population, dtype=numpy.int64)
    else:
        chosen = numpy.sort(numpy.random.default_rng(seed).choice(
            population, size=count, replace=False))
    rows = numpy.searchsorted(offsets[1:], chosen, side='right')
    positions = chosen - offsets[rows] + 1
    return gather_vectors(embeddings, rows, positions, batch_size)


def nearest_centroids(vectors, centroids, centroid_norms):
    # ||x - c||^2 = ||x||^2 + ||c||^2 - 2*x.c. The first term is identical
    # for every centroid, so it is unnecessary when choosing the closest one.
    distances = vectors @ centroids.T
    distances.mul_(-2).add_(centroid_norms[None, :])
    return distances.argmin(dim=1)


class TorchKmeans:
    def __init__(self, centroids):
        self.centroids = centroids.contiguous()
        self.centroid_norms = self.centroids.square().sum(dim=1)

    @property
    def index(self):
        # Retain the search interface used by assign_clusters.
        return self

    def search(self, vectors, k):
        if k != 1:
            raise ValueError('only the nearest centroid is supported')
        with torch.no_grad():
            batch = torch.as_tensor(vectors, device=self.centroids.device)
            labels = nearest_centroids(batch, self.centroids,
                self.centroid_norms)
            return None, labels.cpu().numpy()[:, None]


def Kmeans_torch(vecs, K, max_iterations=100, batch_size=8192, seed=42):
    if len(vecs) < K:
        raise ValueError(f'K-means needs at least {K} training vectors; got {len(vecs)}')
    if not torch.cuda.is_available():
        raise RuntimeError('PyTorch CUDA is required for K-means on this GPU job')

    rng = numpy.random.default_rng(seed)
    device = torch.device('cuda')
    with torch.no_grad():
        data = torch.as_tensor(vecs, device=device)
        initial = rng.choice(len(vecs), size=K, replace=False)
        centroids = data[torch.as_tensor(initial, device=device)].clone()
        ones = torch.ones(batch_size, dtype=torch.int32, device=device)

        for _ in tqdm(range(max_iterations), desc='fitting K-means centroids'):
            centroid_norms = centroids.square().sum(dim=1)
            sums = torch.zeros_like(centroids)
            counts = torch.zeros(K, dtype=torch.int32, device=device)
            for start in range(0, len(data), batch_size):
                batch = data[start:start + batch_size]
                labels = nearest_centroids(batch, centroids, centroid_norms)
                sums.index_add_(0, labels, batch)
                counts.index_add_(0, labels, ones[:len(batch)])

            occupied = counts > 0
            centroids[occupied] = sums[occupied] / counts[occupied].unsqueeze(1)
            empty = torch.where(~occupied)[0]
            if len(empty):
                replacements = rng.integers(len(data), size=len(empty))
                centroids[empty] = data[torch.as_tensor(replacements, device=device)]

        return TorchKmeans(centroids)


def assign_clusters(kmeans, seqs, embeddings, offsets, batch_size):
    """Assign every non-EOS token; retain only labels and token IDs."""
    total = int(offsets[-1])
    labels = numpy.empty(total, dtype=numpy.int32)
    token_ids = numpy.empty(total, dtype=numpy.int64)
    for start in tqdm(range(0, total, batch_size), desc='assigning clusters'):
        stop = min(start + batch_size, total)
        flat_positions = numpy.arange(start, stop, dtype=numpy.int64)
        rows = numpy.searchsorted(offsets[1:], flat_positions, side='right')
        positions = flat_positions - offsets[rows]
        vectors = embeddings[rows, positions].float().numpy()
        _, nearest = kmeans.index.search(vectors, 1)
        labels[start:stop] = nearest[:, 0]
        token_ids[start:stop] = seqs[rows, positions].numpy()
    return labels, token_ids


def update_flows(alpha, beta, gamma, labels, token_ids, lengths, offsets,
        sequence_length, hidden_states, eos_token_id):
    """Count emissions, initial states, and within-sequence transitions."""
    alpha_counts = alpha.numpy()
    beta_counts = beta.numpy()
    gamma_counts = gamma.numpy()
    numpy.add.at(beta_counts, (labels, token_ids), 1.0)

    nonempty = lengths > 0
    starts = offsets[:-1][nonempty]
    ends = offsets[1:][nonempty]
    numpy.add.at(gamma_counts, labels[starts], 1.0)

    # Every adjacent pair within a sequence is a transition. A sequence whose
    # first EOS is within the padded width also transitions to the reserved EOS
    # state. The old code skipped this final transition if there was no EOS.
    if len(labels) > 1:
        adjacent = numpy.ones(len(labels) - 1, dtype=numpy.bool_)
        adjacent[ends[ends < len(labels)] - 1] = False
        numpy.add.at(alpha_counts,
            (labels[:-1][adjacent], labels[1:][adjacent]), 1.0)
    has_eos = nonempty & (lengths < sequence_length)
    last_with_eos = offsets[1:][has_eos] - 1
    numpy.add.at(alpha_counts,
        (labels[last_with_eos], hidden_states - 1), 1.0)

    alpha[hidden_states - 1, hidden_states - 1] = 1.0
    beta[hidden_states - 1, eos_token_id] = 1.0


def write_params(alpha_flow, beta_flow, gamma_flow, pseudocount,
    hidden_states, vocab_size, eos_token_id, output_file):
    alpha_flow.add_(pseudocount / alpha_flow.shape[-1])
    beta_flow.add_(pseudocount / beta_flow.shape[-1])
    gamma_flow.add_(pseudocount / gamma_flow.shape[-1])

    # Normalize in place so this step does not copy the large emission matrix.
    alpha_flow.div_(alpha_flow.sum(dim=-1, keepdim=True))
    beta_flow.div_(beta_flow.sum(dim=-1, keepdim=True)).log_()
    gamma_flow.div_(gamma_flow.sum()).log_()

    hmm_model = HMM(hidden_states, vocab_size, eos_token_id)
    hmm_model.update_params(alpha_flow, beta_flow, gamma_flow)
    hmm_model.save_pretrained(output_file)


def main():
    args = init()
    torch.manual_seed(args.seed)
    hidden_states = args.hidden_states
    vocab_size = args.vocab_size
    eos_token_id = args.eos_token_id

    seqs, embeddings, lengths, offsets = load_examples(
        args.sequences_file, args.embeddings_file, eos_token_id)
    vecs = sample_training_vectors(embeddings, lengths,
        args.kmeans_sample_size, args.assignment_batch_size, args.seed)
    print(f'training K-means with {hidden_states - 1} clusters and '
        f'{len(vecs)} sampled embeddings ...', flush=True)
    kmeans = Kmeans_torch(vecs, hidden_states - 1,
        max_iterations=args.kmeans_iterations,
        batch_size=args.assignment_batch_size, seed=args.seed)
    del vecs

    print(f'clustering all {offsets[-1]} non-EOS embeddings ...', flush=True)
    labels, token_ids = assign_clusters(kmeans, seqs, embeddings, offsets,
        args.assignment_batch_size)
    del kmeans, embeddings

    alpha = torch.zeros(hidden_states, hidden_states)
    beta = torch.zeros(hidden_states, vocab_size)
    gamma = torch.zeros(hidden_states)
    print('computing flows ...', flush=True)
    update_flows(alpha, beta, gamma, labels, token_ids, lengths, offsets,
        seqs.shape[1], hidden_states, eos_token_id)
    del labels, token_ids, seqs

    print(f'storing parameters to {args.output_file} ...', flush=True)
    write_params(alpha, beta, gamma, args.pseudocount,
        hidden_states, vocab_size, eos_token_id, args.output_file)


if __name__ == '__main__':
    main()
