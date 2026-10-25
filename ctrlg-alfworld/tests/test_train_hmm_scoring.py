"""Regression checks for comparable, fixed train/dev HMM evaluation."""

import importlib.util
import json
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ctrlg import HMM
from ctrlg_alfworld.checkpoint_selection import read_latest_log_likelihoods, select_best_checkpoint


SOURCE = Path(__file__).resolve().parents[2] / 'distillation/train_hmm.py'
SPEC = importlib.util.spec_from_file_location('train_hmm_scoring', SOURCE)
trainer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trainer)


def small_hmm():
    model = HMM(2, 5, 0)
    model.update_params(torch.tensor([[.8, .2], [.3, .7]]),
        torch.tensor([[.05, .2, .3, .4, .05], [.3, .1, .1, .1, .4]]).log(),
        torch.tensor([.6, .4]).log())
    return model


TRAIN = torch.tensor([[1, 2, 0, -2], [2, 1, 3, -2], [3, 4, -2, -2],
                      [4, 1, 2, 0], [1, 3, 4, -2], [2, 4, 0, -2]])
CONTEXTS = [1, 2, 0, 3, 1, 2]
DEV = TRAIN.flip(0)[:5].clone()
DEV_CONTEXTS = list(reversed(CONTEXTS))[:5]


def spans(data, contexts):
    return [dict(length=int((row != -2).sum()), context_length=context,
                 completed=bool((row == 0).any())) for row, context in zip(data, contexts)]


def write_dataset(root, *, versioned=True):
    torch.save(DEV, root / 'data.dev')
    # Unequal chunks test global span offsets, including a subset boundary
    # inside the final chunk. These are the builder's flattened row order.
    for index, chunk in enumerate((TRAIN[:3], TRAIN[3:4], TRAIN[4:])):
        torch.save(chunk, root / f'data.train.{index}')
    if versioned:
        (root / 'data.train.spans.json').write_text(json.dumps(spans(TRAIN, CONTEXTS)))
        (root / 'data.dev.spans.json').write_text(json.dumps(spans(DEV, DEV_CONTEXTS)))


def conditional_mean(model, data, contexts):
    total = 0.0
    for row, context in zip(data, contexts):
        observed = row[row != -2][None, :]
        prefix = float(model.forward(observed[:, :context])[-1]) if context else 0.0
        total += float(model.forward(observed)[-1]) - prefix
    return total / len(data)


def test_fixed_subset_preserves_global_context_alignment(tmp_path):
    write_dataset(tmp_path)
    first = trainer.load_evaluation_data(tmp_path, 'data', 3)
    torch.manual_seed(987)
    second = trainer.load_evaluation_data(tmp_path, 'data', 3)
    train, dev, contexts, dev_contexts = first
    assert torch.equal(train, TRAIN[:5])
    assert torch.equal(dev, DEV)
    assert contexts == CONTEXTS[:5]
    assert dev_contexts == DEV_CONTEXTS
    assert torch.equal(second[0], train) and second[2] == contexts
    # Subsequent EM dropout on newly loaded chunks cannot mutate fixed rows.
    chunk = torch.load(tmp_path / 'data.train.0')
    trainer.apply_dropout(chunk, 1, 5, 0)
    assert torch.equal(train, TRAIN[:5])


@pytest.mark.parametrize('world_size', [1, 2, 8])
def test_shards_cover_every_row_and_context_once(world_size):
    shards = [trainer.shard_evaluation_data(TRAIN[:5], CONTEXTS[:5], rank, world_size)
              for rank in range(world_size)]
    assert torch.equal(torch.cat([data for data, _ in shards]), TRAIN[:5])
    assert [context for _, contexts in shards for context in contexts] == CONTEXTS[:5]


@pytest.mark.parametrize('split', ['train', 'dev'])
def test_partial_span_metadata_is_rejected(tmp_path, split):
    write_dataset(tmp_path)
    (tmp_path / f'data.{split}.spans.json').unlink()
    with pytest.raises(ValueError, match='both required'):
        trainer.load_evaluation_data(tmp_path, 'data', 3)


@pytest.mark.parametrize('mutation, message', [
    ('short_train', 'spans do not match'),
    ('long_train', 'total training rows'),
    ('short_dev', 'spans do not match'),
    ('context', 'invalid observed/context length'),
    ('length', 'right padding'),
])
def test_bad_span_alignment_is_rejected(tmp_path, mutation, message):
    write_dataset(tmp_path)
    split = 'dev' if mutation == 'short_dev' else 'train'
    path = tmp_path / f'data.{split}.spans.json'
    metadata = json.loads(path.read_text())
    if mutation.startswith('short'):
        metadata.pop()
    elif mutation == 'long_train':
        metadata.append(metadata[-1])
    elif mutation == 'context':
        metadata[3]['context_length'] = 5
    else:
        metadata[4]['length'] = 4
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=message):
        trainer.load_evaluation_data(tmp_path, 'data', 3)


def test_versioned_data_cannot_be_truncated(tmp_path):
    write_dataset(tmp_path)
    with pytest.raises(ValueError, match='sample_length would censor'):
        trainer.load_evaluation_data(tmp_path, 'data', 3, sample_length=2)


def test_training_logs_conditional_scores_on_same_rows_after_each_update(tmp_path, monkeypatch):
    write_dataset(tmp_path)
    model = small_hmm()
    expected, seen = [], []
    original_score = model.conditional_loglikelihood

    def record_score(data, contexts, batch_size):
        seen.append((data.clone(), list(contexts)))
        expected.append(conditional_mean(model, data, contexts))
        return original_score(data, contexts, batch_size)

    monkeypatch.setattr(model, 'conditional_loglikelihood', record_score)
    monkeypatch.setattr(model, 'loglikelihood', lambda *args: pytest.fail('joint evaluation used'))
    monkeypatch.setattr(trainer.HMM, 'from_pretrained', lambda *args, **kwargs: model)
    monkeypatch.setattr(trainer.dist, 'all_reduce', lambda *args, **kwargs: None)
    monkeypatch.setattr(trainer, 'tqdm', lambda values: values)
    monkeypatch.setattr(model, 'save_pretrained', lambda path: Path(path).mkdir(exist_ok=True))
    model_path = tmp_path / 'model'
    (model_path / 'checkpoint-5').mkdir(parents=True)
    log = tmp_path / 'train.log'
    trainer.train_hmm(0, 1, model_path, 5, 100, tmp_path, 'data', 3, 2, None,
                      [(2, 1)], log, 1.0, .001, device='cpu')

    metrics = read_latest_log_likelihoods(log)
    assert list(metrics) == [5, 6, 7]
    assert len(seen) == 6
    for index, values in enumerate(metrics.values()):
        assert values['train_log_likelihood'] == pytest.approx(expected[2 * index], abs=2e-6)
        assert values['dev_log_likelihood'] == pytest.approx(expected[2 * index + 1], abs=2e-6)
        assert torch.equal(seen[2 * index][0], TRAIN[:5])
        assert seen[2 * index][1] == CONTEXTS[:5]
        assert torch.equal(seen[2 * index + 1][0], DEV)
        assert seen[2 * index + 1][1] == DEV_CONTEXTS
    assert (model_path / 'checkpoint-7').is_dir()  # final step off the save interval
    selected = select_best_checkpoint(log, model_path)
    assert selected['selection_metric'] == 'mean_output_conditional_dev_log_likelihood_per_attempt'
    assert selected['score_definitions'] == dict(train=trainer.CONDITIONAL_SCORE, dev=trainer.CONDITIONAL_SCORE)
    assert selected['best_checkpoint'] == max((5, 7), key=lambda step: metrics[step]['dev_log_likelihood'])


def test_legacy_joint_scores_are_explicit_and_use_same_truncation(tmp_path, monkeypatch):
    write_dataset(tmp_path, versioned=False)
    model = small_hmm()
    monkeypatch.setattr(trainer.HMM, 'from_pretrained', lambda *args, **kwargs: model)
    monkeypatch.setattr(trainer.dist, 'all_reduce', lambda *args, **kwargs: None)
    monkeypatch.setattr(model, 'conditional_loglikelihood', lambda *args: pytest.fail('legacy conditional score'))
    monkeypatch.setattr(model, 'save_pretrained', lambda path: Path(path).mkdir(parents=True))
    log = tmp_path / 'train.log'
    trainer.train_hmm(0, 1, tmp_path / 'model', 0, 100, tmp_path, 'data', 3, 2, 2,
                      [], log, 0, .001, device='cpu')
    values = read_latest_log_likelihoods(log)[0]
    assert values['train_log_likelihood'] == pytest.approx(float(model.loglikelihood(TRAIN[:5, :2], 2)) / 5)
    assert values['dev_log_likelihood'] == pytest.approx(float(model.loglikelihood(DEV[:, :2], 2)) / 5)
    selected = select_best_checkpoint(log, tmp_path / 'model')
    assert selected['selection_metric'] == 'mean_joint_dev_log_likelihood_per_sequence'
    assert selected['score_definitions'] == dict(train=trainer.JOINT_SCORE, dev=trainer.JOINT_SCORE)


def _distributed_scores(rank, store, output):
    dist.init_process_group('gloo', init_method=Path(store).as_uri(), rank=rank, world_size=3,
                            timeout=timedelta(seconds=30))
    try:
        # Smaller train set than dev; rank zero has no local training rows.
        train, contexts = trainer.shard_evaluation_data(TRAIN[:2], CONTEXTS[:2], rank, 3)
        dev, dev_contexts = trainer.shard_evaluation_data(DEV, DEV_CONTEXTS, rank, 3)
        scores = trainer.evaluation_scores(small_hmm(), train, dev, contexts, dev_contexts, 2)
        Path(output, f'{rank}.json').write_text(json.dumps(scores))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason='Gloo unavailable')
def test_distributed_scores_reduce_sums_and_actual_counts_with_empty_ranks(tmp_path):
    mp.spawn(_distributed_scores, args=(str(tmp_path / 'store'), str(tmp_path)), nprocs=3, join=True)
    model = small_hmm()
    expected = [conditional_mean(model, TRAIN[:2], CONTEXTS[:2]),
                conditional_mean(model, DEV, DEV_CONTEXTS)]
    for rank in range(3):
        assert json.loads((tmp_path / f'{rank}.json').read_text()) == pytest.approx(expected, abs=2e-6)


@pytest.mark.parametrize('contexts', [[1], [-1, 1], [1, 4], [1., 1.], [True, False], [[1, 1]]])
def test_conditional_likelihood_rejects_misaligned_contexts(contexts):
    with pytest.raises(ValueError, match='prefix lengths|context extends'):
        small_hmm().conditional_loglikelihood(TRAIN[:2], contexts, 2)


def test_conditional_likelihood_handles_mixed_contexts_censoring_and_padding():
    model = small_hmm()
    expected = conditional_mean(model, TRAIN, CONTEXTS) * len(TRAIN)
    for batch_size in (1, 4, 8):
        actual = float(model.conditional_loglikelihood(TRAIN, CONTEXTS, batch_size))
        assert actual == pytest.approx(expected, abs=3e-6)
        assert actual != pytest.approx(float(model.loglikelihood(TRAIN, batch_size)))
