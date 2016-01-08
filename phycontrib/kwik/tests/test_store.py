# -*- coding: utf-8 -*-

"""Test cluster store."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from ..store import create_cluster_store
from phy.io import Selector


#------------------------------------------------------------------------------
# Test cluster stats
#------------------------------------------------------------------------------

def test_create_cluster_store(model):
    spc = lambda c: model.spikes_per_cluster[c]
    selector = Selector(spike_clusters=model.spike_clusters,
                        spikes_per_cluster=spc)
    cs = create_cluster_store(model, selector=selector)

    nc = model.n_channels
    nfpc = model.n_features_per_channel
    ns = len(model.spikes_per_cluster[1])
    ns2 = len(model.spikes_per_cluster[2])
    nsw = model.n_samples_waveforms

    def _check(out, name, *shape):
        spikes = out.pop('spike_ids')
        arr = out[name]
        assert spikes.shape[0] == shape[0]
        assert arr.shape == shape

    # Model data.
    _check(cs.masks(1), 'masks', ns, nc)
    _check(cs.features(1), 'features', ns, nc, nfpc)
    _check(cs.waveforms(1), 'waveforms', ns, nsw, nc)
    _check(cs.waveforms_masks(1), 'waveforms', ns, nsw, nc)
    _check(cs.waveforms_masks(1), 'masks', ns, nc)

    # Background feature masks.
    data = cs.background_features_masks()
    spike_ids = data.spike_ids
    bgf = data.features
    bgm = data.masks
    assert bgf.ndim == 3
    assert bgf.shape[1:] == (nc, nfpc)
    assert bgm.ndim == 2
    assert bgm.shape[1] == nc
    assert spike_ids.shape == (bgf.shape[0],) == (bgm.shape[0],)

    # Test concat multiple clusters.
    data = cs.features_masks([1, 2])
    spike_ids = data.spike_ids
    f = data.features
    m = data.masks
    assert len(spike_ids) == ns + ns2
    assert f.shape == (ns + ns2, nc, nfpc)
    assert m.shape == (ns + ns2, nc)

    # Test means.
    assert cs.mean_masks(1).shape == (nc,)
    assert cs.mean_features(1).shape == (nc, nfpc)
    assert cs.mean_waveforms(1).shape == (nsw, nc)

    # Limits.
    assert 0 < cs.waveform_lim() < 3
    assert 0 < cs.feature_lim() < 3
    assert cs.mean_traces().shape == (nc,)

    # Statistics.
    assert 1 <= len(cs.best_channels(1)) <= nc
    assert 1 <= len(cs.best_channels_multiple([1, 2])) <= nc
    assert 0 < cs.max_waveform_amplitude(1) < 1
    assert cs.mean_masked_features_score(1, 2) > 0

    assert np.array(cs.most_similar_clusters(1)).shape == (3, 2)