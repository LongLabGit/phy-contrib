"""GMM cluster cut plugin.

This plugin adds automated cluster cutting using a gaussian mixture model.
The number of clusters in the mixture are a user-supplied parameter.
The model is fit on the full PC feature space, and clusters are then
automatically cut based on the GMM predictions for each spike.

To activate the plugin, copy this file to `~/.phy/plugins/` and add this line
to your `~/.phy/phy_config.py`:

```python
c.TemplateGUI.plugins = ['AutomatedClusterCut']
```
"""

import logging

import numpy as np
import sklearn.mixture.gaussian_mixture as gm

from phy import IPlugin

logger = logging.getLogger(__name__)


class AutomatedClusterCut(IPlugin):

    def attach_to_controller(self, controller):

        def _fit_gmm_fixed_n_components(pc, n_components):
            logger.info('Fitting GMM...')
            gmm = gm.GaussianMixture(n_components)
            labels = gmm.fit_predict(pc)
            logger.info('...done!')
            return labels

        def _fit_gmm_var_n_components(pc, max_components=None):
            if max_components is None:
                max_comp = 10
            else:
                max_comp = max_components

            min_bic = np.inf
            min_comp = 0
            min_labels = []
            logger.info('Fitting GMM and trying to estimate optimal number of components...')
            for n_components in range(1, max_comp + 1):
                gmm = gm.GaussianMixture(n_components)
                gmm.fit(pc)
                current_bic = gmm.bic(pc)
                if current_bic < min_bic:
                    min_bic = current_bic
                    min_comp = n_components
                    min_labels = gmm.predict(pc)

            logger.info('...done!')
            logger.info('Estimated number of components: %d' % min_comp)
            return min_labels

        def _fit_gmm(cluster_ids, n_components):
            clustering = controller.supervisor.clustering
            for cluster in cluster_ids:
                logger.info('%d spikes in cluster %d' % (len(clustering.spikes_per_cluster[cluster]), cluster))
            spike_ids = clustering.spikes_in_clusters(cluster_ids)
            logger.info('%d spikes in all clusters' % len(spike_ids))
            # assume that PCs have already been loaded...
            # pc_.shape = n_spikes, n_channels_loc, n_pcs
            # since we want to use the PCs on all channels where they
            # have been calculated, we need:
            # pc.shape = n_spikes, n_channels_loc * n_pcs
            pc_ = controller.model.features[spike_ids]
            pc = np.reshape(pc_, (pc_.shape[0], pc_.shape[1] * pc_.shape[2]))
            if n_components == 'var':
                labels = _fit_gmm_var_n_components(pc)
            else:
                labels = _fit_gmm_fixed_n_components(pc, n_components)
            labels_ = np.unique(labels)
            for label in labels_[:-1]:
                split_spike_ids = spike_ids[np.where(labels == label)]
                controller.supervisor.split(split_spike_ids)

        @controller.supervisor.connect
        def on_create_cluster_views():

            @controller.supervisor.actions.add(alias='gmm')
            def automated_cluster_cut(n_components):
                # TODO: add possibility to use PCs from specific channels
                if n_components == 'var':
                    logger.info('Starting GMM cluster cut with unknown number of components in mixture...')
                    cluster_ids = controller.supervisor.selected
                    _fit_gmm(cluster_ids, n_components)
                elif n_components >= 2:
                    logger.info('Starting GMM cluster cut with %d components in mixture...' % n_components)
                    cluster_ids = controller.supervisor.selected
                    _fit_gmm(cluster_ids, n_components)
                else:
                    logger.error('GMM cluster cut requires at least 2 components in mixture; or -1 to estimate these')


