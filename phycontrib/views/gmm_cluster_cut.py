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
            logger.info('Fitting GMM...')
            gmm = gm.GaussianMixture(n_components)
            labels = gmm.fit_predict(pc)
            logger.info('...done!')
            labels_ = np.unique(labels)
            for label in labels_[:-1]:
                split_spike_ids = spike_ids[np.where(labels == label)]
                clustering.split(split_spike_ids)

        @controller.supervisor.connect
        def on_create_cluster_views():

            @controller.supervisor.actions.add(alias='gmm')
            def automated_cluster_cut(n_components):
                if n_components < 2:
                    logger.error('GMM cluster cut requires at least 2 components in mixture')
                else:
                    logger.info('Starting GMM cluster cut with %d components in mixture...' % n_components)
                    cluster_ids = controller.supervisor.selected
                    _fit_gmm(cluster_ids, n_components)


