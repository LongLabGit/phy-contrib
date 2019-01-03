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
import sklearn.mixture.gaussian_mixture as gmm

from phy import IPlugin

logger = logging.getLogger(__name__)


class AutomatedClusterCut(IPlugin):

    def attach_to_controller(self, controller):

        def _fit_gmm(clusters):
            clustering = controller.clustering
            for cluster in clusters:
                print('%d spikes in cluster %d' % (clustering[cluster].spike_counts, cluster))

        @controller.supervisor.connect
        def on_create_cluster_views():

            @controller.supervisor.actions.add(alias='gmm')
            def automated_cluster_cut(n_clusters):
                print('Starting GMM cluster cut with %d clusters...' % n_clusters)
                cluster_ids = controller.supervisor.selected
                _fit_gmm(cluster_ids)


