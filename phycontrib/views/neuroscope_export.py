"""Neuroscope export plugin.

Automated export of clusters labeled as 'good' to Neuroscope format.
Clusters are sorted by shank (i.e., corresponding to Neuroscope anatomical group)

To activate the plugin, copy this file to `~/.phy/plugins/` and add this line
to your `~/.phy/phy_config.py`:

```python
c.TemplateGUI.plugins = ['AutomatedClusterCut']
```
"""

import logging
import os.path
import numpy as np

from phy import IPlugin

logger = logging.getLogger(__name__)


class ExportToNeuroscope(IPlugin):

    def attach_to_controller(self, controller):

        @controller.supervisor.connect
        def on_create_cluster_views():

            controller.supervisor.actions.separator()

            @controller.supervisor.actions.add()
            def export_to_neuroscope():
                clustering = controller.supervisor.clustering
                spikes_per_cluster = clustering.spikes_per_cluster
                spike_samples = controller.model.spike_samples

                # get all cluster IDs labeled as 'good'
                cluster_labels = controller.supervisor.get_labels('group')
                good_clusters = []
                for cluster in cluster_labels:
                    if cluster_labels[cluster] == 'good':
                        good_clusters.append(cluster)

                logger.info('Saving %d clusters in Neuroscope format.' % len(good_clusters))

                # group everything according to shank
                shanks = np.unique(controller.model.channel_shank_mapping)
                clusters_per_shank = {}
                for shank in shanks:
                    clusters_per_shank[shank] = []
                for cluster in good_clusters:
                    channel = controller.get_best_channel(cluster)
                    shank = controller.model.channel_shank_mapping[channel]
                    clusters_per_shank[shank].append(cluster)

                for shank in clusters_per_shank:
                    clusters = clusters_per_shank[shank]
                    nr_clusters = len(clusters)
                    if not nr_clusters:
                        continue
                    # collect all spike time samples of 'good' clusters
                    good_spike_samples = []
                    good_spike_cluster_ids = []
                    for cluster in clusters:
                        spike_ids = spikes_per_cluster[cluster]
                        good_spike_samples.extend(spike_samples[spike_ids])
                        good_spike_cluster_ids.extend([cluster for i in range(len(spike_ids))])

                    # sort cluster ids and spike samples by spike sample (ascending)
                    good_spike_samples = np.array(good_spike_samples)
                    good_spike_cluster_ids = np.array(good_spike_cluster_ids)
                    sorting = np.argsort(good_spike_samples)
                    spike_samples_sorted = good_spike_samples[sorting]
                    cluster_ids_sorted = good_spike_cluster_ids[sorting]

                    # save clusters and spike times on current shank
                    dir_path = controller.model.dir_path
                    cluster_fname = os.path.join(dir_path, 'clusters.clu.' + str(shank))
                    spike_sample_fname = os.path.join(dir_path, 'clusters.res.' + str(shank))
                    with open(cluster_fname, 'w') as cluster_file:
                        out_str = str(nr_clusters)
                        out_str += '\n'
                        for cluster in cluster_ids_sorted:
                            out_str += str(cluster)
                            out_str += '\n'
                        cluster_file.write(out_str)
                    with open(spike_sample_fname, 'w') as spike_sample_file:
                        out_str = ''
                        for spike_sample in spike_samples_sorted:
                            out_str += str(spike_sample)
                            out_str += '\n'
                        spike_sample_file.write(out_str)
