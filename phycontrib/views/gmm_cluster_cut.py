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
            template_id = controller.get_template_for_cluster(cluster_ids[0])
            channel_ids = controller.model.get_template(template_id).channel_ids
            pc_ = controller.model.get_features(spike_ids, channel_ids)
            # pc_ = controller.model.features[spike_ids]
            pc = np.reshape(pc_, (pc_.shape[0], pc_.shape[1] * pc_.shape[2]))
            if n_components == 'var':
                labels = _fit_gmm_var_n_components(pc)
            else:
                labels = _fit_gmm_fixed_n_components(pc, n_components)
            labels_ = np.unique(labels)
            for label in labels_[:-1]:
                split_spike_ids = spike_ids[np.where(labels == label)]
                controller.supervisor.split(split_spike_ids)

        def _fit_gmm_on_selected_channels(cluster_ids, n_components, channel_0_str, channel_1_str):
            clustering = controller.supervisor.clustering
            for cluster in cluster_ids:
                logger.info('%d spikes in cluster %d' % (len(clustering.spikes_per_cluster[cluster]), cluster))
            spike_ids = clustering.spikes_in_clusters(cluster_ids)
            logger.info('%d spikes in all clusters' % len(spike_ids))

            # figure out PCs on selected channels
            s = 'AaBbCc' # PC 1-3
            channel_0_ID = int(channel_0_str[:-1])
            channel_0_PC_ = channel_0_str[-1]
            channel_0_PC = s.index(channel_0_PC_) // 2
            channel_1_ID = int(channel_1_str[:-1])
            channel_1_PC_ = channel_1_str[-1]
            channel_1_PC = s.index(channel_1_PC_) // 2

            # reverse lookup of channel IDs;
            channel_0 = np.where(controller.model.channel_mapping == channel_0_ID)[0][0]
            channel_1 = np.where(controller.model.channel_mapping == channel_1_ID)[0][0]
            # handle case where we want to use PCs on same channel
            if channel_0 == channel_1:
                channel_ids = (channel_0, )
                channels_PC_lookup = (0, 0)
            else:
                channel_ids = (channel_0, channel_1)
                channels_PC_lookup = (0, 1)

            # figure out which channels the PC order corresponds to?
            # look up values of the PCs on those channels
            cluster_ids = controller.supervisor.selected
            spike_ids = controller.supervisor.clustering.spikes_in_clusters(cluster_ids)
            data_ = controller.model.get_features(spike_ids, channel_ids)
            # shape of features: (n_spikes, n_channels (i.e., 1 or 2), n_PCs)
            pc = np.zeros((data_.shape[0], 2))
            pc[:, 0] = data_[:, channels_PC_lookup[0], channel_0_PC]
            pc[:, 1] = data_[:, channels_PC_lookup[1], channel_1_PC]

            labels = _fit_gmm_fixed_n_components(pc, n_components)
            labels_ = np.unique(labels)
            for label in labels_[:-1]:
                split_spike_ids = spike_ids[np.where(labels == label)]
                controller.supervisor.split(split_spike_ids)

        @controller.supervisor.connect
        def on_create_cluster_views():

            controller.supervisor.actions.separator()

            @controller.supervisor.actions.add(alias='gmm')
            def automated_cluster_cut(n_components, channel_0_str=None, channel_1_str=None):
                if channel_0_str is None and channel_1_str is None:
                    if n_components == 'var':
                        logger.info('Starting GMM cluster cut with unknown number of components in mixture...')
                        cluster_ids = controller.supervisor.selected
                        _fit_gmm(cluster_ids, n_components)
                    elif n_components >= 2:
                        logger.info('Starting GMM cluster cut with %d components in mixture...' % n_components)
                        cluster_ids = controller.supervisor.selected
                        _fit_gmm(cluster_ids, n_components)
                    else:
                        logger.error('GMM cluster cut requires at least 2 components in mixture; '
                                     'or `var` to estimate these')
                else:
                    if n_components < 2:
                        logger.error('GMM cluster cut requires at least 2 components in mixture')
                    logger.info('Starting GMM cluster cut on channels %s and %s with %d components in mixture...'
                                % (channel_0_str, channel_1_str, n_components))
                    cluster_ids = controller.supervisor.selected
                    _fit_gmm_on_selected_channels(cluster_ids, n_components, channel_0_str, channel_1_str)



class RemoveOutliers(IPlugin):

    def attach_to_controller(self, controller):

        @controller.supervisor.connect
        def on_create_cluster_views():

            @controller.supervisor.actions.add(alias='rmo')
            def remove_outliers(channel_0_str, channel_1_str, percentile):
                s = 'AaBbCc' # PC 1-3
                channel_0_ID = int(channel_0_str[:-1])
                channel_0_PC_ = channel_0_str[-1]
                channel_0_PC = s.index(channel_0_PC_) // 2
                channel_1_ID = int(channel_1_str[:-1])
                channel_1_PC_ = channel_1_str[-1]
                channel_1_PC = s.index(channel_1_PC_) // 2

                # reverse lookup of channel IDs;
                channel_0 = np.where(controller.model.channel_mapping == channel_0_ID)[0][0]
                channel_1 = np.where(controller.model.channel_mapping == channel_1_ID)[0][0]
                # handle case where we want to use PCs on same channel
                if channel_0 == channel_1:
                    channel_ids = (channel_0, )
                    channels_PC_lookup = (0, 0)
                else:
                    channel_ids = (channel_0, channel_1)
                    channels_PC_lookup = (0, 1)

                # figure out which channels the PC order corresponds to?
                # look up values of the PCs on those channels
                cluster_ids = controller.supervisor.selected
                spike_ids = controller.supervisor.clustering.spikes_in_clusters(cluster_ids)
                data_ = controller.model.get_features(spike_ids, channel_ids)
                # shape of features: (n_spikes, n_channels (i.e., 1 or 2), n_PCs)
                data = np.zeros((data_.shape[0], 2))
                data[:, 0] = data_[:, channels_PC_lookup[0], channel_0_PC]
                data[:, 1] = data_[:, channels_PC_lookup[1], channel_1_PC]

                # compute mean and covariance matrix and normalized distance
                mu = np.mean(data, axis=0)
                data_centered = data - mu
                data_centered = data_centered.transpose()
                sigma = np.dot(data_centered, data_centered.transpose()) / (data_centered.shape[1] - 1.0)
                sigma_inv = np.linalg.pinv(sigma)
                spike_distances = np.zeros(data_centered.shape[1])
                for i in range(len(spike_distances)):
                    spike_distances[i] = np.sqrt(np.dot(data_centered[:, i].transpose(), np.dot(sigma_inv, data_centered[:, i])))

                # select all above/below percentile parameter and cut
                if not 0.0 < percentile < 1.0:
                    logger.error('Percentile has to be in the interval (0, 1)!')
                else:
                    distance = np.percentile(spike_distances, percentile*100.0)
                    split_spike_ids = spike_ids[np.where(spike_distances <= distance)]
                    percentile_kept = 1.0 * len(split_spike_ids) / len(spike_ids)
                    logger.info('Distance: %.4f; kept %.4f of spikes; %.4f requested' % (distance, percentile_kept, percentile))
                    controller.supervisor.split(split_spike_ids)

