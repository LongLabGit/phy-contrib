import numpy as np
from phy import IPlugin
from pathlib import Path

# from scipy.cluster.vq import kmeans2, whiten

import logging
logger = logging.getLogger(__name__)

# try:
#     from klusta.launch import cluster
# except ImportError:  # pragma: no cover
#     logger.warn("Package klusta not installed: the KwikGUI will not work.")

class ControllerSettings(IPlugin):
    def attach_to_controller(self, controller):
        controller.n_spikes_features = 15000
        controller.n_spikes_waveforms = 500
        @controller.supervisor.connect
        def on_create_cluster_views():
            @controller.supervisor.add_column
            def FiringRate(cluster_id):
                return '%.2f Hz' % (len(controller._get_spike_times(cluster_id, 'None').data) / controller.model.duration)

            # @controller.supervisor.add_column
            # def HorzPos(cluster_id):
            #     channel_id = controller.get_best_channel(cluster_id)
            #     return controller.model.channel_positions[channel_id][0]

            my_file = Path("channel_shank_map.npy")
            if my_file.is_file():
                @controller.supervisor.add_column
                def Shank(cluster_id):
                    def _load_channel_shanks(self):
                        return self._read_array('channel_shank_map')
                    channel_shanks = _load_channel_shanks(controller.model)
                    channel_id = controller.get_best_channel(cluster_id)
                    return channel_shanks[channel_id]

            # @controller.supervisor.actions.add
            # def recluster():
            #     """Relaunch KlustaKwik on the selected clusters."""
            #     # Selected clusters.
            #     cluster_ids = supervisor.selected
            #     spike_ids = self.selector.select_spikes(cluster_ids)
            #     logger.info("Running KlustaKwik on %d spikes.", len(spike_ids))
            #
            #     # Run KK2 in a temporary directory to avoid side effects.
            #     n = 10
            #     with TemporaryDirectory() as tempdir:
            #         spike_clusters, metadata = cluster(self.model,
            #                                            spike_ids,
            #                                            num_starting_clusters=n,
            #                                            tempdir=tempdir,
            #                                            )
            #     controller.supervisor.split(spike_ids, spike_clusters)

            # @controller.supervisor.actions.add
            # def K_means():
            #     """Cut Clusters by means of K-means clustering"""
            #     # Selected clusters.
            #     logger.warn("Running K means clustering...")
            #     cluster_ids = controller.supervisor.selected
            #
            #     spike_ids = controller.selector.select_spikes(cluster_ids)
            #     s = controller.supervisor.clustering.spikes_in_clusters(cluster_ids)
            #     data = controller.model._load_features()
            #     data3 = data.data[spike_ids]
            #     data2 = np.reshape(data3,(data3.shape[0],data3.shape[1]*data3.shape[2]))
            #     logger.warn("Whitening the PCAs")
            #     whitened = whiten(data2)
            #     logger.warn("Running Kmeans")
            #     clusters_out,label = kmeans2(whitened,2)
            #     controller.supervisor.split(s,label)
            #     logger.warn("K means clustering complete")
            #
            # @controller.supervisor.actions.add
            # def MahalanobisDist(x, y):
            #     """Remove outliers defined by the mahalanobis distance."""
            #     # Selected clusters.
            #     logger.warn("Running MahalanobisDist...")
            #     def MahalanobisDistCalc(x, y):
            #         covariance_xy = np.cov(x,y, rowvar=0)
            #         inv_covariance_xy = np.linalg.inv(covariance_xy)
            #         xy_mean = np.mean(x),np.mean(y)
            #         x_diff = np.array([x_i - xy_mean[0] for x_i in x])
            #         y_diff = np.array([y_i - xy_mean[1] for y_i in y])
            #         diff_xy = np.transpose([x_diff, y_diff])
            #
            #         md = []
            #         for i in range(len(diff_xy)):
            #             md.append(np.sqrt(np.dot(np.dot(np.transpose(diff_xy[i]),inv_covariance_xy),diff_xy[i])))
            #         return md
            #     logger.warn("1")
            #     cluster_ids = controller.supervisor.selected
            #     spike_ids = controller.selector.select_spikes(cluster_ids)
            #     s = controller.supervisor.clustering.spikes_in_clusters(cluster_ids)
            #     data = controller.model._load_features()
            #     data3 = data.data[spike_ids]
            #     data2 = np.reshape(data3,(data3.shape[0],data3.shape[1]*data3.shape[2]))
            #     x = data2
            #     # y = data2
            #     logger.warn("2")
            #     md = MahalanobisDistCalc(x,x)
            #     logger.warn("3")
                



