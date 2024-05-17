from skellyalign.models.alignment_config import SpatialAlignmentConfig
from skellyalign.run_alignment import run_ransac_spatial_alignment
from skellyalign.plots.scatter_3d import plot_3d_scatter
from mdn_validation_marker_set import markers_to_extract
from pathlib import Path
import numpy as np
from skellymodels.create_model_skeleton import create_mediapipe_skeleton_model, create_qualisys_mdn_nih_skeleton_model


path_to_recording = Path(r"D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_14_53_48_MDN_NIH_Trial3")
path_to_freemocap_output_data = path_to_recording/'output_data'/'mediapipe_body_3d_xyz.npy'
path_to_qualisys_output_data = path_to_recording/'qualisys'/ 'clipped_qualisys_skel_3d.npy'


freemocap_data = np.load(path_to_freemocap_output_data)
freemocap_model = create_mediapipe_skeleton_model()
freemocap_model.integrate_freemocap_3d_data(freemocap_data)


qualisys_data = np.load(path_to_qualisys_output_data)
qualisys_model = create_qualisys_mdn_nih_skeleton_model()
qualisys_model.integrate_freemocap_3d_data(qualisys_data)

alignment_config = SpatialAlignmentConfig(
    freemocap_skeleton=freemocap_model,
    qualisys_skeleton=qualisys_model,
    markers_for_alignment=markers_to_extract,
    frames_to_sample=20,
    max_iterations=50,
    inlier_threshold=40
)

aligned_freemocap_data, transformation_matrix = run_ransac_spatial_alignment(alignment_config)


# freemocap_data_processor = DataProcessor(data=freemocap_model.marker_data_as_numpy, marker_list=freemocap_model.marker_names, markers_for_alignment=markers_to_extract)
# qualisys_data_processor = DataProcessor(data=qualisys_model.marker_data_as_numpy, marker_list=qualisys_model.marker_names, markers_for_alignment=markers_to_extract)

# best_transformation_matrix = get_best_transformation_matrix_ransac(freemocap_data=freemocap_data_processor.extracted_data_3d, qualisys_data=qualisys_data_processor.extracted_data_3d, frames_to_sample = 20, max_iterations=50, inlier_threshold= 40, )

# freemocap_aligned_data = apply_transformation(best_transformation_matrix, freemocap_model.marker_data_as_numpy)

plot_3d_scatter(freemocap_data=aligned_freemocap_data, qualisys_data=qualisys_model.marker_data_as_numpy)
# # from skellyalign.run_alignment import run_ransac_alignment

# aligned_freemocap_data = run_ransac_alignment(sample_recording_config)

f =2 