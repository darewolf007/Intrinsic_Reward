import numpy as np
import sapien

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils

from .panda import Panda


@register_agent()
class PandaWristCam(Panda):
    """Panda arm robot with the real sense camera attached to gripper"""

    uid = "panda_wristcam"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/panda/panda_v3.urdf"

    @property
    def _sensor_configs(self):
        intrinsic = np.array([[905.0100708007812, 0.0, 653.8479614257812], [0.0, 905.0863037109375, 373.97210693359375], [0.0, 0.0, 1.0]])
        intrinsic_128 = np.array([[90.501, 0.0,   65.38],[0.0,    160.9, 66.23],[0.0,    0.0,   1.0]])
        return [
            CameraConfig(
                uid="hand_camera",
                pose=sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),
                width=128,
                height=128,
                intrinsic=intrinsic_128,
                near=0.01,
                far=100,
                mount=self.robot.links_map["camera_link"],
            )
        ]
