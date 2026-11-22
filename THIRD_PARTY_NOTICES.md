This course extraction derives from Mao455/bcrl_visualTactile, commit
60e7b1fdd806bc31fb5e5cb2f49189018ba0d4f3, and Play2Perfect by Kushal Kedia
and collaborators. The upstream MIT license is retained in LICENSE.

The KUKA iiwa14 / SHARPA URDF and meshes, narrow table, L peg, and 0.5 mm hole
are the exact task assets supplied in that upstream repository. Their bytes
and original paths are recorded in evidence/source_extraction.json. This
package preserves upstream attribution and does not include other task assets.

DINOv2 is pinned to facebookresearch/dinov2 revision
7764ea0f912e53c92e82eb78a2a1631e92725fc8. Its Apache-2.0 license, model card
and upstream notices ship with the resource archive in resources/dinov2/.
The ViT-S/14 weight SHA256 is recorded in configs/runtime_resources.json and the
release checksum file. Original source: https://github.com/facebookresearch/dinov2
Original weight: https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth

Isaac Lab is an external SDK, pinned to official revision
090aed18163b2194d5551c7919f7539283677743. The optional source resource retains
its BSD-3-Clause LICENSE and CONTRIBUTORS.md. Source: https://github.com/isaac-sim/IsaacLab
Isaac Sim 5.1 must be installed from NVIDIA's supported Python package index
using scripts/install-sdk.sh, under NVIDIA's applicable license. The course
release does not redistribute Isaac Sim binaries.

Third-party diffusion modules preserve their original headers in
visuotactile/models/vendor/. Python dependencies retain their own licenses.
