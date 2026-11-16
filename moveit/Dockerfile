# MoveIt2 sidecar for the G1-D + Dex1 stack.
#
# Built with the host-network buildx builder (clash proxy) - see
# run.sh's ut_build; plain `docker build` cannot pull ros:humble on
# this network (docker hub unreachable directly).
#
#   ./run.sh moveit-build
#
# The container is stateless planning only: move_group runs WITHOUT
# controllers - plans are fetched via moveit_py and shipped to the
# simulator over ZMQ (tools/moveit_bridge.py executes them through
# the sim's own DDS rt/lowcmd path, identical to keyboard teleop).

# NOTE ros:jazzy, NOT humble: moveit_py (the python planning API)
# only exists from Jazzy on; humble would force moveit_commander
FROM ros:jazzy

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-jazzy-moveit-py \
      ros-jazzy-moveit-planners-ompl \
      ros-jazzy-moveit-msgs \
      ros-jazzy-joint-state-publisher \
      ros-jazzy-ros2cli \
      python3-numpy python3-zmq python3-yaml \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /moveit
COPY config /moveit/config
COPY plan_pick.py /moveit/plan_pick.py
COPY launch_g1d.py /moveit/launch_g1d.py

# robot_state_publisher + move_group, then the ZMQ planning service
CMD ["bash", "-c", "python3 /moveit/launch_g1d.py & sleep 8 && python3 /moveit/plan_pick.py --serve"]
