论文地址 https://ieeexplore.ieee.org/document/8737778

Abstract—In this paper, we focus on a cellular network aided
an unmanned aerial vehicle (UAV) that serves as an aerial base
station for multiple ground users. The UAV’s trajectory design
is investigated to maximize the expected uplink sum rate with
inaccessibility to user-side information, such as locations and
transmit power as well as channel parameters. The problem
is formulated as a Markov decision process (MDP) and solved
with model-free reinforcement learning. Due to the continuous
and deterministic action space, the deterministic policy gradient
(DPG) algorithm is applied for the reinforcement learning model.
Experiment results show that due to the great generalizability of
the reinforcement learning model, the UAV is able to intelligently
track the ground users with the learned trajectory despite being
unaware of the user-side information and channel parameters,
even when the ground users are mobile. The performance of
the learned trajectory is fairly close to that of the optimized
trajectory derived through conventional optimization problem
solving with such information explicitly known. Moreover, we
also show that the DPG algorithm converges efficiently with
acceptable training time.
Index Terms—UAV-aided communications, trajectory design,
reinforcement learning.
