# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling RVT or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
#
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from multiprocessing import Value

import numpy as np
import torch
from yarr.agents.agent import Agent
from yarr.envs.env import Env
from yarr.utils.transition import ReplayTransition
from yarr.agents.agent import ActResult, VideoSummary
from yarr.utils.gs_processor import GSProcessor

# import zmq
import os
import cv2
import shutil

        
class RolloutGenerator(object):

    def __init__(self, env_device = 'cuda:0', method='base', save_dir='./tracking_results'):
        self._env_device = env_device
        self.save_dir = os.path.join(save_dir)
        self.method = method
        if self.method == 'umvp':

            self.gs_processor = GSProcessor(device=self._env_device)
            print("********Adding mask to observation********")
        
    def _get_type(self, x):
        if x.dtype == np.float64:
            return np.float32
        return x.dtype

    def generator(self, step_signal: Value, env: Env, agent: Agent,
                  episode_length: int, timesteps: int,
                  eval: bool, eval_demo_seed: int = 0,
                  record_enabled: bool = False,
                  replay_ground_truth: bool = False):

        
        # 为每个演示创建单独的目录
        
        demo_save_dir = os.path.join(self.save_dir, str(eval_demo_seed))
        if self.method == 'umvp':
            self.gs_processor.reset(demo_save_dir)
        os.makedirs(demo_save_dir)
        mvt_input_dir = os.path.join(demo_save_dir, "mvt_input")
        mvt_output_dir = os.path.join(demo_save_dir, "mvt_output")
        sim_frames_dir = os.path.join(demo_save_dir, "sim_frames")
        raw_obs_dir = os.path.join(demo_save_dir, "raw_obs")
        test_dir = os.path.join(demo_save_dir, "test")
        os.makedirs(mvt_input_dir)
        os.makedirs(mvt_output_dir)
        os.makedirs(sim_frames_dir)
        os.makedirs(raw_obs_dir)
        os.makedirs(test_dir)
        if self.method == 'umvp':
            umvp_obs_dir = os.path.join(demo_save_dir, "umvp_obs")
            os.makedirs(umvp_obs_dir)
        
        agent.set_dir(demo_save_dir)
            

        if eval:
            obs = env.reset_to_demo(eval_demo_seed)
            print(env._lang_goal)
            
            self.task_description = env._task.get_task_descriptions()[0]
            print(self.task_description)
            # get ground-truth action sequence
            if replay_ground_truth:
                actions = env.get_ground_truth_action(eval_demo_seed)
        else:
            obs = env.reset()

        
        # Masking the first observation
        if self.method == 'umvp':
            rgb_frames = obs['front_rgb'].copy()
            rgb_frames = rgb_frames[None]
            masked_frame = self.gs_processor.process_frame(0, rgb_frames, self.task_description)
            masked_frame = np.array(masked_frame, dtype=np.uint8)
            masked_frame_rgb = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB)
            masked_frame_rgb = masked_frame_rgb.transpose(2,0,1)
            # save_path = os.path.join(demo_save_dir, f"umvp_first.jpg")
            # cv2.imwrite(save_path, cv2.cvtColor(masked_frame_rgb.transpose(1,2,0), cv2.COLOR_RGB2BGR))    
        
        agent.reset()
        obs_history = {k: [np.array(v, dtype=self._get_type(v))] * timesteps for k, v in obs.items()}
        # obs_history.keys() = ['front_rgb', 'front_point_cloud', 'ignore_collisions', 'low_dim_state', 
        # 'front_camera_extrinsics', 'front_camera_intrinsics', 'lang_goal_tokens']
        
        current_step = 0
        for step in range(episode_length):
            agent.set_step(step)
            
            # The image saved here is the observation that is used for the agent to make the action at the current step.
            # prepped_data['front_rgb'][-1].shape = (1, 3, 256, 256)
            prepped_data = {k:torch.tensor(np.array([v]), device=self._env_device) for k, v in obs_history.items()}
            # Convert shape to (256, 256, 3)
            front_rgb_save = prepped_data['front_rgb'][-1].squeeze(0).permute(1, 2, 0).cpu().numpy()
            # Save front_rgb from prepped data
            save_path = os.path.join(raw_obs_dir, f"front_rgb_{step}.jpg")
            cv2.imwrite(save_path, cv2.cvtColor(front_rgb_save, cv2.COLOR_RGB2BGR))
            if self.method == 'umvp':   
                prepped_data['front_rgb'][-1] = torch.tensor(masked_frame_rgb, device=self._env_device)
                masked_front_rgb_save = prepped_data['front_rgb'][-1].squeeze(0).permute(1, 2, 0).cpu().numpy()
                save_path = os.path.join(umvp_obs_dir, f"masked_front_rgb_{step}.jpg")
                cv2.imwrite(save_path, cv2.cvtColor(masked_front_rgb_save, cv2.COLOR_RGB2BGR))

            if not replay_ground_truth:
                act_result = agent.act(step_signal.value, prepped_data,
                                    deterministic=eval)
            else:
                if step >= len(actions):
                    return
                act_result = ActResult(actions[step])

            # Convert to np if not already
            agent_obs_elems = {k: np.array(v) for k, v in
                               act_result.observation_elements.items()}
            extra_replay_elements = {k: np.array(v) for k, v in
                                     act_result.replay_elements.items()}
            transition = env.step(act_result)
                   
            # 保存summaries中的视频帧
            for summary in transition.summaries:
                if isinstance(summary, VideoSummary):
                    video = summary.value  # (T, C, H, W)
            #         print(step, video.shape)

            # Add front_rgb observation to video
            if self.method == 'umvp':
                current_rgb_frame = transition.observation['front_rgb']  # Already in CHW format
                # print(current_rgb_frame.shape)
                # # save this
                # save_path = os.path.join(test_dir, f"front_rgb_{step}.jpg")
                # cv2.imwrite(save_path, cv2.cvtColor(torch.tensor(current_rgb_frame).permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR))
                current_rgb_frame = current_rgb_frame[None]  # Add time dimension (1,C,H,W)
                
                sample_rate = 10
                sampled_video = video[current_step::sample_rate]
                video_all = np.concatenate([sampled_video, current_rgb_frame], axis=0)
                
                
                masked_frame = self.gs_processor.process_frame(step + 1, video_all, self.task_description)
                masked_frame_bgr = np.array(masked_frame, dtype=np.uint8)
                masked_frame_rgb = cv2.cvtColor(masked_frame_bgr, cv2.COLOR_BGR2RGB)
                masked_frame_rgb = masked_frame_rgb.transpose(2,0,1)
                # save_path = os.path.join(demo_save_dir, f"annotated_frame_{step + 1:05d}.jpg")
                # cv2.imwrite(save_path, cv2.cvtColor(anotated_frame_rgb.transpose(1,2,0), cv2.COLOR_RGB2BGR))    
        
            # save_path = os.path.join(demo_save_dir, f"annotated_frame_{step + 1:05d}.jpg")
            # cv2.imwrite(save_path, np.array(result['annotated_frame'], dtype=np.uint8))
            
            current_step = video.shape[0]
            
            obs_tp1 = dict(transition.observation)
            timeout = False
            if step == episode_length - 1:
                # If last transition, and not terminal, then we timed out
                timeout = not transition.terminal
                if timeout:
                    transition.terminal = True
                    if "needs_reset" in transition.info:
                        transition.info["needs_reset"] = True

            if transition.terminal or timeout:
                for t, frame in enumerate(video):
                    frame = frame.transpose(1, 2, 0)  # CHW -> HWC
                    save_path = os.path.join(
                        sim_frames_dir,
                        f"sim_frame_{step:05d}_{t:05d}_bgr.jpg"
                    )
                    cv2.imwrite(save_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                            
            obs_and_replay_elems = {}
            obs_and_replay_elems.update(obs)
            obs_and_replay_elems.update(agent_obs_elems)
            obs_and_replay_elems.update(extra_replay_elements)

            for k in obs_history.keys():
                obs_history[k].append(transition.observation[k])
                obs_history[k].pop(0)

            transition.info["active_task_id"] = env.active_task_id

            replay_transition = ReplayTransition(
                obs_and_replay_elems, act_result.action, transition.reward,
                transition.terminal, timeout, summaries=transition.summaries,
                info=transition.info)

            if transition.terminal or timeout:
                # If the agent gives us observations then we need to call act
                # one last time (i.e. acting in the terminal state).
                if len(act_result.observation_elements) > 0:
                    prepped_data = {k: torch.tensor([v], device=self._env_device) for k, v in obs_history.items()}
                    act_result = agent.act(step_signal.value, prepped_data,
                                           deterministic=eval)
                    agent_obs_elems_tp1 = {k: np.array(v) for k, v in
                                           act_result.observation_elements.items()}
                    obs_tp1.update(agent_obs_elems_tp1)
                replay_transition.final_observation = obs_tp1

            if record_enabled and transition.terminal or timeout or step == episode_length - 1:
                env.env._action_mode.arm_action_mode.record_end(env.env._scene,
                                                                steps=60, step_scene=True)

            obs = dict(transition.observation)

            yield replay_transition

            if transition.info.get("needs_reset", transition.terminal):
                return
