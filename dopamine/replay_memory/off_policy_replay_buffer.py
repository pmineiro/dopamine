# coding=utf-8

"""The standard DQN replay memory + logged probabilities + return vectors/tensors so we can do cool stuff with them

This implementation is an out-of-graph replay memory + in-graph wrapper
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gzip
import math
import os
import pickle

from absl import logging
import numpy as np
import tensorflow as tf

import gin.tf

from dopamine.replay_memory import circular_replay_buffer
from dopamine.replay_memory.circular_replay_buffer import ReplayElement, STORE_FILENAME_PREFIX, CHECKPOINT_DURATION, invalid_range


@gin.configurable
class OutOfGraphOffPolicyReplayBuffer(circular_replay_buffer.OutOfGraphReplayBuffer):
  """An out-of-graph Replay Buffer with logged probabilities and support for off-policy stuff.

  See circular_replay_buffer.py for details.
  """

  def __init__(self,
               observation_shape,
               stack_size,
               replay_capacity,
               batch_size,
               update_horizon=1,
               gamma=0.99,
               max_sample_attempts=1000,
               extra_storage_types=None,
               observation_dtype=np.uint8,
               terminal_dtype=np.uint8,
               action_shape=(),
               action_dtype=np.int32,
               reward_shape=(),
               reward_dtype=np.float32):
    """Initializes OutOfGraphOffPolicyReplayBuffer.

    Args:
      observation_shape: tuple of ints.
      stack_size: int, number of frames to use in state stack.
      replay_capacity: int, number of transitions to keep in memory.
      batch_size: int.
      update_horizon: int, length of update ('n' in n-step update).
      gamma: int, the discount factor.
      max_sample_attempts: int, the maximum number of attempts allowed to
        get a sample.
      extra_storage_types: list of ReplayElements defining the type of the extra
        contents that will be stored and returned by sample_transition_batch.
      observation_dtype: np.dtype, type of the observations. Defaults to
        np.uint8 for Atari 2600.
      terminal_dtype: np.dtype, type of the terminals. Defaults to np.uint8 for
        Atari 2600.
      action_shape: tuple of ints, the shape for the action vector. Empty tuple
        means the action is a scalar.
      action_dtype: np.dtype, type of elements in the action.
      reward_shape: tuple of ints, the shape of the reward vector. Empty tuple
        means the reward is a scalar.
      reward_dtype: np.dtype, type of elements in the reward.
    """

    super(OutOfGraphOffPolicyReplayBuffer, self).__init__(
        observation_shape=observation_shape,
        stack_size=stack_size,
        replay_capacity=replay_capacity,
        batch_size=batch_size,
        update_horizon=update_horizon,
        gamma=gamma,
        max_sample_attempts=max_sample_attempts,
        extra_storage_types=extra_storage_types,
        observation_dtype=observation_dtype,
        terminal_dtype=terminal_dtype,
        action_shape=action_shape,
        action_dtype=action_dtype,
        reward_shape=reward_shape,
        reward_dtype=reward_dtype)

    # wmax has not been checkpointed.
    # If we try to keep track of it here, batch_rl complains it cannot find a checkpoint
    # self.wmax = 2


  def get_add_args_signature(self):
    """The signature of the add function.

    The signature is the same as the one for OutOfGraphReplayBuffer, with an
    added probability.

    Returns:
      list of ReplayElements defining the type of the argument signature needed
        by the add function.
    """
    parent_add_signature = super(OutOfGraphOffPolicyReplayBuffer,
                                 self).get_add_args_signature()
    add_signature = parent_add_signature + [
        ReplayElement('prob', (), np.float32)
    ]
    return add_signature


  def _add(self, *args):
    self._check_args_length(*args)

    transition = {}
    for i, element in enumerate(self.get_add_args_signature()):
      # commented out because wmax was not checkpointed
      # if element.name == 'prob':
      #   prob = args[i]
      transition[element.name] = args[i]

    # commented out because wmax was not checkpointed
    #self.wmax = max(self.wmax, 1.0/prob)
    super(OutOfGraphOffPolicyReplayBuffer, self)._add_transition(transition)

  def get_transition_elements(self, batch_size=None):
    """Returns a 'type signature' for sample_transition_batch.

    Args:
      batch_size: int, number of transitions returned. If None, the default
        batch_size will be used.
    Returns:
      signature: A namedtuple describing the method's return type signature.
    """
    parent_transition_type = (
        super(OutOfGraphOffPolicyReplayBuffer,
              self).get_transition_elements(batch_size))
    update_horizon = self._update_horizon
    batch_size = self._batch_size if batch_size is None else batch_size

    trajectory_type = [
        ReplayElement('traj_state', (batch_size, update_horizon) + self._state_shape, self._observation_dtype),
        ReplayElement('traj_action', (batch_size, update_horizon), np.int32),
        ReplayElement('traj_reward', (batch_size, update_horizon), np.float32),
        ReplayElement('traj_prob', (batch_size, update_horizon), np.float32),
        ReplayElement('traj_discount', (batch_size, update_horizon), np.float32),
    ]
    return parent_transition_type + trajectory_type

  def sample_transition_batch(self, batch_size=None, indices=None):
    """Returns a batch of transitions (including any extra contents).

    If get_transition_elements has been overridden and defines elements not
    stored in self._store, an empty array will be returned and it will be
    left to the child class to fill it. For example, for the child class
    OutOfGraphPrioritizedReplayBuffer, the contents of the
    sampling_probabilities are stored separately in a sum tree.

    When the transition is terminal next_state_batch has undefined contents.

    NOTE: This transition contains the indices of the sampled elements. These
    are only valid during the call to sample_transition_batch, i.e. they may
    be used by subclasses of this replay buffer but may point to different data
    as soon as sampling is done.

    Args:
      batch_size: int, number of transitions returned. If None, the default
        batch_size will be used.
      indices: None or list of ints, the indices of every transition in the
        batch. If None, sample the indices uniformly.

    Returns:
      transition_batch: tuple of np.arrays with the shape and type as in
        get_transition_elements().

    Raises:
      ValueError: If an element to be sampled is missing from the replay buffer.
    """
    if batch_size is None:
      batch_size = self._batch_size
    if indices is None:
      indices = self.sample_index_batch(batch_size)
    assert len(indices) == batch_size

    transition_elements = self.get_transition_elements(batch_size)
    batch_arrays = self._create_batch_arrays(batch_size)
    for batch_element, state_index in enumerate(indices):
      trajectory_indices = [(state_index + j) % self._replay_capacity
                            for j in range(self._update_horizon)]
      trajectory_terminals = self._store['terminal'][trajectory_indices]
      is_terminal_transition = trajectory_terminals.any()
      if not is_terminal_transition:
        trajectory_length = self._update_horizon
      else:
        # np.argmax of a bool array returns the index of the first True.
        trajectory_length = np.argmax(trajectory_terminals.astype(np.bool),
                                      0) + 1
      next_state_index = state_index + trajectory_length
      trajectory_discount_vector = (
          self._cumulative_discount_vector[:trajectory_length])
      trajectory_states = np.array([self.get_observation_stack(i) for i in trajectory_indices[:trajectory_length]])
      trajectory_actions = self.get_range(self._store['action'], state_index,
                                          next_state_index)
      trajectory_rewards = self.get_range(self._store['reward'], state_index,
                                          next_state_index)
      trajectory_probs = self.get_range(self._store['prob'], state_index,
                                          next_state_index)

      # Fill the contents of each array in the sampled batch.
      assert len(transition_elements) == len(batch_arrays)
      for element_array, element in zip(batch_arrays, transition_elements):
        if element.name == 'traj_state':
          element_array[batch_element, :trajectory_length] = trajectory_states
          element_array[batch_element, trajectory_length:] = 0
        elif element.name == 'traj_action':
          element_array[batch_element, :trajectory_length] = trajectory_actions
          element_array[batch_element, trajectory_length:] = 0
        elif element.name == 'traj_reward':
          element_array[batch_element, :trajectory_length] = trajectory_rewards
          element_array[batch_element, trajectory_length:] = 0
        elif element.name == 'traj_prob':
          element_array[batch_element,:trajectory_length] = trajectory_probs
          element_array[batch_element,trajectory_length:] = 1
        elif element.name == 'traj_discount':
          element_array[batch_element,:trajectory_length] = trajectory_discount_vector
          element_array[batch_element,trajectory_length:] = 0
        elif element.name == 'state':
          element_array[batch_element] = self.get_observation_stack(state_index)
        elif element.name == 'reward':
          # compute the discounted sum of rewards in the trajectory.
          element_array[batch_element] = np.sum(
                trajectory_discount_vector * trajectory_rewards, axis=0)
        elif element.name == 'next_state':
          element_array[batch_element] = self.get_observation_stack(
              (next_state_index) % self._replay_capacity)
        elif element.name in ('next_action', 'next_reward'):
          element_array[batch_element] = (
              self._store[element.name.lstrip('next_')][(next_state_index) %
                                                        self._replay_capacity])
        elif element.name == 'terminal':
          element_array[batch_element] = is_terminal_transition
        elif element.name == 'indices':
          element_array[batch_element] = state_index
        elif element.name in self._store.keys():
          element_array[batch_element] = (
              self._store[element.name][state_index])
        # We assume the other elements are filled in by the subclass.

    return batch_arrays


@gin.configurable(blacklist=['observation_shape', 'stack_size',
                             'update_horizon', 'gamma'])
class WrappedOffPolicyReplayBuffer(
    circular_replay_buffer.WrappedReplayBuffer):
  """Wrapper of OutOfGraphOffPolicyReplayBuffer with in-graph sampling.

  Usage:

    * To add a transition:  Call the add function.

    * To sample a batch:  Query any of the tensors in the transition dictionary.
                          Every sess.run that requires any of these tensors will
                          sample a new transition.
  """

  def __init__(self,
               observation_shape,
               stack_size,
               use_staging=False,
               replay_capacity=1000000,
               batch_size=32,
               update_horizon=1,
               gamma=0.99,
               wrapped_memory=None,
               max_sample_attempts=1000,
               extra_storage_types=None,
               observation_dtype=np.uint8,
               terminal_dtype=np.uint8,
               action_shape=(),
               action_dtype=np.int32,
               reward_shape=(),
               reward_dtype=np.float32):
    """Initializes WrappedOffPolicyReplayBuffer.

    Args:
      observation_shape: tuple of ints.
      stack_size: int, number of frames to use in state stack.
      use_staging: bool, when True it would use a staging area to prefetch
        the next sampling batch.
      replay_capacity: int, number of transitions to keep in memory.
      batch_size: int.
      update_horizon: int, length of update ('n' in n-step update).
      gamma: int, the discount factor.
      wrapped_memory: The 'inner' memory data structure. If None, use the
        default prioritized replay.
      max_sample_attempts: int, the maximum number of attempts allowed to
        get a sample.
      extra_storage_types: list of ReplayElements defining the type of the extra
        contents that will be stored and returned by sample_transition_batch.
      observation_dtype: np.dtype, type of the observations. Defaults to
        np.uint8 for Atari 2600.
      terminal_dtype: np.dtype, type of the terminals. Defaults to np.uint8 for
        Atari 2600.
      action_shape: tuple of ints, the shape for the action vector. Empty tuple
        means the action is a scalar.
      action_dtype: np.dtype, type of elements in the action.
      reward_shape: tuple of ints, the shape of the reward vector. Empty tuple
        means the reward is a scalar.
      reward_dtype: np.dtype, type of elements in the reward.

    Raises:
      ValueError: If update_horizon is not positive.
      ValueError: If discount factor is not in [0, 1].
    """
    if wrapped_memory is None:
      wrapped_memory = OutOfGraphOffPolicyReplayBuffer(
          observation_shape, stack_size, replay_capacity, batch_size,
          update_horizon, gamma, max_sample_attempts,
          extra_storage_types=extra_storage_types,
          observation_dtype=observation_dtype)

    super(WrappedOffPolicyReplayBuffer, self).__init__(
        observation_shape,
        stack_size,
        use_staging,
        replay_capacity,
        batch_size,
        update_horizon,
        gamma,
        wrapped_memory=wrapped_memory,
        extra_storage_types=extra_storage_types,
        observation_dtype=observation_dtype,
        terminal_dtype=terminal_dtype,
        action_shape=action_shape,
        action_dtype=action_dtype,
        reward_shape=reward_shape,
        reward_dtype=reward_dtype)