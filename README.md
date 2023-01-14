# Reinforcement Learning

- This repository contains Python implementation of basic RL algorithms, including
  - Value iteration
  - Q-Learning
- Environments are provided by [Gymnasium](https://gymnasium.farama.org/)
  - Gymnasium is a maintained fork of OpenAI’s [Gym](https://www.gymlibrary.dev/) library
  - Recent official message
    - All development of Gym has been moved to Gymnasium, a new package in the Farama Foundation that's maintained by the same team of developers who have maintained Gym for the past 18 months. 
    - If you're already using the latest release of Gym (v0.26.2), then you can switch to v0.27.0 of Gymnasium by simply replacing ```import gym``` with ```import gymnasium as gym``` with no additional steps. 
    - Gym will not be receiving any future updates or bug fixes, and no further changes will be made to the core API in Gymnasium.

## Status

- Currently, I'm actively working on modifying this project to make it more readable and modularized

## Notes

- The main purpose of creating this repository is to have fun playing around with RL
- More specifically, I wanted to
  1. run experiment new RL algorithms on simple game-like environments
  2. refresh my memory of the basic RL algorithms
  3. hopefully I can figure out some interesting variations of the existing RL algorithms and contribute to this field
  4. share my implementation with the community to get feedback
  5. maybe help people who are also interested in RL and just started to explore this intriguing field
- Researchers from DeepMind published an article about the relationship between neuroscience and AI in early 2020: **Dopamine and temporal difference learning: A fruitful relationship between neuroscience and AI**
  - [Blogpost](https://www.deepmind.com/blog/dopamine-and-temporal-difference-learning-a-fruitful-relationship-between-neuroscience-and-ai)
  - [Paper](https://www.nature.com/articles/s41586-019-1924-6.epdf?shared_access_token=3Bcr-ZWATXBxuAME25rI7tRgN0jAjWel9jnR3ZoTv0OgnvLoVhK46-VND2gsGkjz89fNskUJsDZNDD1PQ0vP4GRakb69mL9k_JklOh9EofWr26Xzkg5xKBwi24XiemaDtez3u5DhPPuVfqxLmAcCIw%3D%3D)
  - [Podcast](https://www.deepmind.com/blog/the-podcast-episode-1-ai-and-neuroscience-the-virtuous-circle)
- I learn RL mainly from 
  - the book *Reinforcement Learning, Second Edition* by Rich Sutton and Andrew Barto
  - courses in the *Reinforcement Learning Specialization* offered by University of Alberta
  - research papers on Google Scholar, ACM DL, [Google AI Blog](https://ai.googleblog.com/), and [Workshops](https://neurips.cc/virtual/2022/workshop/49957)
- I always take [notes](https://github.com/ypchen520/alishan-range/tree/main/RL/RL-Coursera), and share them in another [repository](https://github.com/ypchen520/alishan-range) to get feedback and to help people who are also interested in various topics that I touched on