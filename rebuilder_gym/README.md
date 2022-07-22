# Rebuild Gym Environment
The game is very simple: the agent's goal is to get the ball to land on the ground of its opponent's side, causing its opponent to lose a life. Each agent starts off with five lives. The episode ends when either agent loses all five lives, or after 3000 timesteps has passed. An agent receives a reward of +1 when its opponent loses or -1 when it loses a life.

### Notable features

- Only dependencies are gym and numpy.

- In the normal single agent setting, the agent plays against a tiny 120-parameter [neural network](https://otoro.net/slimevolley/) baseline agent from 2015. This opponent can easily be replaced by another policy to enable a multi-agent or self-play environment.

- Runs at around 12.5K timesteps per second on 2015 MacBook (core i7) for state-space observations, resulting in faster iteration in experiments.

- A [tutorial](TRAINING.md) demonstrating several different training methods (e.g. single agent, self-play, evolution) that require only a single CPU machine in most cases. Potentially useful for educational purposes.

- A pixel observation mode is available. Observations are directly rendered to numpy arrays and runs on headless cloud machines. The pixel version of the environment mimics gym environments based on the Atari Learning Environment and has been tested on several Atari gym wrappers and RL models tuned for Atari.

- The opponent's observation is made available in the optional `info` object returned by `env.step()` for both state and pixel settings. The observations are constructed as if the agent is always playing on the right court, even if it is playing on the left court, so an agent trained to play on one side can play on the other side without adjustment.

## Basic Usage

```
python test_state.py
```

<p align="left">
  <img width="50%" src="https://otoro.net/img/slimegym/state.gif"></img>
  <!--<br/><i>State-space observation mode.</i>-->
</p>

You can control the agent on the right using the arrow keys, or the agent on the left using (A, W, D).

Similarly, `test_pixel.py` allows you to play in the pixelated environment, and `test_atari.py` lets you play the game by observing the preprocessed stacked frames (84px x 84px x 4 frames) typically done for Atari RL agents:

<p align="left">
  <img width="50%" src="https://media.giphy.com/media/W3NItV6PINmbgUFKPf/giphy.gif"></img>
  <br/><i>Atari gym wrappers combine 4 frames as one observation.</i>
</p>

## Environments

There are two types of environments: state-space observation or pixel observations:

|Environment Id|Observation Space|Action Space
|---|---|---|
|SlimeVolley-v0|Box(12)|MultiBinary(3)
|SlimeVolleyPixel-v0|Box(84, 168, 3)|MultiBinary(3)
|SlimeVolleyNoFrameskip-v0|Box(84, 168, 3)|Discrete(6)

`SlimeVolleyNoFrameskip-v0` identical to `SlimeVolleyPixel-v0` except that the action space is now a one-hot vector typically used in Atari RL agents.

In state-space observation, the 12-dim vector corresponds to the following states:

<img src="https://render.githubusercontent.com/render/math?math=\left(x_{agent}, y_{agent}, \dot{x}_{agent}, \dot{y}_{agent}, x_{ball}, y_{ball}, \dot{x}_{ball}, \dot{y}_{ball}, x_{opponent}, y_{opponent}, \dot{x}_{opponent}, \dot{y}_{opponent}\right)"></img>

The origin point (0, 0) is located at the bottom of the fence.

Both state and pixel observations are presented assuming the agent is playing on the right side of the screen.

### Using Multi-Agent Environment

It is straight forward to modify the gym loop to enable multi-agent or self-play. Here is a basic gym loop:

```python
import gym
import slimevolleygym

env = gym.make("SlimeVolley-v0")

obs = env.reset()
done = False
total_reward = 0

while not done:
  action = my_policy(obs)
  obs, reward, done, info = env.step(action)
  total_reward += reward
  env.render()

print("score:", total_reward)
```

The `info` object contains extra information including the observation for the opponent:

```
info = {
  'ale.lives': agent's lives left,
  'ale.otherLives': opponent's lives left,
  'otherObs': opponent's observations,
  'state': agent's state (same as obs in state mode),
  'otherState': opponent's state (same as otherObs in state mode),
}
```

This modification allows you to evaluate `policy1` against `policy2`

```python
obs1 = env.reset()
obs2 = obs1 # both sides always see the same initial observation.

done = False
total_reward = 0

while not done:

  action1 = policy1(obs1)
  action2 = policy2(obs2)

  obs1, reward, done, info = env.step(action1, action2) # extra argument
  obs2 = info['otherObs']

  total_reward += reward
  env.render()

print("policy1's score:", total_reward)
print("policy2's score:", -total_reward)
```

Note that in both state and pixel modes, `otherObs` is given as if the agent is playing on the right side of the screen, so one can swap an agent to play either side without modifying the agent.

<p align="left">
  <img width="50%" src="https://media.giphy.com/media/IeA1Nv2WZSOoZJrh6Z/giphy.gif"></img>
  <br/><i>Opponent's observation is rendered in the smaller window.</i>
</p>

One can consider replacing `policy2` with earlier versions of your agent (self-play) and wrapping the multi-agent environment as if it were a single-agent environment so that it can use standard RL algorithms. There are several examples of these techniques described in more detail in the [TRAINING.md](TRAINING.md) tutorial.

## Evaluating against other agents

Several pre-trained agents (`ppo`, `cma`, `ga`, `baseline`) are discussed in the [TRAINING.md](TRAINING.md) tutorial.

You can run them against each other using the following command:

```
python eval_agents.py --left ppo --right cma --render
```

<p align="left">
  <!--<img width="50%" src="https://media.giphy.com/media/VGPfocuIS7YYh6kyMv/giphy.gif"></img>-->
  <img width="50%" src="https://media.giphy.com/media/WsMaF3xeATeiCv7dBq/giphy.gif"></img>
  <br/><i>Evaluating PPO agent (left) against CMA-ES (right).</i>
</p>

It should be relatively straightforward to modify `eval_agents.py` to include your custom agent.

## Leaderboard

Below are scores achieved by various algorithms and links to their implementations. Feel free to add yours here:

### SlimeVolley-v0

|Method|Average Score|Episodes|Other Info
|---|---|---|---|
|Maximum Possible Score|5.0|  | 
|PPO | 1.377 ± 1.133 | 1000 | [link](https://github.com/hardmaru/slimevolleygym/blob/master/TRAINING.md)
|CMA-ES | 1.148 ± 1.071 | 1000 | [link](https://github.com/hardmaru/slimevolleygym/blob/master/TRAINING.md)
|GA (Self-Play) | 0.353 ± 0.728 | 1000 | [link](https://github.com/hardmaru/slimevolleygym/blob/master/TRAINING.md)
|CMA-ES (Self-Play) | -0.071 ± 0.827 | 1000 | [link](https://github.com/hardmaru/slimevolleygym/blob/master/TRAINING.md)
|PPO (Self-Play) | -0.371 ± 1.085 | 1000 | [link](https://github.com/hardmaru/slimevolleygym/blob/master/TRAINING.md)
|Random Policy | -4.866 ± 0.372 | 1000 | 
|[Add Method](https://github.com/hardmaru/slimevolleygym/edit/master/README.md) |  |  |  

### SlimeVolley-v0 (Sample Efficiency)

For sample efficiency, we can measure how many timesteps it took to train an agent that can achieve a positive average score (over 1000 episodes) against the built-in baseline policy:

|Method| Timesteps (Best) | Timesteps (Median)| Trials | Other Info
|---|---|---|---|---|
|PPO | 1.274M | 2.998M | 17 | [link](https://github.com/hardmaru/slimevolleygym/blob/master/TRAINING.md)
|Data-efficient Rainbow | 0.750M | 0.751M | 3 | [link](https://github.com/pfnet/pfrl/blob/master/examples/slimevolley/README.md)
|[Add Method](https://github.com/hardmaru/slimevolleygym/edit/master/README.md) |  |  |  | 

### SlimeVolley-v0 (Against Other Agents)

Table of average scores achieved versus agents other than the default baseline policy ([1000 episodes](https://github.com/hardmaru/slimevolleygym/blob/master/eval_agents.py)):

|Method|Baseline|PPO|CMA-ES|GA (Self-Play)| Other Info
|---|---|---|---|---|---|
|PPO |  1.377 ± 1.133 | — |  0.133 ± 0.414 | -3.128 ± 1.509 | [link](https://github.com/hardmaru/slimevolleygym/blob/master/TRAINING.md)
|CMA-ES | 1.148 ± 1.071 | -0.133 ± 0.414 | — | -0.301 ± 0.618 | [link](https://github.com/hardmaru/slimevolleygym/blob/master/TRAINING.md)
|GA (Self-Play) | 0.353 ± 0.728  | 3.128 ± 1.509 | 0.301 ± 0.618 | — | [link](https://github.com/hardmaru/slimevolleygym/blob/master/TRAINING.md)
|CMA-ES (Self-Play) | -0.071 ± 0.827  |  -0.749 ± 0.846 |  -0.351 ± 0.651 |  -4.923 ± 0.342 | [link](https://github.com/hardmaru/slimevolleygym/blob/master/TRAINING.md)
|PPO (Self-Play) | -0.371 ± 1.085  | 0.119 ± 1.46 |  -2.304 ± 1.392 |  -0.42 ± 0.717 | [link](https://github.com/hardmaru/slimevolleygym/blob/master/TRAINING.md)
|[Add Method](https://github.com/hardmaru/slimevolleygym/edit/master/README.md) |  |  |

It is interesting to note that while GA (Self-Play) did not perform as well against the baseline policy compared to PPO and CMA-ES, it is a superior policy if evaluated against these methods that trained directly against the baseline policy.

### SlimeVolleyPixel-v0

Results for pixel observation version of the environment (`SlimeVolleyPixel-v0` or `SlimeVolleyNoFrameskip-v0`):

|Pixel Observation|Average Score|Episodes|Other Info
|---|---|---|---|
|Maximum Possible Score|5.0| | |
|PPO | 0.435 ± 0.961 | 1000 | [link](https://github.com/hardmaru/slimevolleygym/blob/master/TRAINING.md)
|Rainbow | 0.037 ± 0.994 | 1000 | [link](https://github.com/hardmaru/RainbowSlimeVolley)
|A2C | -0.079 ± 1.091 | 1000 | [link](https://github.com/hardmaru/rlzoo)
|ACKTR | -1.183 ± 1.480 | 1000 | [link](https://github.com/hardmaru/rlzoo)
|ACER | -1.789 ± 1.632 | 1000 | [link](https://github.com/hardmaru/rlzoo)
|DQN | -4.091 ± 1.242 | 1000 | [link](https://github.com/hardmaru/rlzoo)
|Random Policy | -4.866 ± 0.372 | 1000 | 
|[Add Method](https://github.com/hardmaru/slimevolleygym/edit/master/README.md) |  | (>= 1000) | 

## Publications

If you have publications, articles, projects, blog posts that use this environment, feel free to add a link here via a [PR](https://github.com/hardmaru/slimevolleygym/edit/master/README.md).

## Citation

<!--<p align="left">
  <img width="100%" src="https://media.giphy.com/media/WsMaF3xeATeiCv7dBq/giphy.gif"></img></img>
</p>-->

Please use this BibTeX to cite this repository in your publications:

```
@misc{slimevolleygym,
  author = {David Ha},
  title = {Slime Volleyball Gym Environment},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/hardmaru/slimevolleygym}},
}
```


# A Tutorial on Training Self-Play Agents

Here, we provide training examples of 3 self-play methods: genetic algorithm (GA), PPO, and cooperative CMA-ES. We show that self-play can produce agents that can defeat the baseline policy without the need to train against it. Before going into self-play, we will first go through examples that use standard RL methods such as PPO (using [stable-baselines v2.10](https://github.com/hill-a/stable-baselines) library) and evolution strategies (CMA-ES) to train an agent in the standard single-agent environment, where the agent learns by playing against the “expert” baseline policy from [2015](https://otoro.net/slimevolley/).

## SlimeVolley-v0: State Observation Environment

<p align="left">
  <img width="100%" src="https://otoro.net/img/slimegym/state.gif"></img>
</p>

We will first train agents to play Slime Volleyball using state observations (SlimeVolley-v0) and discuss various methods for training agents via self-play. First, we would like to measure the performance of agents trained to play directly in the single-agent environment against the built-in opponent that is controlled by the baseline policy.

## PPO and CMA-ES Example: Train directly against baseline policy

In this first [example](https://github.com/hardmaru/slimevolleygym/blob/master/training_scripts/train_ppo.py), we run stable-baseline's PPO implementation to train an agent to play `SlimeVolley-v0`. In this environment, the agent will play against the built-in baseline policy.

To get a sense of the sample efficiency of the standard [PPO algorithm](https://arxiv.org/abs/1707.06347) for this task, below are results from running a single-thread PPO trainer (see [code](https://github.com/hardmaru/slimevolleygym/blob/master/training_scripts/train_ppo.py)) 17 times with different initial random seeds. The hyperparameters used are roughly the same as the ones in the [stable-baselines](https://github.com/hill-a/stable-baselines) examples chosen for mujoco environments. Training stops when the policy evaluated 1000 times achieves a mean score above zero versus the baseline policy inside the environment.

![ppo_training](figure/ppo_results.svg)

Out of 17 trials with different initial random seeds, the best one solved the task in 1.274M timesteps, and the median number of timesteps is 2.998M. On a single CPU machine, the wall clock speed to train 1M steps is roughly 1 hour, so we can expect to see the agent learning a reasonable policy after a few hours of training. It is interesting to note that some trials took PPO a long time to learn a reasonable strategy, and it could be due to the fact that we are training a randomly initialized network that knows nothing about Slime Volleyball, against an expert player right at the beginning. It's like an infant learning to play volleyball against an Olympic gold medalist. Here, our agent will likely receive the lowest possible score all the time regardless of any small improvement, making it difficult to learn from constant failure. That PPO still manages to eventually find a good policy is a testament of how good it is. This is an important point that we will revisit.

In addition to sample efficiency, we want to know what the best possible performance we can get out of PPO. We ran multi-processor PPO (see [code](https://github.com/hardmaru/slimevolleygym/blob/master/training_scripts/train_ppo_mpi.py)) on a 96-core CPU machine for a while and achieved an average score of 1.377 ± 1.133 over 1000 trials. The highest possible score is 5.0.

<p align="left">
  <img width="50%" src="figure/mpi_ppo_results.svg"></img><img width="50%" src="figure/cmaes_results.svg"></img>
  <br/><i>Training multi-processor version of PPO and CMA-ES. Optimized for wall clock over sample efficiency.</i>
</p>

We also try a standard evolution strategies algorithm, CMA-ES (explained in a [blog post](https://blog.otoro.net/2017/10/29/visual-evolution-strategies/)), to optimize the weight parameters of a neural network agent for this task. An earlier project, [estool](https://github.com/hardmaru/estool) is used to interface the gym environment with CMA-ES. After training for a few thousand generations, the performance achieved is comparable to PPO.

While both PPO and CMA-ES methods trained an agent to play Slime Volleyball against an expert baseline policy, it is of no surprise that, given enough training, they can eventually defeat the baseline policy consistently. We want to see if methods trained with self-play, *without* access to an expert baseline policy, are good enough to consistently beat the baseline policy. Afterall, the baseline policy was also originally trained using [self-play](https://blog.otoro.net/2015/03/28/neural-slime-volleyball/) in 2015. We can also investigate whether the PPO and CMA-ES methods trained against the baseline policy overfit to that particular agent.

# Self-Play Methods

We have shown that standard RL algorithms can defeat the baseline policy in SlimeVolley-v0 by simply training agents to play from scratch directly against the built-in opponent. But what if we didn't have an expert opponent to begin with to learn from? With self-play, we train agents to play against a version of itself (either a past version for the case of PPO, or a sibling in the case of a genetic algorithm (GA)), so they can become incrementally better players over time. We also want to measure the performance of agents trained using self-play against agents trained against the expert.

## Self-Play via Genetic Algorithm

While self-play has gained popularity in Deep RL, it has actually been around for decades in genetic algorithms (see references below) in the evolutionary computing literature. It is also really easy to implement–our [example](https://github.com/hardmaru/slimevolleygym/blob/master/training_scripts/train_ga_selfplay.py) consists of a dozen or so lines of code that implements it.

For demonstration purposes, we are going to use the simplest version of tournament selection GA, without any bells and whistles. It is even simpler than the genetic algorithm in [2015](https://blog.otoro.net/2015/03/28/neural-slime-volleyball/) that trained the baseline policy.

Tournament Selection by Genetic Algorithm:
```
Create a population of agents with random initial parameters.
Play a total of N games. For each game:
  Randomly choose two agents in the population and have them play against each other.
  If the game is tied, add a bit of noise to the second agent's parameters.
  Otherwise, replace the loser with a clone of the winner, and add a bit of noise to the clone.
```

In Python pseudocode:
```python
# param_count is the number of weight parameters in the neural net agent
population = np.random.normal(size=(population_size, param_count))

epsilon = 0.1 # small amount of gaussian noise to be added to weights

for tournament in range(total_tournaments):

  # randomly choose two different agents in the population
  m, n = np.random.choice(population_size, 2, replace=False)

  policy1.set_model_params(population[m])
  policy2.set_model_params(population[n])

  # tournament between the mth and nth member of the population
  score = rollout(env, policy1, policy2)

  # if score is positive, it means policy1 won.
  if score == 0: # if the game is tied, add noise to one of the agents.
    population[n] += np.random.normal(size=param_count) * epsilon
  if score > 0: # erase the loser, set it to the winner and add some noise
    population[n] = population[m] + np.random.normal(size=param_count) * epsilon
  if score < 0:
    population[m] = population[n] + np.random.normal(size=param_count) * epsilon
```

In the actual [code](https://github.com/hardmaru/slimevolleygym/blob/master/training_scripts/train_ga_selfplay.py), we track how many generations an agent has survived for (e.g. its evolutionary lineage), as a proxy for how good it is within the population without actually computing who is best to save time.

We ran this code for 500,000 tournaments on a single CPU which took only a few hours. Due to the simplicity of this method, the wall clock time per step is surprisingly fast compared to RL or even other ES algorithms. As a challenge, you may want to try implementing an asynchronous parallel processing version of this algorithm for performance.

After 500K games, the agent in the population that had the longest evolutionary lineage is used as a proxy for the best agent in the population. It played against the original baseline policy for the first time, and achieved an average score of 0.353 ± 0.728 over 1000 episodes. While it underperformed PPO and CMA-ES which trained directly against the baseline policy, we notice when we evaluated self-play GA against PPO and CMA-ES and measured the agents' performance head on, the GA completely dominated the PPO agent, and also beat the CMA-ES agent, suggesting that the earlier agents had somewhat overfit to a particular opponent's playing style.

We also logged the historical agent parameters during the tournament selection process, and evaluated each of them against the baseline policy afterwards to get a sense of the improvement over time:

![self_play_ga_training](figure/ga_results.svg)

**References**

*Blickle and Thiele, [A comparison of selection schemes used in evolutionary algorithms](https://pdfs.semanticscholar.org/a553/2dda955228ea44e2d224c6b42916959705b1.pdf), Evolutionary Computation, 1996.*

*Miller and Goldberg, [Genetic algorithms, tournament selection, and the effects of noise](https://pdfs.semanticscholar.org/df6e/e94e2cf14c38e9cff4d2446a50db0aedd4ca.pdf), Complex Systems, 1995.*

## Self-Play via PPO

Reinforcement learning can also incorporate self-play, by incorporating in the environment an earlier version of the agent, allowing the agent to continually learn to improve against itself. This approach also leads to a natural curriculum that adapts to the agent's current abilities, because unlike starting out against an expert opponent, here, the level of difficulty will be on par with the agent. An outline of a self-play algorithm for RL:

```text
Champion List:
Initially, this list contains a random policy agent.

Environment:
At the beginning of each episode, load the most recent agent archived in the Champion List.
Set this agent to be the Opponent.

Agent:
Trains inside the Environment against the Opponent with our choice of RL method.
Once the performance exceeds some threshold, checkpoint the agent into the Champion List.
```

There are a few ways we can implement this algorithm. We can wrap the multi-agent loop in a new gym environment, and train the agent in the new environment. Alternatively, an easier way is to directly replace the `policy` object in the `SlimeVolley-v0` environment with previous checkpointed PPO agents (see [code](https://github.com/hardmaru/slimevolleygym/blob/master/training_scripts/train_ppo_selfplay.py)). In our example, the current PPO agent must achieve an average score above 0.5 against the previous champion in order to become the next champion. After running the self-play code, the training converged at around 140 generations (it takes a long time after that to produce the next Champion). The most recent champion is then evaluated for the first time against the baseline policy achieving an average score of -0.371 ± 1.085 over 1000 episodes. We also logged all previous Champion policies and evaluated those as well against the baseline policy to retroactively measure its training progress:

![self_play_ppo_training](figure/sp_results.svg)

Note that Bansel et al. (see References below) discuss alternate ways to sample from the history of archived agents, since setting it to the most recent opponent may lead to the agent specializing at playing against its own policy. This may explain the fluctuations observed in the performance chart. In our [implementation](https://github.com/hardmaru/slimevolleygym/blob/master/training_scripts/train_ppo_selfplay.py), we modified the call back evaluation method in stable-baselines to assign a number label to each champion, so if we wanted to, we can implement such sampling methods, and we leave it as a challenge to the reader to experiment with variations.

**References**

Bansal et al., [Emergent complexity via multi-agent competition](https://arxiv.org/abs/1710.03748), ICLR, 2018.

## Self-Play by Cooperation via CMA-ES

The last self-play method we consider is one where agents don't learn by trying to destroy each other, but rather learn by cooperating with each other to achieve some mutually beneficial game state. This might be a weird concept. It's kind of like a GAN but where both networks work together to achieve a common dream (see [Stopping GAN Violence](https://arxiv.org/abs/1703.02528) :)

In this experiment, we will set the opponent's policy in the environment to be the *exact* same policy as our agent. This can be done in a similar way as discussed in the previous section. We won't train the agent to win against itself, but rather we train it to cooperate with a clone of itself to have it try to play Slime Volleyball *for as long as they can*, without either side losing prematurely before the maximum possible duration of 3000 timesteps.

Under this set up, CMA-ES can easily optimize the agent's parameters to maximize the expected duration of an episode:

![self_play_ppo_training](figure/cmaes_sp_results.svg)

This set up with a clear "survival" metric converges quickly to the maximum possible duration after a few hundred generations. We took the best "cooperative" agent in the population after 500 generations, and had it play, for the first time, against the baseline policy where it achieved an average score of -0.071 ± 0.827, which is not too bad considering that it wasn't trained to win. The performance against other agents, such as PPO and the earlier CMA-ES is also satisfactory, although it consistently loses against the other self-play methods.

There are a few reasons why this alternate setup is an effective way to indirectly learn an effective Slime Volleyball policy. Training an agent to be really good at *not losing* may allow it to indirectly learn defensive strategies against a good opponent. It can also wait until a weak opponent makes a mistake and likely result in the ball landing on the other side.

The survival time as a reward also makes the agent easier to train because it is easier to incrementally improve this measure of performance. As we have seen in the PPO section earlier, sometimes it can take a really long time for an initially random agent to learn from playing directly against an expert policy, as it is difficult to improve against getting the minimum possible score every time.

## Results Summary

Table of average scores of various methods discussed versus the default baseline policy ([1000 episodes](https://github.com/hardmaru/slimevolleygym/blob/master/eval_agents.py)):

|Method|Average Score
|---|---|
|Maximum Possible Score|5.0
|PPO | 1.377 ± 1.133
|CMA-ES | 1.148 ± 1.071
|GA (Self-Play) | 0.353 ± 0.728
|PPO (Self-Play) | -0.371 ± 1.085
|CMA-ES (Self-Play) | -0.071 ± 0.827
|Random Policy | -4.866 ± 0.372

Table of average scores of the discussed approaches versus each other ([1000 episodes](https://github.com/hardmaru/slimevolleygym/blob/master/eval_agents.py)):

|Method|PPO|CMA-ES|GA<br/>(Self-Play) |PPO<br/>(Self-Play) | CMA-ES<br/>(Self-Play)
|---|---|---|---|---|---|
|PPO | — |  0.133 ± 0.414 | -3.128 ± 1.509 | -0.119 ± 1.4 | 0.749 ± 0.846
|CMA-ES | -0.133 ± 0.414 | — | -0.301 ± 0.618  | 2.304 ± 1.392 | 0.351 ± 0.651
|GA<br/>(Self-Play) | 3.128 ± 1.509 | 0.301 ± 0.618 | —  | 0.42 ± 0.717 |  4.923 ± 0.342
|PPO<br/>(Self-Play) | 0.119 ± 1.46 |  -2.304 ± 1.392 | -0.420 ± 0.717 | — | 4.703 ± 0.582
|CMA-ES<br/>(Self-Play) | -0.749 ± 0.846 |  -0.351 ± 0.651 |  -4.923 ± 0.342  | -4.703 ± 0.582 | —

In the above table, the score represents the agent under the Method column playing against the Method in the top row. While we saw earlier that the simple GA didn't perform as well as methods that trained against the baseline policy, the GA ended up defeating all other approaches, and also completely dominated PPO. Performing well against one opponent may not necessarily transfer to other opponents.

# Pixel Observation Environment

Training an agent to play Slime Volleyball only from pixel observations is more challenging–not only does the agent need to work with a much larger observation space, it also needs to learn to infer important information such as velocities that are not explicitly provided. We approach this problem by taking advantage of the vast existing work in Deep RL that focused on training agents to play Atari games from pixels, and created the pixel version of the environment that looks and feels like an Atari gym environment. As an added bonus, we can use the same hyper parameters for existing methods that were already tuned for Atari games, without the need to perform hyper parameter search from scratch.

## Pixel Observation PPO

The PPO implementation in stable-baselines includes a CNN Policy for working with pixel observations. The standard pre-processing for Atari RL agents is to first convert each RGB frame into grayscale, resize them to 84x84 pixels, and consecutive 4 frames are stacked together as one observation so local temporal information could be inferred.

<p align="left">
  <img width="50%" src="https://media.giphy.com/media/W3NItV6PINmbgUFKPf/giphy.gif"></img>
  <br/><i>PPO trained to play from pixels using hyperparameters and settings (e.g. 4-frame stacking) pre-tuned for Atari.</i>
</p>

Although not required, we did find that it was easier to train the pixel observation version of the PPO agent using a reward function that incorporated a small survival reward discussed earlier (in the cooperative section) to facilitate early learning. This can be incorporated by applying the wrapper `SurvivalRewardEnv` over the original environment (before the Atari pre-processing), or simply make the environment using the registered env ID `SlimeVolleySurvivalNoFrameskip-v0` (refer to [code](https://github.com/hardmaru/slimevolleygym/blob/master/training_scripts/train_ppo_pixel.py)).

The agent is evaluated in the original environment without the extra survival bonus rewards. The best PPO agent (using pixel observations) achieved a score of 0.435 ± 0.961 versus the baseline policy (using state observations).

## Pixel Observation Rainbow

While PPO, GA, Evolution Strategies are all on-policy algorithms, we would like to test out off-policy methods, such as DQN, where batch-training can be done on a replay buffer that records an agent's historical experiences. In our experiments, DQN is not able to learn any reasonable policy for this task, so we tried the more advanced [Rainbow](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewFile/17204/16680) algorithm using [Chainer RL](https://github.com/chainer/chainerrl)'s implementation which has hyperparameters for reproducing the published Atari results.

<p align="left">
  <img width="50%%" src="https://media.giphy.com/media/hrox9TOfiChCpcbMYw/giphy.gif"></img>
  <br/><i>Example Rainbow agents trained to play from pixel observations using hyperparameters pre-tuned for Atari.</i>
</p>

After training the agent for around 25M timesteps, it achieved an average score of 0.037 ± 0.994 over 1000 episodes.

You can find the details, and also the pre-trained Rainbow model in a separate [repo](https://github.com/hardmaru/RainbowSlimeVolley) as it is too big to combine with the repo of the gym environment.
