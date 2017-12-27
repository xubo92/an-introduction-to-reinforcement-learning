
# A-introduction-to-reinforcement-learning

## Instructions 

### - why do I maintain this repository?
Since it's hard for me to find a complete and feasible algorithms set on ‘Reinforcement Learning’ which is of vital help for freshers who want to get into this field.  

Nowthat I have made a detailed reading notes on the excellent book "A-introduction-to-reinforcement-learning" and implemented most of its algorithms, why not share them to public?   

I believe the quote > 'talk is cheap, show me the code'.  

### - what are in this repository?  
This repository consists of the algorithms from first 9 chapters in this book: [Reinforcement Learning:An introduction --2012 version](https://files.cnblogs.com/files/lvlvlvlvlv/SuttonBook.pdf)  

![on-policy MC](https://github.com/lvlvlvlvlv/A-introduction-to-reinforcement-learning/blob/master/RLF/MC_ON-POLICY_RACETRACK.png)

The algorithms after 9th chapter will be added continuously by another version of this book.  

- **chapter4** : Dynamic Programming  

  Includes two exercises:  
  1. [The Gambler](https://github.com/lvlvlvlvlv/A-introduction-to-reinforcement-learning/blob/master/chapter4/The_Gambler.py) 
  2. [Jack's Car Rental](https://github.com/lvlvlvlvlv/A-introduction-to-reinforcement-learning/blob/master/chapter4/Jack%E2%80%99s_Car_Rental.py)  
  
- **chapter5** : Monte Carlo Methods  

  Includes an exercise called "racetrack" and experiment performances on **racetrack**.  
  1. [Racetrack.py](https://github.com/lvlvlvlvlv/A-introduction-to-reinforcement-learning/blob/master/chapter5/Racetrack.py)
  
  **Note**:the Monte Carlo algorithms from chapter 5 are all implemented in the single file [Racetrack.py](https://github.com/lvlvlvlvlv/A-introduction-to-reinforcement-learning/blob/master/chapter5/Racetrack.py). Specifically in func: `def update_policy(episode):`  

- **chapter6** : Temporal-Difference Learning  

  Includes td-related algorithms and experiment performances on **racetrack**.  
  1. [td.py](https://github.com/lvlvlvlvlv/A-introduction-to-reinforcement-learning/blob/master/chapter6/td.py)  
  
  **Note**: From this chapter, I quit implementing environment of every exercise. Since almost each exercise has a different environment. If I just used different algorithms on different environments, one can hardly has a comparison between those algorithms, Therefore, I decided to show the performances of different algorithms on single same environment: **racetrack**.  
  

- **chapter7** : Eligibility Traces

  Includes td-lambda related algorithms and experiment performances on **racetrack**  
  1. [td_lambda.py](https://github.com/lvlvlvlvlv/A-introduction-to-reinforcement-learning/blob/master/chapter7/td_lambda.py)

- **chapter8** : Planning and Learning with Tabular Methods

  Includes Dyna_Q algorithm.  
  1. [Dyna_Q](https://github.com/lvlvlvlvlv/A-introduction-to-reinforcement-learning/blob/master/chapter8/Dyna_Q.py)  

- **RLF** ： An reinforcement learning algorithms library which pulls together all algorithms mentioned above and *some new deep reinforment learning algorithms like DQN or DDPG* for purpose of convenient external call.  
  **Note**: One remarkable point of this library is the seperation of *'environment'* and *'agent algorithm'* via the attribute of 'python Class'. In this way, you could add or modify your own environment without interfering agent part. Also, you can just connect agent algorithms(td,monte-carlo,dp,...) to your environment.  
  + environment code:
    - [env.py](https://github.com/lvlvlvlvlv/A-introduction-to-reinforcement-learning/blob/master/RLF/env.py)
  + agent algorithms:  
    - [monte_carlo.py](https://github.com/lvlvlvlvlv/A-introduction-to-reinforcement-learning/blob/master/RLF/monte_carlo.py)
    - [dp.py](https://github.com/lvlvlvlvlv/A-introduction-to-reinforcement-learning/blob/master/RLF/dp.py)
    - [td.py](https://github.com/lvlvlvlvlv/A-introduction-to-reinforcement-learning/blob/master/RLF/td.py)
    - [td_lambda.py](https://github.com/lvlvlvlvlv/A-introduction-to-reinforcement-learning/blob/master/RLF/td_lambda.py)
    - [Dyna_Q.py](https://github.com/lvlvlvlvlv/A-introduction-to-reinforcement-learning/blob/master/RLF/Dyna_Q.py)
    - [DQN.py](https://github.com/lvlvlvlvlv/A-introduction-to-reinforcement-learning/blob/master/RLF/DQN.py)
  + main function:
    - [main.py](https://github.com/lvlvlvlvlv/A-introduction-to-reinforcement-learning/blob/master/RLF/main.py)
    
### - How to utilize them for your projects?
