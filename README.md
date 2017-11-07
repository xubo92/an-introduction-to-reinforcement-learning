# A-introduction-to-reinforcement-learning

## Instructions 

### - why do I maintain this repository?
Since it's hard for me to find a complete and feasible algorithms set on ‘Reinforcement Learning’ which is of vital help for freshers who want to get into this field.  

Nowthat I have made a detailed reading notes on the excellent book "A-introduction-to-reinforcement-learning" and implemented most of its algorithms, why not share them to public?   

I believe the quote > 'talk is cheap, show me the code'.  

### - what are in this repository?  
This repository consists of the algorithms from first 9 chapters in this book: [Reinforcement Learning:An introduction --2012 version](https://files.cnblogs.com/files/lvlvlvlvlv/SuttonBook.pdf)  

The algorithms after 9th chapter will be added continuously by another version of this book.  

- **chapter4** : Dynamic Programming  

&ensp;&ensp;Includes two exercises:  
  1. [The Gambler](https://github.com/lvlvlvlvlv/A-introduction-to-reinforcement-learning/blob/master/chapter4/The_Gambler.py) 
  2. [Jack's Car Rental](https://github.com/lvlvlvlvlv/A-introduction-to-reinforcement-learning/blob/master/chapter4/Jack%E2%80%99s_Car_Rental.py)  
  
- **chapter5** : Monte Carlo Methods  
&ensp;&ensp;Includes an exercise called "racetrack" and experiment performances on **racetrack**.  
**Note**:the Monte Carlo algorithms from chapter 5 are all implemented in the single file [Racetrack.py](https://github.com/lvlvlvlvlv/A-introduction-to-reinforcement-learning/blob/master/chapter5/Racetrack.py). Specifically in func: `def update_policy(episode):`  

- **chapter6** : Temporal-Difference Learning  

&ensp;&ensp;Includes td-related algorithms and experiment performances on **racetrack**.  
**Note**: From this chapter, I quit implementing environment of every exercise. Since almost each exercise has a different environment. If I just used different algorithms on different environments, one can hardly has a comparison between those algorithms, Therefore, I decided to show the performances of different algorithms on single same environment:**racetrack**.  
[td.py](https://github.com/lvlvlvlvlv/A-introduction-to-reinforcement-learning/blob/master/chapter6/td.py)

- **chapter7** : Eligibility Traces

&ensp;&ensp;Includes td-lambda related algorithms and experiment performances on **racetrack**
[td_lambda.py](https://github.com/lvlvlvlvlv/A-introduction-to-reinforcement-learning/blob/master/chapter7/td_lambda.py)

- **chapter8** : Planning and Learning with Tabular Methods

&ensp;&ensp;Includes Dyna_Q algorithm.
### - How to utilize them for your projects?
