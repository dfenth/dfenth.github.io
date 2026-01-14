---
layout: post
title: "Hacking Pokemon for R(ea)L fun"
date: 2026-01-14 15:00:00 +0000
categories: research ai rl
usemathjax: true

---
My first real mind-blowing experience with AI came from DeepMind's [Atari Deep Q-learning agent](https://arxiv.org/abs/1312.5602) way back in 2013! Seeing a computer gradually learn to become proficient at a game and then surpass human experts really captured my imagination and was the catalyst for my investment (through education and research) in AI. Since that time, the AI space has exploded with breakthroughs and investment, but with the renewed interest, I feel like the space has lost some of the magic.

In an attempt to find that spark again, I thought I'd embark on a project that aims to be nothing but an interesting exercise for my own amusement. I've always been interested in Reinforcement Learning (RL), but haven't had many opportunities to work in the space (my research is mostly in computer vision/neural network fundamentals), so, to honour DeepMind's Atari agent, I thought I'd make a computer play a game.

## Setup
The game in question is [Pokémon Crystal](https://en.wikipedia.org/wiki/Pok%C3%A9mon_Crystal). If you're unfamiliar with the Pokémon games, you control a character who wanders around a large (mostly linear) world, collecting and battling Pokémon. You collect Pokémon by walking through tall grass in the world, which can trigger random battles with wild Pokémon. In these battles, you can throw Pokéballs which have a percentage chance of capturing the Pokémon you're battling against (with the percentage dependent on the amount of health the wild Pokémon has, along with multiple other factors). If you successfully capture the Pokémon, you can use it in subsequent battles. Game progress is blocked by 'trainer battles', which are battles against other Pokémon trainers. The aim of the game is to beat the 8 gym leaders (more powerful trainers), get the 8 gym badges, then beat the elite 4 (even more powerful trainers), and finally the champion (the most powerful trainer). There are some side stories along the way as well, but that's the basic premise. So, from an AI perspective, the game can be split into two: 

1. Navigation around the world (knowing/finding out where to go next)
2. Beating trainers in Pokémon battles

A few years ago, Peter Whidden published a [YouTube video](https://www.youtube.com/watch?v=DcYLT37ImBY) where he [trained an RL agent to play Pokémon Red](https://github.com/PWhiddy/PokemonRedExperiments). This is a seriously impressive project that addresses the two halves of the game, with the agent both navigating the world and being competent enough to beat the (not very challenging) CPU-controlled trainers. My attempt takes a much less complete approach. I don't have the money to pay for either GPUs or GPU time, so I thought I'd work on something smaller. I'll focus on just winning some battles.

By focusing on a subset of the game (the battles), we can significantly reduce the RL agent's search space, leading to faster training, lower resource requirements, and faster iteration. We can further reduce the search space by intelligently setting the RL agent's world state, but I'll discuss this more later!

If you're unfamiliar with the Pokémon battle system, it's a synchronous turn-based system where all users (AI agents, CPU-controlled agents, or human users) select an action at the same time, and those actions then play out over a turn. The actions can include using an attack (that has a limited number of uses), swapping to a different Pokémon, or using an item. Once a turn has played out, the new turn begins (actions are chosen), and this repeats until all of your, or your opponent's Pokémon have fainted (have 0 health points (HP)). The strategy of Pokémon comes from matching types of Pokémon. Each Pokémon has an elemental type (e.g. Fire, Water, Grass, etc.), with some types being stronger against others. When battling, you want to use a Pokémon that has a type advantage against the opponent to deal maximum damage to them and take less damage yourself. That is the very basic explanation. You can also have different attack types like physical and special, with Pokémon having better defences against physical and special attacks, and then moves that increase the attack/defence ability of your Pokémon, or decrease the attack/defence ability of the opponent, but I'm not skilled enough to write a primer on optimal Pokémon battle techniques (that's for the agent to find out).

## Game emulation
To run the game and pass input to an RL model, I'm using [PyBoy](https://github.com/Baekalfen/PyBoy), which is a Game Boy/Game Boy Color emulator written in Python. It gives us access to the screen (pixel values in a convenient numpy array), we can send inputs (d-pad, a, b, start, select) and can save and reload states (useful for resetting runs). The emulator can also run at uncapped speeds, allowing us to train rapidly since each battle usually takes a few minutes. All of these features provide us with a suitable environment for automating Pokémon gameplay. 

When training an RL agent, we need a metric to assess how well it's performing and to guide future behaviour with rewards or the promise of a reward. As a first idea, the health of the opponent Pokémon seems like a good place to start. The question is, how do we get this information for the reward? 

PyBoy emulates the hardware of the Game Boy Color, recreating the CPU, sound and graphics processor as closely as possible (or as closely as is needed for the program's aims). This extends to the behaviour and setup of RAM. Since the opponent's Pokémon's health is in the game, it should be somewhere in RAM 🙄.

## Hacking Pokémon Crystal
The 'RAM map' of Pokémon Crystal is quite well documented (I started with [datacrystal](https://datacrystal.tcrf.net/wiki/Pok%C3%A9mon_Crystal/RAM_map), then moved on to [Glitch City](https://archives.glitchcity.info/forums/board-76/thread-1342/page-0.html), which seems more complete). From the RAM map, we can find the locations of many pieces of data from the game, e.g. the amount of money the player has (`0xd84e-0xd850`), or the badges they have from the Johto region (`0xd857`). 

Checking the RAM map, we can easily find the memory address for the opponent Pokémon's health (`0xd216-0xd217`).

Now, while we're here, it's fun to read program data from RAM. But it's more fun to write to it! 😏 Tangent time!

Let's start with something easy. Address `0xdca1` tells the game how many repel steps we have left (repels block wild Pokémon encounters). So, if we change that value in RAM, then we shouldn't encounter a wild Pokémon!
So, I set up the code:
```python
start_time = time.time()
prev_repel = pb.memory[0xdca1]

while pb.tick():
    if time.time() - start_time > 10:
        print("Reset repel")
        pb.memory[0xdca1] = 10
        start_time = time.time()
    
    # Print the repel value if it changes
    if pb.memory[0xdca1] != prev_repel:
        print(pb.memory[0xdca1])
        prev_repel = pb.memory[0xdca1]
```
This code updates our repel count to 10 every 10 seconds, and prints out the new repel value every time we take a step. What's interesting is that the game also reports when we run out of repel:

<video controls width="640">
  <source src="{{ '/research/ai/rl/2026/01/14/res/pkm_hack_repel.mp4' | relative_url }}" type="video/mp4">
  Your browser does not support the video tag.
</video>

This works!

Let's do a little more experimentation. Let's try to force a specific Pokémon (Celebi) to appear during a wild Pokémon battle, and set that Pokémon's health to 1:
```python
# Set's Pokémon health to 1 across the two bytes that represent it
pb.memory[0xd216] = 0
pb.memory[0xd217] = 1
```
According to the memory map, `0xd204`, `0xd206` and `0xd22e` correspond to the ID of the opposing Pokémon. When we start a random battle, those memory values change to the Pokémon's Pokédex number. So what happens if we overwrite these values with a target Pokédex ID?
```python
pb.memory[0xd204] = 251
pb.memory[0xd206] = 251
pb.memory[0xd22e] = 251
```

<video controls width="640">
  <source src="{{ '/research/ai/rl/2026/01/14/res/target_pkm_attempt_1.mp4' | relative_url }}" type="video/mp4">
  Your browser does not support the video tag.
</video>

Celebi (Pokédex ID 251) should not be in this patch of grass. In fact, Celebi was a special legendary Pokémon that could only be captured in Ilex Forest (and was only available from a special event in Gold and Silver).

Now we're comfortable with hacking RAM values, what do we want to achieve? Well, we'd like an agent that can take any team of Pokémon at any level and beat the CPU-controlled opponent in a reasonable battle (e.g., if the agent has level 1 Pokémon and the CPU has level 100, I don't expect it to win that match-up!).

What do we need to achieve this?
- Have a save state that loads the game *just* before a battle begins (i.e. a few frames before).
- Set the agent's and CPU's party to some arbitrary Pokémon with arbitrary moves and arbitrary levels.
- Provide the RL agent with the world state at the start of each turn.

Let's get started!

## Setting up the AI environment

### Battle state
PyBoy allows us to save the state of a game, which captures the contents of RAM, along with the execution environment (program counter, etc.) at any point. I chose to save just before your first battle with your rival. My first attempt was too early: the rival's Pokémon were set after the first frame was executed, overwriting the hacked team I gave him. After a little trial and error, I managed to save a state just after the rival's Pokémon were set, which means that we have free rein to set whatever Pokémon we wanted him to have in memory, and it will not be overwritten with any initialisations at the start of the battle.

### Setting Agent and CPU Pokémon
This was a little more involved 😅. The teams of Pokémon the agent and CPU can have are nicely arranged in memory, with one Pokémon following on from the previous and all of the properties being consistent in their offset from the first property (Pokémon species (the Pokédex ID)). This makes it easy for us to automate the population of this data.

Apart from the values of `start_addr`, `species_addr` and `name_addr`, which change depending on whether we're populating the agent or CPU Pokémon and which Pokémon number (1-6), all of the data can be filled in as an offset:
```python
memory[start_addr] = pkm_data['dex_id']
memory[species_addr] = pkm_data['dex_id']

memory[start_addr+1] = 0 # Held item

memory[start_addr+2:start_addr+6] = pkm_data['moves']

# EV values don't matter here
memory[start_addr+11:start_addr+13] = store_integer(0, 2) # HP EV
memory[start_addr+13:start_addr+15] = store_integer(0, 2) # Attk EV
memory[start_addr+15:start_addr+17] = store_integer(0, 2) # Def EV
memory[start_addr+17:start_addr+19] = store_integer(0, 2) # Speed EV
memory[start_addr+19:start_addr+21] = store_integer(0, 2) # Special EV

memory[start_addr+23:start_addr+27] = pkm_data['move_pp']

memory[start_addr+31] = pkm_data['level'] # Pkm level
memory[start_addr+9:start_addr+11] = store_integer(0, 2)

memory[start_addr+33] = 255 # Seems to be related to evolution? Setting this to 255 stops evolution

memory[start_addr+34:start_addr+36] = store_integer(pkm_data['hp'], 2)
memory[start_addr+36:start_addr+38] = store_integer(pkm_data['hp'], 2)
memory[start_addr+38:start_addr+40] = store_integer(pkm_data['attk'], 2)
memory[start_addr+40:start_addr+42] = store_integer(pkm_data['def'], 2)
memory[start_addr+42:start_addr+44] = store_integer(pkm_data['spd'], 2)
memory[start_addr+44:start_addr+46] = store_integer(pkm_data['spatk'], 2)
memory[start_addr+46:start_addr+48] = store_integer(pkm_data['spdef'], 2)

if pkm_data['nickname'] == None:
	pkm_data['nickname'] = "TEST"

nickname = [int(chr_to_hex[s], 16) for s in pkm_data['nickname']]

if len(nickname) > 10:
	long_name = nickname[:10] + [int(0x50)]
else:
	long_name = nickname + [int(0x50)]*(11-len(nickname))

memory[name_addr[1]:name_addr[1]+11] = long_name # 11 chars (10 with terminating hex 50)
```

In the code above, we're populating the memory addresses with data from a dictionary (`pkm_data`), but where does this data come from? As Pokémon fans know, the Pokémon properties (like HP, attack, and defence) depend on the Pokémon itself (some Pokémon have naturally higher attack or defence) and the level. If we want to create teams of Pokémon of arbitrary levels, we need to know how to calculate the correct properties. This is where Pokémon base stats, EVs and IVs come in... 😮‍💨.

 Each Pokémon has a series of base stats for HP, attack, defence, special attack, special defence and speed. These base stats kind of tell you what the Pokémon is good at (e.g., a high defence stat tells you that a Pokémon can take a lot of hits, a high speed stat tells you that the Pokémon is likely to make its move first in the turn order). These values change as the Pokémon levels up, but the way they change depends on both the EVs and the IVs. EVs are given to a Pokémon after they defeat another Pokémon in battle, based on the base stats of the Pokémon it defeated (kind of like absorbing the power of enemies you defeat 😬). As the Pokémon defeats enemies, these EVs accrue and contribute to the increase in stats during level up (I don't think the EVs reset at level up). The maximum EV for each stat is 65535. Pokémon IVs (Individual Values) are kind of like a Pokémon's genes. They are values specific to each individual Pokémon that determine how well it scales in a particular stat (the maximum IV is 15 for each stat (0b1111)). For example, Umbreon (the best Pokémon) can have stats:

|              | **HP**  | Attack  | Defence | Sp.Attack | Sp.Defence | Speed   |
| ------------ | ------- | ------- | ------- | --------- | ---------- | ------- |
| **Base**     | 95      | 65      | 110     | 60        | 130        | 65      |
| **Lvl. 50**  | 155-201 | 70-116  | 115-161 | 65-111    | 135-181    | 70-116  |
| **Lvl. 100** | 300-393 | 135-228 | 225-318 | 125-218   | 265-358    | 135-228 |

(Data from [Serebii](https://www.serebii.net/pokedex-gs/197.shtml)).

The variation in stats comes from the EVs and IVs. The equation for calculating a Pokémon's HP at any level based on base stats, EVs and IVs is:

 $$
 \text{HP} = \left\lfloor \frac{\left( (\text{Base} + \text{IV}) \cdot 2 + \left\lfloor \frac{\left\lceil \sqrt{\text{EV}} \right\rceil}{4} \right\rfloor \right) \cdot \text{level}}{100} \right\rfloor + \text{level} + 10.
$$

With all other stats calculated as:

$$
 \text{Stat} = \left\lfloor \frac{\left( (\text{Base} + \text{IV}) \cdot 2 + \left\lfloor \frac{\left\lceil \sqrt{\text{EV}} \right\rceil}{4} \right\rfloor \right) \cdot \text{level}}{100} \right\rfloor + 5.
$$

(Equations from [Bulbapedia](https://bulbapedia.bulbagarden.net/wiki/Individual_values)). To keep things simple, when giving the agent or CPU Pokémon, we give them with the maximum stats (EVs of 65535 and IVs of 15). An interesting caveat to note in the calculation is that $\sqrt{\text{EV}}$  is capped at 255.

So, we can see how we can add whatever Pokémon we want to the team, and calculate the correct (maximised) stats at whatever level. But we still have two issues. How do we know the Pokémon's base stats, and how do we know what moves we can assign to the Pokémon, consistent with its level?

There are many Pokémon databases out there, but the one by [veekun](https://github.com/veekun/pokedex) does exactly what we need. It just has a bunch of CSV files that we can comb through to get the data we need. In particular, a [list of all Pokémon](https://github.com/veekun/pokedex/blob/master/pokedex/data/csv/pokemon.csv) (so we can find the Pokémon name based on the Pokédex ID), a [list of Pokémon stats](https://github.com/veekun/pokedex/blob/master/pokedex/data/csv/pokemon_stats.csv) (so we can get the Pokémon base stats), a [list of all possible Pokémon moves](https://github.com/veekun/pokedex/blob/master/pokedex/data/csv/moves.csv), and finally [a list of moves that each Pokémon can learn](https://github.com/veekun/pokedex/blob/master/pokedex/data/csv/pokemon_moves.csv), and the levels they learn them at. With some cool SQL inner-joins (that I did have LLM help with as it's been a while since I've done databases), we can query any Pokémon from Gen II (the generation of Crystal), get the Pokémon base stats, and find all possible moves it can learn, and the level it can use those moves at. Here's how we can create the Pokémon in code:

```python
pkm_info = process_input_pkm([
    ("Umbreon", "Hello", 100, ["tackle", "pursuit", "moonlight", "bite"]),
    ("porygon2", "World", 100, ["conversion-2", "psybeam", "tri-attack", "zap-cannon"]),
    ("skarmory", "This", 100, ["steel-wing", "peck", "fly", "sky-attack"]),
    ("Entei", "Seems", 100, ["flamethrower", "fire-blast", "stomp", "roar"]),
    ("gyarados", "To", 100, ["hydro-pump", "Rain-dance", "Hyper-Beam", "Dragon-Breath"]),
    ("pikachu", "Work", 100, ["Thunderbolt", "dynamic-punch", "thunder", "mega-punch"])
])
```

This gives us the result:

<video controls width="640">
  <source src="{{ '/research/ai/rl/2026/01/14/res/pkm_custom_team.mp4' | relative_url }}" type="video/mp4">
  Your browser does not support the video tag.
</video>

### Getting a world state
Now that we can set the agent and CPU Pokémon arbitrarily, we can re-visit the our todo list:
- ~~Have a save state that loads the game *just* before a battle begins (i.e. a few frames before).~~
- ~~Set the agent's and CPU's party to some arbitrary Pokémon with arbitrary moves and arbitrary levels.~~
- Provide the RL agent with the world state at the start of each turn.

Now we *just* need to provide the RL agent with the world state so it can decide what to do at some regular interval. In Pokémon battles, we have turns, so getting a world state at the start of the turn, then deciding what to do with that turn sounds like an obvious choice. But how do we invoke the agent at the correct time?

This seems like a ridiculous question. There should *just* be a number somewhere in memory that increments with the turn, right? Well, it doesn't seem to be as easy as that. First off, there is no documented turn counter reported by [datacrystal](https://datacrystal.tcrf.net/wiki/Pok%C3%A9mon_Crystal/RAM_map) or [Glitch City](https://archives.glitchcity.info/forums/board-76/thread-1342/page-0.html), and second, as I'll discuss later, there is no (to my knowledge) clean turn transition point.

To investigate how we can call the agent at the correct time, I created a live memory map using [PyGame](https://www.pygame.org/news). The live memory map shows how specified RAM values change over time (my very simple implementation changes the colour of a block when the previous frame had a different value in that RAM address). This should allow us to see which RAM values are associated with the turn transition. The [Pan Docs memory map](https://gbdev.io/pandocs/Memory_Map.html) tells us that the Game Boy Color working RAM (WRAM) is between `0xc000 - 0xdfff`, which also aligns with the RAM values we've been changing to manipulate the game! So I'll start from `0xc000-0xcfff` to keep things manageable:

<video controls width="640">
  <source src="{{ '/research/ai/rl/2026/01/14/res/c_block_mem_check.mp4' | relative_url }}" type="video/mp4">
  Your browser does not support the video tag.
</video>

Now `0xd000-0xdfff`:

<video controls width="640">
  <source src="{{ '/research/ai/rl/2026/01/14/res/d_block_mem_check.mp4' | relative_url }}" type="video/mp4">
  Your browser does not support the video tag.
</video>

We can see much more activity in the `0xc...` block of memory than `0xd...`, but both are quite busy around the turn transition.

After a lot of trial-and-error, narrowing down, and times of pure frustration, I think that memory address `0xc6dc` is the turn counter. So, we can grab the world state (Pokémon health information, number of times we can use a move etc.) when the data in location `0xc6dc` changes? Of course not! From some experimentation, the turn counter increments before all damage calculations have been done, so the health of the Pokémon is incorrect. This means that we need to wait until the official beginning of the next turn i.e. when the user can actually provide inputs. 😤 Yet another hurdle. Almost like the game was made with little regard to how an AI could be trained on it in the future...

Something that I noticed while inspecting the RAM of the running game, was that when the user selects a different option (Fight, Pkmn, Pack, Run), there is a memory address (`0xcfac`) that responds to this, storing a different value for each selected value. The fight action is always selected at the start of the turn, clearing whatever was selected at the end of the last turn, meaning that we usually have a transition from some value to 193. So, to detect the start of the turn we can trigger on this transition. The problem is that if we just listen for a transition from any number to 193, we will trigger when we select a different option (e.g. Pkmn) then change our mind and go back to fight. But none of the selections set memory location `0xcfac` to 0. So, can we listen for the official turn change measured by `0xc6dc`, set the value of `0xcfac` to 0, and then listen for the transition from 0 → 193?

<video controls width="640">
  <source src="{{ '/research/ai/rl/2026/01/14/res/turn_transition.mp4' | relative_url }}" type="video/mp4">
  Your browser does not support the video tag.
</video>

Yes. It works... 😵‍💫

Back to the todo list:
- ~~Have a save state that loads the game *just* before a battle begins (i.e. a few frames before).~~
- ~~Set the agent's and CPU's party to some arbitrary Pokémon with arbitrary moves and arbitrary levels.~~
- ~~Provide the RL agent with the world state at the start of each turn.~~

Amazing! 

## Next steps
You may have been a bit confused by some of the choices being made in that battle. That may be because an agent made those choices at random! So now we need to train it to make intelligent decisions!

This brings you up to date with where I'm currently at. The next steps are going to be the tricky part (as if everything else was simple to do). Most of the future complexity will be around the formal definition of the problem and the architecture of the RL model.

I started this project thinking that I'd define the world state as the observables of the game (agent Pokémon health, amount of uses each of each move of the agent's Pokémon, opponent health), while this gives us a very restricted search space (which hopefully translates to fast training), it may also be too restricted to allow for generality if we change Pokémon teams. It also means the agent may not be able to learn Pokémon type matchups, since we never explicitly state what our Pokémon is or what the opponent's is. Time and experiments will tell, but I'm pleased with the progress I've been able to make, and look forward to making more mistakes in the future!

This has been a really awesome project so far. I've learned a lot about hacking Game Boy Color games (not sure how useful that is), and I really had to think about the design of the pipeline feeding an AI model (in something other than the image-based networks I'm used to), specifically in terms of how we represent the world with data that can be used to find something useful.

I'm still feeling energised to get this project finished and hope to have more to show soon!