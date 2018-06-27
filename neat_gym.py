import gym
import neat
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from random import random

import numpy as np
from Tkinter import *
from tqdm import tqdm
import threading

from time import sleep

plt.ion()

TEST=False

class Plotter():
    def __init__(self, frame):
        self.frame = frame

    def on_launch(self):
        #Set up plot
        self.figure = plt.figure()
        # Create 2x2 sub plots
        gs = gridspec.GridSpec(2, 2)
        self.ax1 = plt.subplot(gs[0, 0:])
        self.ax2 = plt.subplot(gs[1, 0:])
        #self.ax3 = plt.subplot(gs[1,1])
       
        self.ax1.plot([])
        self.ax1.set_title("Fitness evolution")
        self.ax1.set_xlim(0, 200)
        self.ax1.set_ylim(-500, 350)
        self.ax1.grid()
        self.ax1.set_xlabel("generation")
        self.ax1.set_ylabel("fitness")


        self.fitness_generation_hist = self.ax2.hist([])[2]
        #self.ax2.margins(2, 2)
        self.ax2.set_title("Generation's fitness")
        self.ax2.set_xlabel("fitness")
        self.ax2.set_autoscaley_on(True)

        #self.species_pie = self.ax3.pie([], labels=[])
        #self.ax3.margins(x=0, y=-0.25)
        #self.ax3.set_title('Species Domain')
        #self.ax3.set_autoscaley_on(True)

        self.figure.tight_layout()


    def on_running(self, plot_data=None, hist_data=None, pie_data=None):
        #Update data (with the new _and_ the old points)
        if plot_data is not None:
            y11,y12,y13  = plot_data
            self.ax1.clear()
            self.fitness_evolution_plot_mean = self.ax1.plot(y11, "*-")
            self.fitness_evolution_plot_best = self.ax1.plot(y12, "g*-")
            #self.fitness_evolution_plot = self.ax1.plot(y13, "r*-")
            #Need both of these in order to rescale
            #self.ax1.relim()
            self.ax1.set_xlim(0, 200)
            self.ax1.set_ylim(-500, 350)
            #self.ax1.autoscale_view()
            self.ax1.set_title("Fitness evolution")
            #self.ax1.set_autoscaley_on(True)
            self.ax1.grid()
            self.ax1.set_xlabel("generation")
            self.ax1.set_ylabel("fitness")
            self.ax1.legend([self.fitness_evolution_plot_mean, self.fitness_evolution_plot_best], ['Mean', 'Best'])


        if hist_data is not None:
            x2, generation = hist_data
            #Need both of these in order to rescale
            self.ax2.clear()
            self.fitness_generation_hist = self.ax2.hist(x2)
            self.ax2.set_title("Generation " + str(generation) )
            self.ax2.set_xlabel("fitness")
            #self.ax2.set_autoscaley_on(True)


        if pie_data is not None:
            pass
            #x3, x3_labels = pie_data
            #self.ax3.clear()
            #self.species_pie = self.ax3.pie(x3, labels=x3_labels)
            #self.ax3.margins(x=0, y=-0.25)
            #self.ax3.set_title('Species Domain')
            #self.ax3.set_autoscaley_on(True)

        #We need to draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

class App(Frame):
    def load_config(self):
        self.games_config = \
        {
            "Lunar lander" :
                {
                    "neat_config_file" : "./config-feedforward.txt",
                    "gym_environment" : "LunarLander-v2",
                    "actions_mapper" : lambda l: 2*np.array(l, float) - 1
                },
            "Lunar lander Continuous" :
                {
                    "neat_config_file" : "./config-feedforward.txt",
                    "gym_environment" : "LunarLanderContinuous-v2",
                    "actions_mapper" : lambda l: 2*np.array(l, float) - 1
                }
        }

    def abort(self):
        if self.env_thread is not None:
            self.env_signal = "STOP"
            self.env_thread.join()

        if self.plot_thread is not None:
            self.plot_signal = "STOP"
            self.plot_update_event.set()
            self.plot_thread.join()

        self.quit()

    def skip(self):
        self.env_signal = "SKIP"

    def skip_generation(self):
        self.env_signal = "SKIPGEN"

    def skip_20_generations(self):
        self.env_signal = "SKIP20"

    def run_plotter(self):
        while True:
            #{"population": len(genomes), "fitness": generations_fitness, "min":min(generations_fitness), "max":max(generations_fitness)}
            _ = self.plot_update_event.wait()
            
            if self.plot_signal == "STOP":
                return

            gen = len(self.evolution_log) - 1
            x2 = self.evolution_log[-1]["fitness"]

            y11 = []
            y12 = []
            y13 = []
            for generation in self.evolution_log:
                y11.append(sum(generation["fitness"])/ float(generation["population"]))  
                y12.append(generation["max"])
                y13.append(generation["min"])

            self.plotter.on_running(plot_data=(y11,y12,y13), hist_data=(x2,gen))
            

            # if len(self.evolution_log) > 0 and self.plot_signal == "MEM":
            #     x2 = self.evolution_log[-1]["fitness"]
            #     print(self.plot_signal)
            #     self.plotter.on_running(hist_data=x2)
            # elif self.plot_signal == "GEN":
            #     print(self.plot_signal)
            # elif self.plot_signal == "STOP":
            #     print("exit")
            self.plot_signal = "WAIT"
            self.plot_update_event.clear()
        

    def run_evolution(self):
    
        self.generation = 0
        
        def game_loop(genomes, config):
            generations_fitness = []
            show_generation = True
            for genomeid, genome in tqdm(genomes, desc='Generation ' + str(self.generation), postfix="Genome:"):
                rewards = 0
                done = False
                show_member = True
                net = neat.nn.FeedForwardNetwork.create(genome, config)
                observation = self.env.reset()
                while not done:
                    if self.env_signal == "STOP":
                        exit()
                    elif self.env_signal == "SKIP":
                        show_member = False
                    elif self.env_signal == "SKIPGEN":
                        show_generation = False
                    elif self.env_signal == "SKIP20":
                        self._skip_generations = 20
                    elif self.env_signal == "START":
                        pass
                    self.env_signal = "STEP"

                    if self._skip_generations > 0:
                        show_generation = False

                    if not TEST and show_generation and show_member:
                        self.env.render()


                    if TEST:
                        action = self.env.action_space.sample()
                    else:
                        action = self.actions_mapper(net.activate(observation))
                    observation, reward, done, info = self.env.step(action)
                    rewards += reward
                
                genome.fitness = rewards
                generations_fitness.append(rewards) 
          

            #update hist
            self.evolution_log.append({"population": len(genomes), "fitness":generations_fitness , "min":min(generations_fitness), "max":max(generations_fitness)})
            self.generation += 1    
            self.plot_signal = "MEM"
            self.plot_update_event.set()

            if self._skip_generations > 0:
                self._skip_generations -= 1
            self.plot_signal = "UPDATE"
            

        self._skip_generations = 0
        self.population.run(game_loop)

    def startScene(self):
        # load widgets
        self.abort_button = Button(self)

        self.abort_button["text"] = "ABORT!"
        self.abort_button["fg"]   = "red"
        self.abort_button["command"] =  self.abort
        self.abort_button.pack({"side": "left"})

        self.the_game = StringVar(self)
        self.the_game.set(self.games_config.keys()[0])
        self.games_options = OptionMenu(self, self.the_game, *self.games_config.keys())
        self.games_options.config(width=30)
        self.games_options.pack({"side": "left"})

        self.start_button = Button(self)
        self.start_button["text"] = "START"
        self.start_button["fg"] = "blue"
        self.start_button["command"] = self.evolutionScene
        self.start_button.pack({"side": "left"})

        #threads
        self.env_thread = None
        self.plot_thread = None

    def evolutionScene(self):

        # change widgets
        self.start_button.pack_forget()
        self.games_options.pack_forget()

        self.next_button = Button(self)
        self.next_button["text"] = "Next"
        self.next_button["command"] = self.skip
        self.next_button.pack({"side": "left"})

        self.skip_genomes_button = Button(self)
        self.skip_genomes_button["text"] = "Next Generation"
        self.skip_genomes_button["command"] = self.skip_generation
        self.skip_genomes_button.pack({"side": "left"})

        self.skip_generations_button = Button(self)
        self.skip_generations_button["text"] = "Skip 20 Generations"
        self.skip_generations_button["command"] = self.skip_20_generations
        self.skip_generations_button.pack({"side": "left"})

        # setup environment
        self._env_config = self.games_config[self.the_game.get()]
        self.env = gym.make(self._env_config["gym_environment"])
        self._neat_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        self._env_config["neat_config_file"])
        self.population = neat.Population(self._neat_config)
        self.actions_mapper = self._env_config["actions_mapper"]

        # setup plotter and info
        self.plotter = Plotter(self)
        self.plotter.on_launch()
        # new Frame
        #self.plotter_frame = Toplevel(self)
        #button = Button(self.plotter_frame)
        #button["text"] = "ABORT!"
        #button["fg"]   = "red"
        #button["command"] =  self.abort
        #button.pack({"side": "left"})

        self.plot_signal = "SETUP"
        self.plot_update_event = threading.Event()
        self.evolution_log = []
        self.plot_thread = threading.Thread(name="plot", target=self.run_plotter)
        self.plot_thread.start()

        #self.best =self.population.run(self.game_loop)
        self.env_signal = "START"
        self.env_thread = threading.Thread(name="env", target=self.run_evolution)
        self.env_thread.start()

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master.title("Neat Gym")
        self.load_config()
        self.pack()
        self.startScene()

root = Tk()
app = App(master=root)
app.mainloop()