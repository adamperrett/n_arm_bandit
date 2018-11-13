import spynnaker8 as p
# from spynnaker_external_devices_plugin.pyNN.connections.\
#     spynnaker_live_spikes_connection import SpynnakerLiveSpikesConnection
from spynnaker.pyNN.connections.\
    spynnaker_live_spikes_connection import SpynnakerLiveSpikesConnection
from spinn_front_end_common.utilities.globals_variables import get_simulator
from pympler.tracker import SummaryTracker

import pylab
from spynnaker.pyNN.spynnaker_external_device_plugin_manager import \
    SpynnakerExternalDevicePluginManager as ex
from spinn_bandit.python_models.bandit import Bandit
import sys, os
import time
import socket
import numpy as np
import math
import csv
import traceback
import random
import gc
import fnmatch
import matplotlib.pyplot as plt

from pyNN.utility.plotting import Figure, Panel

from peas.methods.neat import NEATPopulation, NEATGenotype
from peas.networks.rnn import NeuralNetwork

def read_agent():
    agent_connections = []
    agent_nodes = []
    agent_genes = []
    with open(file) as from_file:
        csvFile = csv.reader(from_file)
        for row in csvFile:
            temp = row
            fitness_check = fnmatch.filter([temp[0]], 'fitness*')
            if temp[1] == 'excitatory' or temp[1] == 'inhibitory' or temp[1] == 'sin' or temp[1] == 'bound' or \
                    temp[1] == 'linear' or temp[1] == 'gauss' or temp[1] == 'sigmoid' or temp[1] == 'abs':
                # for i in range(len(temp)):
                #     agent_nodes.append(temp[i])
                agent_nodes.append(temp)
            elif len(fitness_check) > 0:
                break
            else:
                # for i in range(len(temp)):
                #     agent_connections.append(temp[i])
                agent_connections.append(temp)

    agent_genes.append(agent_connections)
    agent_genes.append(agent_nodes)

    return agent_genes

def get_scores(bandit_pop,simulator):
    b_vertex = bandit_pop._vertex
    scores = b_vertex.get_data(
        'score', simulator.no_machine_time_steps, simulator.placements,
        simulator.graph_mapper, simulator.buffer_manager, simulator.machine_time_step)

    return scores.tolist()

def row_col_to_input_breakout(row, col, is_on_input, row_bits, event_bits=1, colour_bits=1, row_start=0):
    row_bits = np.uint32(row_bits)
    idx = np.uint32(0)

    if is_on_input:
        idx = 1

    row += row_start
    idx = idx | (row << (colour_bits))  # colour bit
    idx = idx | (col << (row_bits + colour_bits))

    # add two to allow for special event bits
    idx = idx + 2

    return idx


def subsample_connection(x_res, y_res, subsamp_factor_x, subsamp_factor_y, weight,
                         coord_map_func):
    # subY_BITS=int(np.ceil(np.log2(y_res/subsamp_factor)))
    connection_list_on = []
    connection_list_off = []

    sx_res = int(x_res) // int(subsamp_factor_x)
    row_bits = 8#int(np.ceil(np.log2(y_res)))
    for j in range(int(y_res)):
        for i in range(int(x_res)):
            si = i // subsamp_factor_x
            sj = j // subsamp_factor_y
            # ON channels
            subsampidx = sj * sx_res + si
            connection_list_on.append((coord_map_func(j, i, 1, row_bits),
                                       subsampidx, weight, 1.))
            # OFF channels only on segment borders
            # if((j+1)%(y_res/subsamp_factor)==0 or (i+1)%(x_res/subsamp_factor)==0 or j==0 or i==0):
            connection_list_off.append((coord_map_func(j, i, 0, row_bits),
                                        subsampidx, weight, 1.))

    return connection_list_on, connection_list_off

def cm_to_fromlist(number_of_nodes, cm):
    i2i_ex = []
    i2i_in = []
    i2h_ex = []
    i2h_in = []
    i2o_ex = []
    i2o_in = []
    h2i_ex = []
    h2i_in = []
    h2h_ex = []
    h2h_in = []
    h2o_ex = []
    h2o_in = []
    o2i_ex = []
    o2i_in = []
    o2h_ex = []
    o2h_in = []
    o2o_ex = []
    o2o_in = []
    hidden_size = number_of_nodes - output_size - input_size
    for i in range(number_of_nodes):
        for j in range(number_of_nodes):
            connect_weight = cm[j][i] * (weight_max / 50.)
            if connect_weight != 0 and not math.isnan(connect_weight):
                if i < input_size:
                    if j < input_size:
                        if connect_weight > 0:
                            i2i_ex.append((i, j, connect_weight, delay))
                        else:
                            i2i_in.append((i, j, connect_weight, delay))
                    elif j < input_size + output_size:
                        if connect_weight > 0:
                            i2o_ex.append((i, j - input_size, connect_weight, delay))
                        else:
                            i2o_in.append((i, j - input_size, connect_weight, delay))
                    elif j < input_size + hidden_size + output_size:
                        if connect_weight > 0:
                            i2h_ex.append((i, j - input_size - output_size, connect_weight, delay))
                        else:
                            i2h_in.append((i, j - input_size - output_size, connect_weight, delay))
                    else:
                        print "shit is broke"
                elif i < input_size + output_size:
                    if j < input_size:
                        if connect_weight > 0:
                            o2i_ex.append((i - input_size, j, connect_weight, delay))
                        else:
                            o2i_in.append((i - input_size, j, connect_weight, delay))
                    elif j < input_size + output_size:
                        if connect_weight > 0:
                            o2o_ex.append((i - input_size, j - input_size, connect_weight, delay))
                        else:
                            o2o_in.append((i - input_size, j - input_size, connect_weight, delay))
                    elif j < input_size + hidden_size + output_size:
                        if connect_weight > 0:
                            o2h_ex.append((i - input_size, j - input_size - output_size, connect_weight, delay))
                        else:
                            o2h_in.append((i - input_size, j - input_size - output_size, connect_weight, delay))
                    else:
                        print "shit is broke"
                elif i < input_size + hidden_size + output_size:
                    if j < input_size:
                        if connect_weight > 0:
                            h2i_ex.append((i - input_size - output_size, j, connect_weight, delay))
                        else:
                            h2i_in.append((i - input_size - output_size, j, connect_weight, delay))
                    elif j < input_size + output_size:
                        if connect_weight > 0:
                            h2o_ex.append((i - input_size - output_size, j - input_size, connect_weight, delay))
                        else:
                            h2o_in.append((i - input_size - output_size, j - input_size, connect_weight, delay))
                    elif j < input_size + hidden_size + output_size:
                        if connect_weight > 0:
                            h2h_ex.append(
                                (i - input_size - output_size, j - input_size - output_size, connect_weight, delay))
                        else:
                            h2h_in.append(
                                (i - input_size - output_size, j - input_size - output_size, connect_weight, delay))
                    else:
                        print "shit is broke"
                else:
                    print "shit is broke"

    return i2i_ex, i2h_ex, i2o_ex, h2i_ex, h2h_ex, h2o_ex, o2i_ex, o2h_ex, o2o_ex, i2i_in, i2h_in, i2o_in, h2i_in, h2h_in, h2o_in, o2i_in, o2h_in, o2o_in

def connect_genes_to_fromlist(number_of_nodes, connections, nodes):
    i2i_ex = []
    i2i_in = []
    i2h_ex = []
    i2h_in = []
    i2o_ex = []
    i2o_in = []
    h2i_ex = []
    h2i_in = []
    h2h_ex = []
    h2h_in = []
    h2o_ex = []
    h2o_in = []
    o2i_ex = []
    o2i_in = []
    o2h_ex = []
    o2h_in = []
    o2o_ex = []
    o2o_in = []

    ex_or_in = []
    i = 0
    for node in nodes:
        ex_or_in.append(nodes[i][1])
        i += 1

    #individual: Tuples of (innov, from, to, weight, enabled)

    hidden_size = number_of_nodes - output_size - input_size

    for c in connections:
        c[4] = bool(c[4])
        if c[4] == True:
            c[1] = int(c[1])
            c[2] = int(c[2])
            c[3] = float(c[3])
            connect_weight = c[3]
            if c[1] < input_size:
                if c[2] < input_size:
                    if ex_or_in[c[1]] == 'excitatory':
                        i2i_ex.append((c[1], c[2], connect_weight, delay))
                    else:
                        i2i_in.append((c[1], c[2], connect_weight, delay))
                elif c[2] < input_size + output_size:
                    if ex_or_in[c[1]] == 'excitatory':
                        i2o_ex.append((c[1], c[2]-input_size, connect_weight, delay))
                    else:
                        i2o_in.append((c[1], c[2]-input_size, connect_weight, delay))
                elif c[2] < input_size + hidden_size + output_size:
                    if ex_or_in[c[1]] == 'excitatory':
                        i2h_ex.append((c[1], c[2]-input_size-output_size, connect_weight, delay))
                    else:
                        i2h_in.append((c[1], c[2]-input_size-output_size, connect_weight, delay))
                else:
                    print "shit broke"
            elif c[1] < input_size + output_size:
                if c[2] < input_size:
                    if ex_or_in[c[1]] == 'excitatory':
                        o2i_ex.append((c[1]-input_size, c[2], connect_weight, delay))
                    else:
                        o2i_in.append((c[1]-input_size, c[2], connect_weight, delay))
                elif c[2] < input_size + output_size:
                    if ex_or_in[c[1]] == 'excitatory':
                        o2o_ex.append((c[1]-input_size, c[2]-input_size, connect_weight, delay))
                    else:
                        o2o_in.append((c[1]-input_size, c[2]-input_size, connect_weight, delay))
                elif c[2] < input_size + hidden_size + output_size:
                    if ex_or_in[c[1]] == 'excitatory':
                        o2h_ex.append((c[1]-input_size, c[2]-input_size-output_size, connect_weight, delay))
                    else:
                        o2h_in.append((c[1]-input_size, c[2]-input_size-output_size, connect_weight, delay))
                else:
                    print "shit broke"
            elif c[1] < input_size + hidden_size + output_size:
                if c[2] < input_size:
                    if ex_or_in[c[1]] == 'excitatory':
                        h2i_ex.append((c[1]-input_size-output_size, c[2], connect_weight, delay))
                    else:
                        h2i_in.append((c[1]-input_size-output_size, c[2], connect_weight, delay))
                elif c[2] < input_size + output_size:
                    if ex_or_in[c[1]] == 'excitatory':
                        h2o_ex.append((c[1]-input_size-output_size, c[2]-input_size, connect_weight, delay))
                    else:
                        h2o_in.append((c[1]-input_size-output_size, c[2]-input_size, connect_weight, delay))
                elif c[2] < input_size + hidden_size + output_size:
                    if ex_or_in[c[1]] == 'excitatory':
                        h2h_ex.append((c[1]-input_size-output_size, c[2]-input_size-output_size, connect_weight, delay))
                    else:
                        h2h_in.append((c[1]-input_size-output_size, c[2]-input_size-output_size, connect_weight, delay))
                else:
                    print "shit broke"
            else:
                print "shit broke"
    return i2i_ex, i2h_ex, i2o_ex, h2i_ex, h2h_ex, h2o_ex, o2i_ex, o2h_ex, o2o_ex, i2i_in, i2h_in, i2o_in, h2i_in, h2h_in, h2o_in, o2i_in, o2h_in, o2o_in

def test_agent(pop):
    #test the whole population and return scores

    #Acquire all connection matrices and node types
    # networks = []
    # for individual in pop:
    #     networks.append(NeuralNetwork(individual))

    agent = read_agent()

    number_of_nodes = len(agent[1])
    hidden_size = number_of_nodes - output_size - input_size

    # breakout_pops = []
    # bandit_pops = []
    # hidden_node_pops = []
    # hidden_count = 0

    # Setup pyNN simulation
    p.setup(timestep=1.0)
    p.set_number_of_neurons_per_core(p.IF_cond_exp, 100)

    number_of_nodes = len(agent[1])
    hidden_size = number_of_nodes - output_size - input_size
    [i2i_ex, i2h_ex, i2o_ex, h2i_ex, h2h_ex, h2o_ex, o2i_ex, o2h_ex, o2o_ex, i2i_in, i2h_in, i2o_in, h2i_in, h2h_in,
     h2o_in, o2i_in, o2h_in, o2o_in] = connect_genes_to_fromlist(number_of_nodes, agent[0], agent[1])

    random_seed = []
    for j in range(4):
        random_seed.append(np.random.randint(0xffff))
    band = Bandit(arms, reward_based=reward_based, reward_delay=duration_of_trial, rand_seed=random_seed,
                  label="bandit")
    bandit_pop = p.Population(band.neurons(), band, label="bandit")

    # Create output population and remaining population
    output_pop = p.Population(output_size, p.IF_cond_exp(), label="output_pop")
    p.Projection(output_pop, bandit_pop, p.OneToOneConnector())
    if noise_rate != 0:
        output_noise = p.Population(output_size, p.SpikeSourcePoisson(rate=noise_rate), label="output noise")
        p.Projection(output_noise, output_pop, p.OneToOneConnector(),
                     p.StaticSynapse(weight=noise_weight), receptor_type='excitatory')
    # print "after creating output"
    # tracker.print_diff()

    if hidden_size != 0:
        hidden_node_pop = p.Population(hidden_size, p.IF_cond_exp(), label="hidden_pop")
        if noise_rate != 0:
            hidden_noise = p.Population(hidden_size, p.SpikeSourcePoisson(rate=noise_rate), label="hidden noise")
            p.Projection(hidden_noise, hidden_node_pop, p.OneToOneConnector(),
                         p.StaticSynapse(weight=noise_weight), receptor_type='excitatory')
        hidden_node_pop.record('spikes')
    # print "after creating hidden"
    # tracker.print_diff()
    # bandit_pop.record('spikes')
    output_pop.record('spikes')
    #     hidden_node_pop.record()
    # bandit_pop.record()
    # output_pop.record()

    if len(i2i_ex) != 0:
        connection = p.FromListConnector(i2i_ex)
        p.Projection(bandit_pop, bandit_pop, connection,
                     receptor_type='excitatory')
    if len(i2h_ex) != 0:
        connection = p.FromListConnector(i2h_ex)
        p.Projection(bandit_pop, hidden_node_pop, connection,
                     receptor_type='excitatory')
    if len(i2o_ex) != 0:
        connection = p.FromListConnector(i2o_ex)
        p.Projection(bandit_pop, output_pop, connection,
                     receptor_type='excitatory')
    if len(h2i_ex) != 0:
        p.Projection(hidden_node_pop, bandit_pop, p.FromListConnector(h2i_ex),
                     receptor_type='excitatory')
    if len(h2h_ex) != 0:
        p.Projection(hidden_node_pop, hidden_node_pop, p.FromListConnector(h2h_ex),
                     receptor_type='excitatory')
    if len(h2o_ex) != 0:
        p.Projection(hidden_node_pop, output_pop, p.FromListConnector(h2o_ex),
                     receptor_type='excitatory')
    if len(o2i_ex) != 0:
        p.Projection(output_pop, bandit_pop, p.FromListConnector(o2i_ex),
                     receptor_type='excitatory')
    if len(o2h_ex) != 0:
        p.Projection(output_pop, hidden_node_pop, p.FromListConnector(o2h_ex),
                     receptor_type='excitatory')
    if len(o2o_ex) != 0:
        p.Projection(output_pop, output_pop, p.FromListConnector(o2o_ex),
                     receptor_type='excitatory')
    if len(i2i_in) != 0:
        p.Projection(bandit_pop, bandit_pop, p.FromListConnector(i2i_in),
                     receptor_type='inhibitory')
    if len(i2h_in) != 0:
        p.Projection(bandit_pop, hidden_node_pop, p.FromListConnector(i2h_in),
                     receptor_type='inhibitory')
    if len(i2o_in) != 0:
        p.Projection(bandit_pop, output_pop, p.FromListConnector(i2o_in),
                     receptor_type='inhibitory')
    if len(h2i_in) != 0:
        p.Projection(hidden_node_pop, bandit_pop, p.FromListConnector(h2i_in),
                     receptor_type='inhibitory')
    if len(h2h_in) != 0:
        p.Projection(hidden_node_pop, hidden_node_pop, p.FromListConnector(h2h_in),
                     receptor_type='inhibitory')
    if len(h2o_in) != 0:
        p.Projection(hidden_node_pop, output_pop, p.FromListConnector(h2o_in),
                     receptor_type='inhibitory')
    if len(o2i_in) != 0:
        p.Projection(output_pop, bandit_pop, p.FromListConnector(o2i_in),
                     receptor_type='inhibitory')
    if len(o2h_in) != 0:
        p.Projection(output_pop, hidden_node_pop, p.FromListConnector(o2h_in),
                     receptor_type='inhibitory')
    if len(o2o_in) != 0:
        p.Projection(output_pop, output_pop, p.FromListConnector(o2o_in),
                     receptor_type='inhibitory')



    print "reached here 1"

    simulator = get_simulator()

    p.run(runtime)

    print "reached here 2"

    scores = get_scores(bandit_pop=bandit_pop, simulator=simulator)

    spikes_h = hidden_node_pop.get_data('spikes').segments[0].spiketrains
    hidden_spikes = 0
    for neuron in spikes_h:
        for spike in neuron:
            # print spike
            hidden_spikes += 1

    spikes_o = output_pop.get_data('spikes').segments[0].spiketrains
    out_spikes = 0
    for neuron in spikes_o:
        for spike in neuron:
            # print spike
            out_spikes += 1
    Figure(
        Panel(spikes_h, xlabel="Time (ms)", ylabel="nID", xticks=True),
        Panel(spikes_o, xlabel="Time (ms)", ylabel="nID", xticks=True)#,
        # Panel(spikes_t, xlabel="Time (ms)", ylabel="nID", xticks=True)
    )
    plt.show()

    # pylab.figure()
    # spikes_on = bandit_pop.getSpikes()
    # ax = pylab.subplot(1, 3, 1)#4, 1)
    # pylab.plot([i[1] for i in spikes_on], [i[0] for i in spikes_on], "r.")
    # pylab.xlabel("Time (ms)")
    # pylab.ylabel("neuron ID")
    # pylab.axis([0, runtime, -1, receive_pop_size + 1])
    # # pylab.show()
    # pylab.figure()
    # if hidden_size != 0:
    #     spikes_on = hidden_node_pop.getSpikes()
    #     ax = pylab.subplot(1, 3, 2)#4, 1)
    #     pylab.plot([i[1] for i in spikes_on], [i[0] for i in spikes_on], "r.")
    #     pylab.xlabel("Time (ms)")
    #     pylab.ylabel("neuron ID")
    #     pylab.axis([0, runtime, -1, receive_pop_size + 1])
    # # pylab.show()
    # # pylab.figure()
    # spikes_on = output_pop.getSpikes()
    # ax = pylab.subplot(1, 3, 3)#4, 1)
    # pylab.plot([i[1] for i in spikes_on], [i[0] for i in spikes_on], "r.")
    # pylab.xlabel("Time (ms)")
    # pylab.ylabel("neuron ID")
    # pylab.axis([0, runtime, -1, output_size + 1])
    # pylab.show()

    j = 0
    for score in scores:
        print j, score
        j += 1

    # End simulation
    p.end()


runtime = 31000

reward_based = 1
duration_of_trial = 200

noise_rate = 0
noise_weight = 0.01
delay = 2

# arms = [0.8, 0.2]
# arms = [0.2, 0.8]
arms = [0.1, 0.9]
# arms = [0.9, 0.1]
output_size = len(arms)
input_size = 2

# file = 'NEAT bandit champion 268 - a2 -e2 - cTrue - sTrue - n0-0.01 - gcap - r0 f=0.46.csv'
# file = 'NEAT bandit champion score 0 - a2 -e1 - cTrue - sTrue - n0-0.01 - gboth - r0 f=0.98.csv'
# file = 'f=0.65 - a2 -e2 - cTrue - sTrue - n0-0.01 - gcap - r1.csv'
# file = 'NEAT bandit champion score 58:1.689 - a2 -e2 - cTrue - sTrue - n50-0.01 - gcap - r1.csv'
file = 'NEAT bandit champion score 122:0.889 - a2 -e2 - cTrue - sTrue - n0-0.01 - gweighted - r1.csv'
hyper = False

# if hyper == True:
#     weight_max = 1
#
#     geno_kwds = dict(feedforward=True,
#                      inputs=4,
#                      outputs=2,
#                      weight_range=(-50.0, 50.0),
#                      prob_add_conn=0.1,
#                      prob_add_node=0.03,
#                      bias_as_node=False,
#                      types=['sin', 'bound', 'linear', 'gauss', 'sigmoid', 'abs'])
#
#     geno = lambda: NEATGenotype(**geno_kwds)
#
#
#     # Configure substrate
#     substrate = Substrate()
#     print "adding input"
#     substrate.add_nodes([(r, theta) for r in np.linspace(-1,1,int(x_res/x_factor))
#                               for theta in np.linspace(-1, 1, int(y_res/y_factor))], 'input')
#     print "adding output"
#     substrate.add_nodes([(r, theta) for r in np.linspace(0,0,1)
#                               for theta in np.linspace(-1, 1, 2)], 'output')
#     print "adding hidden"
#     substrate.add_nodes([(r, theta) for r in np.linspace(-1,1,int(x_res/x_factor))
#                               for theta in np.linspace(-1, 1, int(y_res/y_factor))], 'hidden')
#
#     print "adding connections"
#     substrate.add_connections('input', 'hidden', -1)
#     substrate.add_connections('hidden', 'output',-2)
#
#     developer = HyperNEATDeveloper(substrate=substrate,
#                                    add_deltas=False,
#                                    sandwich=False,
#                                    feedforward=False,
#                                    node_type=(('excitatory'), ('inhibitory')))
#
#     pop = NEATPopulation(geno, popsize=1, target_species=8)
#
#
#     results = pop.epoch(generations=1,
#                         evaluator=test_agent,
#                         solution=None
#                         )

test_agent(0)