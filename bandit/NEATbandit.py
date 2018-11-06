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

from peas.methods.neat import NEATPopulation, NEATGenotype
from peas.networks.rnn import NeuralNetwork

def get_scores(bandit_pop,simulator):
    b_vertex = bandit_pop._vertex
    scores = b_vertex.get_data(
        'score', simulator.no_machine_time_steps, simulator.placements,
        simulator.graph_mapper, simulator.buffer_manager, simulator.machine_time_step)

    return scores.tolist()

def row_col_to_input_bandit(row, col, is_on_input, row_bits, event_bits=1, colour_bits=1, row_start=0):
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
    i2i = []
    i2h = []
    i2o = []
    h2i = []
    h2h = []
    h2o = []
    o2i = []
    o2h = []
    o2o = []
    hidden_size = number_of_nodes - output_size - input_size
    for i in range(number_of_nodes):
        for j in range(number_of_nodes):
            connect_weight = cm[j][i] * (weight_max / weight_scale)
            if connect_weight != 0 and not math.isnan(connect_weight):
                if i < input_size:
                    if j < input_size:
                        i2i.append((i, j, connect_weight, delay))
                    elif j < input_size + output_size:
                        i2o.append((i, j-input_size, connect_weight, delay))
                    elif j < input_size + hidden_size + output_size:
                        i2h.append((i, j-input_size-output_size, connect_weight, delay))
                    else:
                        print "shit is broke"
                elif i < input_size + output_size:
                    if j < input_size:
                        o2i.append((i-input_size, j, connect_weight, delay))
                    elif j < input_size + output_size:
                        o2o.append((i-input_size, j-input_size, connect_weight, delay))
                    elif j < input_size + hidden_size + output_size:
                        o2h.append((i-input_size, j-input_size-output_size, connect_weight, delay))
                    else:
                        print "shit is broke"
                elif i < input_size + hidden_size + output_size:
                    if j < input_size:
                        h2i.append((i-input_size-output_size, j, connect_weight, delay))
                    elif j < input_size + output_size:
                        h2o.append((i-input_size-output_size, j-input_size, connect_weight, delay))
                    elif j < input_size + hidden_size + output_size:
                        h2h.append((i-input_size-output_size, j-input_size-output_size, connect_weight, delay))
                    else:
                        print "shit is broke"
                else:
                    print "shit is broke"

    return i2i, i2h, i2o, h2i, h2h, h2o, o2i, o2h, o2o

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

    for conn in connections:
        c = connections[conn]
        connect_weight = c[3] * (weight_max / weight_scale)
        if c[4] == True:
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

def test_pop(pop, tracker):#, noise_rate=50, noise_weight=1):
    #test the whole population and return scores
    global all_fails
    print "start"
    gen_stats(pop)
    save_champion(pop)
    # tracker.print_diff()

    #Acquire all connection matrices and node types

    print len(pop)
    # tracker.print_diff()
    #create the SpiNN nets
    scores = []
    spike_counts = []
    max_arms = []
    flagged_agents = []
    for trial in range(number_of_epochs):
        try_except = 0
        while try_except < try_attempts:
            print "\narms:", number_of_arms, "- epochs:", number_of_epochs, "- complimentary:", complimentary, \
                "- shared:", shared_probabilities, "- fails:", all_fails, "- noise rate/weight:", noise_rate, "/", \
                noise_weight, "- grooming:", grooming, "- reward:", reward_based, "\n"
            time.sleep(5)
            bandit_pops = []
            # receive_on_pops = []
            hidden_node_pops = []
            hidden_count = -1
            hidden_marker = []
            output_pops = []
            # Setup pyNN simulation
            p.setup(timestep=1.0)
            p.set_number_of_neurons_per_core(p.IF_cond_exp, 100)
            for i in range(len(pop)):
                if i not in flagged_agents:
                    number_of_nodes = len(pop[i].node_genes)
                    hidden_size = number_of_nodes - output_size - input_size

                    [i2i_ex, i2h_ex, i2o_ex, h2i_ex, h2h_ex, h2o_ex, o2i_ex, o2h_ex, o2o_ex, i2i_in, i2h_in, i2o_in, h2i_in, h2h_in, h2o_in, o2i_in, o2h_in, o2o_in] = \
                        connect_genes_to_fromlist(number_of_nodes, pop[i].conn_genes, pop[i].node_genes)
                    # print "after creating connections"
                    # tracker.print_diff()
                    # [i2i, i2h, i2o, h2i, h2h, h2o, o2i, o2h, o2o] = cm_to_fromlist(number_of_nodes, networks[i].cm)
                    if (i == 0 or shared_probabilities == False) and try_except == 0:
                        arms = []
                        if fixed_arms:
                            arms = fixed_arms[trial]
                        else:
                            total = 1
                            for j in range(number_of_arms-1):
                                arms.append(random.uniform(0, total))
                                total -= arms[j]
                            arms.append(total)
                            if trial % number_of_arms == 0:
                                arms.sort(reverse=True)
                            elif trial % number_of_arms == 1:
                                arms.sort()
                            else:
                                np.random.shuffle(arms)
                        max_arms.append((float(format(max(arms), '.5f')), arms.index(max(arms))))
                    # Create bandit population
                    random_seed = []
                    for j in range(4):
                        random_seed.append(np.random.randint(0xffff))
                    band = Bandit(arms, reward_based=reward_based, reward_delay=duration_of_trial, rand_seed=random_seed, label="bandit {}".format(i))
                    bandit_pops.append(p.Population(band.neurons(), band, label="bandit {}".format(i)))
                    # print "after creating bandit"
                    # tracker.print_diff()

                    # Create input population and connect break out to it
                    # receive_on_pops.append(p.Population(receive_pop_size, p.IF_cond_exp(), label="receive_pop {}".format(i)))
                    # print "after creating receive pop"
                    # tracker.print_diff()
                    # p.Projection(bandit_pops[i], receive_on_pops[i], p.OneToOneConnector(), p.StaticSynapse(weight=0.1))
                    # print "after creating receive projection"
                    # tracker.print_diff()

                    # Create output population and remaining population
                    output_pops.append(p.Population(output_size, p.IF_cond_exp(), label="output_pop {}".format(i)))
                    p.Projection(output_pops[i], bandit_pops[i], p.OneToOneConnector())
                    if noise_rate != 0:
                        output_noise = p.Population(output_size, p.SpikeSourcePoisson(rate=noise_rate), label="output noise")
                        p.Projection(output_noise, output_pops[i], p.OneToOneConnector(),
                                     p.StaticSynapse(weight=noise_weight), receptor_type='excitatory')
                    # print "after creating output"
                    # tracker.print_diff()

                    if hidden_size != 0:
                        hidden_node_pops.append(p.Population(hidden_size, p.IF_cond_exp(), label="hidden_pop {}".format(i)))
                        hidden_count += 1
                        hidden_marker.append(i)
                        if noise_rate != 0:
                            hidden_noise = p.Population(hidden_size, p.SpikeSourcePoisson(rate=noise_rate), label="hidden noise")
                            p.Projection(hidden_noise, hidden_node_pops[hidden_count], p.OneToOneConnector(),
                                         p.StaticSynapse(weight=noise_weight), receptor_type='excitatory')
                        hidden_node_pops[hidden_count].record('spikes')
                    # print "after creating hidden"
                    # tracker.print_diff()
                    # receive_on_pops[i].record('spikes')
                    output_pops[i].record('spikes')

                    # Create the remaining nodes from the connection matrix and add them up
                    if len(i2i_ex) != 0:
                        connection = p.FromListConnector(i2i_ex)
                        p.Projection(bandit_pops[i], bandit_pops[i], connection,
                                     receptor_type='excitatory')
                    if len(i2h_ex) != 0:
                        connection = p.FromListConnector(i2h_ex)
                        p.Projection(bandit_pops[i], hidden_node_pops[hidden_count], connection,
                                     receptor_type='excitatory')
                    if len(i2o_ex) != 0:
                        connection = p.FromListConnector(i2o_ex)
                        p.Projection(bandit_pops[i], output_pops[i], connection,
                                     receptor_type='excitatory')
                    if len(h2i_ex) != 0:
                        p.Projection(hidden_node_pops[hidden_count], bandit_pops[i], p.FromListConnector(h2i_ex),
                                     receptor_type='excitatory')
                    if len(h2h_ex) != 0:
                        p.Projection(hidden_node_pops[hidden_count], hidden_node_pops[hidden_count], p.FromListConnector(h2h_ex),
                                     receptor_type='excitatory')
                    if len(h2o_ex) != 0:
                        p.Projection(hidden_node_pops[hidden_count], output_pops[i], p.FromListConnector(h2o_ex),
                                     receptor_type='excitatory')
                    if len(o2i_ex) != 0:
                        p.Projection(output_pops[i], bandit_pops[i], p.FromListConnector(o2i_ex),
                                     receptor_type='excitatory')
                    if len(o2h_ex) != 0:
                        p.Projection(output_pops[i], hidden_node_pops[hidden_count], p.FromListConnector(o2h_ex),
                                     receptor_type='excitatory')
                    if len(o2o_ex) != 0:
                        p.Projection(output_pops[i], output_pops[i], p.FromListConnector(o2o_ex),
                                     receptor_type='excitatory')
                    if len(i2i_in) != 0:
                        p.Projection(bandit_pops[i], bandit_pops[i], p.FromListConnector(i2i_in),
                                     receptor_type='inhibitory')
                    if len(i2h_in) != 0:
                        p.Projection(bandit_pops[i], hidden_node_pops[hidden_count], p.FromListConnector(i2h_in),
                                     receptor_type='inhibitory')
                    if len(i2o_in) != 0:
                        p.Projection(bandit_pops[i], output_pops[i], p.FromListConnector(i2o_in),
                                     receptor_type='inhibitory')
                    if len(h2i_in) != 0:
                        p.Projection(hidden_node_pops[hidden_count], bandit_pops[i], p.FromListConnector(h2i_in),
                                     receptor_type='inhibitory')
                    if len(h2h_in) != 0:
                        p.Projection(hidden_node_pops[hidden_count], hidden_node_pops[hidden_count], p.FromListConnector(h2h_in),
                                     receptor_type='inhibitory')
                    if len(h2o_in) != 0:
                        p.Projection(hidden_node_pops[hidden_count], output_pops[i], p.FromListConnector(h2o_in),
                                     receptor_type='inhibitory')
                    if len(o2i_in) != 0:
                        p.Projection(output_pops[i], bandit_pops[i], p.FromListConnector(o2i_in),
                                     receptor_type='inhibitory')
                    if len(o2h_in) != 0:
                        p.Projection(output_pops[i], hidden_node_pops[hidden_count], p.FromListConnector(o2h_in),
                                     receptor_type='inhibitory')
                    if len(o2o_in) != 0:
                        p.Projection(output_pops[i], output_pops[i], p.FromListConnector(o2o_in),
                                     receptor_type='inhibitory')
                    # print "after creating projections"
                    # tracker.print_diff()



            print "reached here 1"
            # tracker.print_diff()

            simulator = get_simulator()
            try:
                p.run(runtime)
                try_except = try_attempts
                break
            except:
                traceback.print_exc()
                all_fails += 1
                try_except += 1
                print "\nfailed to run on attempt", try_except, ". total fails:", all_fails, "\n"
                # new_scores = []
                # for i in range(len(pop)):
                #     if i not in flagged_agents:
                #         new_scores.append(get_scores(bandit_pop=bandit_pops[i], simulator=simulator))
                #         if new_scores[i] > spike_cap:
                #             flagged_agents.append(i)

                # p.end()


        hidden_count = 0
        new_spike_counts = []
        out_spike_count = [0 for i in range(len(pop))]
        hid_spike_count = [0 for i in range(len(pop))]
        for i in range(len(pop)):
            if i not in flagged_agents:
                spikes = output_pops[i].get_data('spikes').segments[0].spiketrains
                for neuron in spikes:
                    for spike in neuron:
                        out_spike_count[i] += 1
                if i in hidden_marker:
                    spikes = hidden_node_pops[hidden_count].get_data('spikes').segments[0].spiketrains
                    hidden_count += 1
                    for neuron in spikes:
                        for spike in neuron:
                            hid_spike_count[i] += 1
                new_spike_counts.append(hid_spike_count[i] + out_spike_count[i])
            else:
                new_spike_counts.append(10000000)
                out_spike_count.append(10000000)
                hid_spike_count.append(10000000)
        spike_counts.append(new_spike_counts)

        print "reached here 2"
        new_scores = []
        for i in range(len(pop)):
            if i not in flagged_agents:
                new_scores.append(get_scores(bandit_pop=bandit_pops[i], simulator=simulator))
            else:
                new_scores.append(-10000000)
        scores.append(new_scores)
        #     if i == 0:
        #         pylab.figure()
        #     spikes_on = output_pops[i].getSpikes()
        #     ax = pylab.subplot(1, len(pop), i+1)#4, 1)
        #     pylab.plot([i[1] for i in spikes_on], [i[0] for i in spikes_on], "r.")
        #     pylab.xlabel("Time (ms)")
        #     pylab.ylabel("neuron ID")
        #     pylab.axis([0, runtime, -1, output_size + 1])
        # pylab.show()
        # # pylab.figure()
        # # for i in range(hidden_count):
        # #     spikes_on = hidden_node_pops[i].getSpikes()
        # #     ax = pylab.subplot(1, len(pop), i+1)#4, 1)
        # #     pylab.plot([i[1] for i in spikes_on], [i[0] for i in spikes_on], "r.")
        # #     pylab.xlabel("Time (ms)")
        # #     pylab.ylabel("neuron ID")
        # #     pylab.axis([0, runtime, -1, receive_pop_size + 1])
        # # pylab.show()
        # pylab.figure()
        # for i in range(len(pop)):
        #     spikes_on = receive_on_pops[i].getSpikes()
        #     ax = pylab.subplot(1, len(pop), i+1)#4, 1)
        #     pylab.plot([i[1] for i in spikes_on], [i[0] for i in spikes_on], "r.")
        #     pylab.xlabel("Time (ms)")
        #     pylab.ylabel("neuron ID")
        #     pylab.axis([0, runtime, -1, receive_pop_size + 1])
        # pylab.show()

        j = 0
        for score in scores[trial]:
            print j, "s:", new_spike_counts[j], "o:", out_spike_count[j], "h:", hid_spike_count[j], score
            j += 1
        print "\narms:", number_of_arms, "- epochs:", number_of_epochs, "- complimentary:", complimentary, \
            "- shared:", shared_probabilities, "- fails:", all_fails, "- noise rate/weight:", noise_rate, "/", \
            noise_weight, "- grooming:", grooming, "- reward:", reward_based, "\n"
        if shared_probabilities == True:
            print "probabilities = ", arms
        # End simulation
        p.end()
        print "\nfinished epoch:", trial+1, "/", number_of_epochs, "\n"
    pop_scores = []
    pop_spikes = []
    min_score = 10000000
    for i in range(len(pop)):
        temp_score = 0
        temp_spike = 0
        for j in range(number_of_epochs):
            # print np.double(scores[i+(len(pop)*j)][len(scores[i+(len(pop)*j)]) - 1][0]), (np.double(scores[i+(len(pop)*j)][len(scores[i+(len(pop)*j)]) - 1][0]) / number_of_trials) / max_arms[j]
            if reward_based == 0:
                temp_score += (np.double(scores[j][i][len(scores[j][i]) - 1][0]) / number_of_trials)
            else:
                temp_score += (np.double(scores[j][i][len(scores[j][i]) - 1][0]) / number_of_trials) / max_arms[j][0]
            temp_spike += spike_counts[j][i]
        if temp_score < min_score:
            min_score = temp_score
        pop_scores.append([temp_score, i])
        pop_spikes.append([temp_spike, i])
    pop_scores.sort()
    pop_spikes.sort(reverse=True)
    combined_fitness = [0 for i in range(len(pop))]
    # failed_score = 0
    failed_spikes = 0
    for i in range(len(pop)):
        if grooming == 'both':
            if pop_scores[i][0] != min_score:
                # failed_score += 1
                combined_fitness[pop_scores[i][1]] += i#failed_score
            if pop_spikes[i][0] < spike_cap:
                if pop_spikes[i][0] > number_of_trials:
                    failed_spikes += 1
                combined_fitness[pop_spikes[i][1]] += failed_spikes
        elif grooming == 'cap':
            if pop_spikes[i][0] > spike_cap:
                combined_fitness[pop_spikes[i][1]] -= 10000
            combined_fitness[pop_scores[i][1]] += pop_scores[i][0]
        elif grooming == 'strict':
            if pop_spikes[i][0] > spike_cap:
                combined_fitness[pop_spikes[i][1]] -= 10000
                combined_fitness[pop_scores[i][1]] += i
            if pop_scores[i][0] == min_score:
                combined_fitness[pop_scores[i][1]] -=10000
                combined_fitness[pop_spikes[i][1]] += i
            else:
                combined_fitness[pop_scores[i][1]] += i
                combined_fitness[pop_spikes[i][1]] += i
        elif grooming == 'break it':
            combined_fitness[pop_scores[i][1]] += i
            combined_fitness[pop_spikes[i][1]] -= i*i
        else:
            combined_fitness[pop_scores[i][1]] += i
            combined_fitness[pop_spikes[i][1]] += i
    pop_scores.sort(key=lambda x: x[1])
    pop_spikes.sort(key=lambda x: x[1])
    for i in range(len(pop)):
        pop[i].stats = {'fitness': combined_fitness[i], 'score': float(format(pop_scores[i][0], '.3f')), 'spikes': -pop_spikes[i][0]}
    print "finished all epochs"
    print "max probabilities were ", max_arms
    print "floor score is ", format(min_score, '.5f')
    # for i in range(len(pop)):
    #     print i, "|", pop[i].stats['fitness']
    # print "floor score is ", format(min_score, '.5f')
    # print "finished all epochs"
    # print "max probabilities were ", max_arms
    # gc.DEBUG_STATS
    # gc.DEBUG_COLLECTABLE

def gen_stats(list_pop):
    # pop._gather_stats(list_pop)
    for stat in NEAT_pop.stats:
        print "{}: {}".format(stat, NEAT_pop.stats[stat])

def save_champion(agent_pop):
    iteration = len(NEAT_pop.champions) - 1
    if iteration >= 0:
        best_score = -1000000
        for i in range(len(agent_pop)):
            try:
                if agent_pop[i].stats['score'] > best_score:
                    best_score = agent_pop[i].stats['score']
                    best_agent = i
            except:
                None
        with open('NEAT bandit champion score {}:{} - a{} -e{} - c{} - s{} - n{}-{} - g{} - r{}.csv'.format(
                iteration, best_score, number_of_arms, number_of_epochs, complimentary, shared_probabilities,
                noise_rate, noise_weight, grooming, reward_based), 'w') as file:
            writer = csv.writer(file, delimiter=',', lineterminator='\n')
            for i in agent_pop[best_agent].conn_genes:
                writer.writerow(agent_pop[best_agent].conn_genes[i])
            for i in agent_pop[best_agent].node_genes:
                writer.writerow(i)
            for i in agent_pop[best_agent].stats:
                writer.writerow(["fitness {}".format(i), agent_pop[best_agent].stats[i]])
        with open('NEAT bandit champions score a{} - c{} -e{} - s{} - n{}-{} - g{} - r{}.csv'.format(
                number_of_arms, complimentary, number_of_epochs, shared_probabilities, noise_rate, noise_weight,
                grooming, reward_based), 'a') as file:
            writer = csv.writer(file, delimiter=',', lineterminator='\n')
            writer.writerow([iteration, best_score])
        with open('NEAT bandit champion {} - a{} -e{} - c{} - s{} - n{}-{} - g{} - r{}.csv'.format(
                iteration, number_of_arms, number_of_epochs, complimentary, shared_probabilities, noise_rate,
                noise_weight, grooming, reward_based), 'w') as file:
            writer = csv.writer(file, delimiter=',', lineterminator='\n')
            for i in NEAT_pop.champions[iteration].conn_genes:
                writer.writerow(NEAT_pop.champions[iteration].conn_genes[i])
            for i in NEAT_pop.champions[iteration].node_genes:
                writer.writerow(i)
            for i in NEAT_pop.champions[iteration].stats:
                writer.writerow(["fitness {}".format(i), NEAT_pop.champions[iteration].stats[i]])
            # writer.writerow("\n")
        with open('NEAT bandit champions a{} - c{} -e{} - s{} - n{}-{} - g{} - r{}.csv'.format(
                number_of_arms, complimentary, number_of_epochs, shared_probabilities, noise_rate, noise_weight,
                grooming, reward_based), 'a') as file:
            writer = csv.writer(file, delimiter=',', lineterminator='\n')
            for i in NEAT_pop.champions[iteration].conn_genes:
                writer.writerow(NEAT_pop.champions[iteration].conn_genes[i])
            for i in NEAT_pop.champions[iteration].node_genes:
                writer.writerow(i)
            for i in NEAT_pop.champions[iteration].stats:
                writer.writerow(["fitness {}".format(i), NEAT_pop.champions[iteration].stats[i]])

# gc.set_debug(gc.DEBUG_LEAK)


number_of_arms = 2
number_of_epochs = 2
fixed_arms = [[0.8, 0.2], [0.2, 0.8]]
complimentary = True
shared_probabilities = True
grooming = 'cap'
reward_based = 0
spike_cap = 10000
noise_rate = 0
noise_weight = 0.01

# UDP port to read spikes from
UDP_PORT1 = 17887
UDP_PORT2 = UDP_PORT1 + 1

number_of_trials = 100
duration_of_trial = 200
runtime = number_of_trials * duration_of_trial
try_attempts = 5
all_fails = 0

weight_max = 1.0
weight_scale = 1.0
delay = 2

weight = 0.1

# p.setup(timestep=1.0)
# p.set_number_of_neurons_per_core(p.IF_cond_exp, 100)
# bandit_connections = p.FromListConnector(Connections_on)
# p.end()

#current rounds off each number to create a super rounded off int
input_size = 2
output_size = number_of_arms

genotype = lambda: NEATGenotype(inputs=input_size,
                                outputs=output_size,
                                prob_add_node=0.3,
                                prob_add_conn=0.5,
                                weight_range=(0, 0.1),
                                initial_weight_stdev=0.05,
                                stdev_mutate_weight=0.02,
                                types=['excitatory', 'inhibitory'],
                                feedforward=False)

# Create a population
NEAT_pop = NEATPopulation(genotype, popsize=200,
                     stagnation_age=30,
                     old_age=75,
                          target_species=5)

# Run the evolution, tell it to use the task as an evaluator
print "beginning epoch"
tracker = SummaryTracker()
NEAT_pop.epoch(tracker=tracker, generations=2000, evaluator=test_pop, solution=None, SpiNNaker=True)
save_champion()


