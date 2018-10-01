//
//  bkout.c
//  BreakOut
//
//  Created by Steve Furber on 26/08/2016.
//  Copyright © 2016 Steve Furber. All rights reserved.
//
// Standard includes
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// Spin 1 API includes
#include <spin1_api.h>

// Common includes
#include <debug.h>

// Front end common includes
#include <data_specification.h>
#include <simulation.h>

#include <recording.h>

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------

// Frame delay (ms)
//#define reward_delay 200 //14//20

//----------------------------------------------------------------------------
// Enumerations
//----------------------------------------------------------------------------
typedef enum
{
  REGION_SYSTEM,
  REGION_BREAKOUT,
  REGION_RECORDING,
  REGION_ARMS,
} region_t;

typedef enum
{
  SPECIAL_EVENT_REWARD,
  SPECIAL_EVENT_NO_REWARD,
  SPECIAL_EVENT_MAX,
} special_event_t;

typedef enum
{
  KEY_ARM_0  = 0x0,
  KEY_ARM_1  = 0x1,
  KEY_ARM_2  = 0x2,
  KEY_ARM_3  = 0x3,
  KEY_ARM_4  = 0x4,
  KEY_ARM_5  = 0x5,
  KEY_ARM_6  = 0x6,
  KEY_ARM_7  = 0x7,
} arm_key_t;

//----------------------------------------------------------------------------
// Globals
//----------------------------------------------------------------------------

static uint32_t _time;

//! Should simulation run for ever? 0 if not
static uint32_t infinite_run;

const int max_number_of_arms = 8;

uint32_t *arm_probabilities;

int number_of_arms;

int arm_choices[8] = {0};

int32_t current_score = 0;

uint32_t reward_delay;

//! How many ticks until next frame
static uint32_t tick_in_frame = 0;

//! The upper bits of the key value that model should transmit with
static uint32_t key;

//! the number of timer ticks that this model should run for before exiting.
uint32_t simulation_ticks = 0;
uint32_t score_change_count=0;

//----------------------------------------------------------------------------
// Inline functions
//----------------------------------------------------------------------------
static inline void add_reward()
{
  spin1_send_mc_packet(key | (SPECIAL_EVENT_REWARD), 0, NO_PAYLOAD);
  log_debug("Got a reward");
  current_score++;
}

static inline void add_no_reward()
{
  spin1_send_mc_packet(key | (SPECIAL_EVENT_NO_REWARD), 0, NO_PAYLOAD);
  log_debug("No reward");
  current_score--;
}

void resume_callback() {
    recording_reset();
}

//void add_event(int i, int j, colour_t col, bool bricked)
//{
//  const uint32_t colour_bit = (col == COLOUR_BACKGROUND) ? 0 : 1;
//  const uint32_t spike_key = key | (SPECIAL_EVENT_MAX + (i << 10) + (j << 2) + (bricked<<1) + colour_bit);
//
//  spin1_send_mc_packet(spike_key, 0, NO_PAYLOAD);
//  log_debug("%d, %d, %u, %08x", i, j, col, spike_key);
//}

static bool initialize(uint32_t *timer_period)
{
    log_info("Initialise breakout: started");

    // Get the address this core's DTCM data starts at from SRAM
    address_t address = data_specification_get_data_address();

    // Read the header
    if (!data_specification_read_header(address))
    {
      return false;
    }
    /*
    simulation_initialise(
        address_t address, uint32_t expected_app_magic_number,
        uint32_t* timer_period, uint32_t *simulation_ticks_pointer,
        uint32_t *infinite_run_pointer, int sdp_packet_callback_priority,
        int dma_transfer_done_callback_priority)
    */
    // Get the timing details and set up thse simulation interface
    if (!simulation_initialise(data_specification_get_region(REGION_SYSTEM, address),
    APPLICATION_NAME_HASH, timer_period, &simulation_ticks,
    &infinite_run, 1, NULL))
    {
      return false;
    }
    log_info("simulation time = %u", simulation_ticks);


    // Read breakout region
    address_t breakout_region = data_specification_get_region(REGION_BREAKOUT, address);
    key = breakout_region[0];
    log_info("\tKey=%08x", key);
    log_info("\tTimer period=%d", *timer_period);

    //get recording region
    address_t recording_address = data_specification_get_region(
                                       REGION_RECORDING,address);
    // Setup recording
    uint32_t recording_flags = 0;
    if (!recording_initialize(recording_address, &recording_flags))
    {
       rt_error(RTE_SWERR);
       return false;
    }

    address_t arms_region = data_specification_get_region(REGION_ARMS, address);
    reward_delay = arms_region[0];
    number_of_arms = arms_region[1];
    arm_probabilities = (uint32_t *)arms_region[2];
    //TODO check this prints right, ybug read the address
    log_info("%d", (uint32_t *)arms_region[0]);
    log_info("%d", (uint32_t *)arms_region[1]);
    log_info("%d", (uint32_t *)arms_region[2]);


    log_info("Initialise: completed successfully");

    return true;
}

bool was_there_a_reward(){
    int choice = rand() % number_of_arms;
    int highest_value = 0;
    if(arm_choices[0] > highest_value){
        choice = 0;
        highest_value = arm_choices[0];
    }
    log_info("0 was spiked %d times", arm_choices[0]);
    arm_choices[0] = 0;
    for(int i=1; i<number_of_arms; i=i+1){
        if (arm_choices[i] > highest_value){
            choice = i;
            highest_value = arm_choices[i];
        }
        log_info("%d was spiked %d times", i, arm_choices[i]);
        arm_choices[i] = 0;
    }
    double probability_roll = (double)rand() / (double)RAND_MAX;
    log_info("roll was %d and prob was %d", probability_roll, arm_probabilities[choice]);
    if(probability_roll < arm_probabilities[choice]){
        log_info("reward given");
        return true;
    }
    else{
        log_info("no cigar");
        return false;
    }
}

void mc_packet_received_callback(uint key, uint payload)
{
    uint32_t compare;
    compare = key & 0x7;
    log_info("compare = %x", compare);
    use(payload);
    if(compare == KEY_ARM_0){
        arm_choices[0]++;
    }
    else if(compare == KEY_ARM_1){
        arm_choices[1]++;
    }
    else if(compare == KEY_ARM_2){
        arm_choices[2]++;
    }
    else if(compare == KEY_ARM_3){
        arm_choices[3]++;
    }
    else if(compare == KEY_ARM_4){
        arm_choices[4]++;
    }
    else if(compare == KEY_ARM_5){
        arm_choices[5]++;
    }
    else if(compare == KEY_ARM_6){
        arm_choices[6]++;
    }
    else if(compare == KEY_ARM_7){
        arm_choices[7]++;
    }
    else {
        log_info("it broke arm selection %d", key);
    }
}

void timer_callback(uint unused, uint dummy)
{
    use(unused);
    use(dummy);

    _time++;
    score_change_count++;

    if (!infinite_run && _time >= simulation_ticks)
    {
        //spin1_pause();
        recording_finalise();
        // go into pause and resume state to avoid another tick
        simulation_handle_pause_resume(resume_callback);
        //    spin1_callback_off(MC_PACKET_RECEIVED);

        log_info("infinite_run %d; time %d",infinite_run, _time);
        log_info("simulation_ticks %d",simulation_ticks);
        //    log_info("key count Left %u", left_key_count);
        //    log_info("key count Right %u", right_key_count);

        log_info("Exiting on timer.");
        simulation_handle_pause_resume(NULL);

        _time -= 1;
        return;
    }
    // Otherwise
    else
    {
        // Increment ticks in frame counter and if this has reached frame delay
        tick_in_frame++;
        if(tick_in_frame == reward_delay)
        {
            if (was_there_a_reward()){
                add_reward();
            }
            else{
                add_no_reward();
            }
            // Reset ticks in frame and update frame
            tick_in_frame = 0;
//            update_frame();
            // Update recorded score every 10s
            if(score_change_count>=10000){
                recording_record(0, &current_score, 4);
                score_change_count=0;
            }
        }
    }
//    log_info("time %u", ticks);
//    log_info("time %u", _time);
}
//-------------------------------------------------------------------------------

INT_HANDLER sark_int_han (void);


//-------------------------------------------------------------------------------

//----------------------------------------------------------------------------
// Entry point
//----------------------------------------------------------------------------
void c_main(void)
{
  // Load DTCM data
  uint32_t timer_period;
  if (!initialize(&timer_period))
  {
    log_error("Error in initialisation - exiting!");
    rt_error(RTE_SWERR);
    return;
  }

  tick_in_frame = 0;

  // Set timer tick (in microseconds)
  log_info("setting timer tick callback for %d microseconds",
              timer_period);
  spin1_set_timer_tick(timer_period);

  log_info("simulation_ticks %d",simulation_ticks);

  // Register callback
  spin1_callback_on(TIMER_TICK, timer_callback, 2);
  spin1_callback_on(MC_PACKET_RECEIVED, mc_packet_received_callback, -1);

  _time = UINT32_MAX;

  simulation_run();




}
