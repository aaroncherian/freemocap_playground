# FreeMoCap Validation Project - Running Notes
[![hackmd-github-sync-badge](https://hackmd.io/JdUwnYRhSle8D8UxfBYSCw/badge)](https://hackmd.io/JdUwnYRhSle8D8UxfBYSCw)
###### tags: `freemocap` `validation` `aaron` `NIH`

## 2023-05-03

- Need a new subject, want to check in about protocols and planned analyses
- Just standing or also including walking (treadmill/overground)?
    - leaning towards standing only 
    - don't need to do NIH stuff on the treadmill if we don't do any walking things 
    - can use force plates as well
- Time syncing BalancePod with FreeMoCap
    - start FreeMoCap and then start BalancePod, try and keep track of when we started BalancePod
- Protocol:
    - Set up cameras (circle of 6 ideally)
    - Calibrate 
    - Hold A-pose
    - Eyes Open/Solid Ground 
    - Eyes Closed/Solid Ground
    - Eyes Open/Foam
    - Eyes Closed/Foam
    - Tandem / eyes open / solid ground
    - End recording, start new recording 
    - One foot standing / eyes open / solid ground
    - One foot standing / eyes closed / solid ground
    - Repeat x amount of times
    - Step on and off the force plates between different balance conditions
    - Check NIH instructions (add a line about maxing performance in some way if not included)
- 
- 6 - 12 subjects total?
- Try and record audio


- NIH weaknesses:
    - Can't time sync (control when things start and end well)
    - Can't control the order of the tasks 
====
old stuff below

## 2022-01-08 Jon/Aaron/Trent

in meeting: Jon, Aaron, Trent

### Main project research question

Is it possible to use FreeMoCap as a research tool to get "clinically relevant measurements":
Where by 'clincally relevant measurements', we will compare against the NIH toolbox for standing posture and TBD battery of standard measurements for walking and running

#### NIH Toolbox
- Assess standing posture equal or better than NIH Toolbox based posture-assessment method
    - Try to match the capacity afforded by that tool using `freemocap`, based on the perspective on an end-user of the NIH Toolbox
        - https://www.healthmeasures.net/explore-measurement-systems/nih-toolbox
        - https://nihtoolbox.my.salesforce.com/sfc/p/#2E000001H4ee/a/2E0000004yR3/Ckb_AKw1oFUC56tgf6tdxcGDYaYbu8rsmBSFOX2Ec4g
    - 
        
    - TASK LIST 
        - Understand the specific algorithm involved
            - the equation used to calculate path length is on page 17
            - ![](https://i.imgur.com/JQS6Ssq.png)
        - Understand how it is used in practice
            - Finding recent studies that use it
                - e.g. this validation of the NIH toolbox vs 'biodex' which is some force plate thing
                    - https://www.tandfonline.com/doi/full/10.1080/09593985.2022.2027584
                    - Is there a meta-analysis/systematic review of the NIH toolbox?
            - talking to folks who use this as part of their research methods
    - OPEN QUESTIONS
        - Do we need to exactly mimic the 'ipod touch' based methods, or can we use other accelerometers and mocap based methods?
            - there is an advantage of following all their instructions to a T
            - So yes, probably let's do the standard approach
        - Probably - 
            - Follow standard methods for NIH  toolbox to get the 'score' 
                - Use qualisys to match against the deeper levels of kinematic analysis 
        - Are there other assessment things in the NIH toolbox that we could/should look at (other than the postural one)
        - How many subjects will we need?
            - Power analysis based on our data? 
            - Based on others' data?


#### Walking/Running
- Record walking/running gait on a treadmill with sufficient accuracy for clinical assessment
    - soft matching the '2 minute walk' task in the NIH Toolbox
        - probably have folks walk around in a circle for that time and measure distance
    - OPEN QUESTIONS
        - Treadmill, overground, or both?
        - Which measures to use? 
            - e.g. joint angles

#### General Tasks
- Reach out to clinical biomechanists to understand how these and other tools are used in practice
    - Josh Stefanik (but probably starting with Corey and Kara)
        - We need to use Stafanik-style marker placement
        - Probably also want to get numbers from Visual3d based analysis
    - Max Shepherd 
        - mostly knows prosthetics, but works in clinical settings
        - may know who to talk to AND/OR be looking for an excuse to reach out to people
    - Local hospitals/rehab centers (esp those with gait labs), including
        - Spaulding
        - Boston Children's 
        - etc
- Aaron will - 
    - assess the state of the github repo and move most/all things to an `old` folder so that we can pull only what is needed as it is needed, using a pull-request based workflow
    - Similar conversation for available`data`
        - What data do we have on hand that is 
            - relevant enough to work on 
            - clean enough to care about
            - etc
    - Put together notes about relevant papers, 
        - e.g.
            - papers validating various markerless methods
            - 'real' uses of NIH toolbox
            - Papers on general gait assessment batteries (esp. w.r.t. validation stuff)
        - Specifically looking for - 
            - standard measurements 
            - numbers of subjects (for reference and maybe for power analysis)

#### Open Questions Other
- compare  gopro/webcam, different tracking algos?, 
    - focus on the core stuff, we can add in the secondary comparisons later if that makes sense
    - Core stuff - 
        - Webcams
        - mediapipe vs qualisys

#### Top of our heads - State of the whosits
- COM calculation -  is pretty good
    - from freemocap
    - from qualisys
    - and ways to compare between them
- We have pilot data roughly matching NIH Toolbox postural test
    - freemocap & qualisys
- We have pilot data of locomotion on treadmill
    - both raw videos and paired fmc/qualisys
- We need good method for measuring joint angles
    - We can use Pose2Sim/OpenSim for that
- WE need better foam pad
- We need to be able mimick clinical marker placement procedure
- We need better clean up of raw freemocap data
    - specifically - gap filling and butterworth filter
    - Again, we can use Pose2Sim for filtering
    - need our own gap filler
    - possibly using `tidymocap` 
        - https://github.com/roaldarbol/tidymocap
        - needs convert `npy` to `csv` pandas dataframe
- I would LIKE a better triangulation filter in `freemocap`
    - i.e. using reprojection error to decide which camera's views to use (rather than current method of trusting `mediapipe`'s confidence value)
- Better method for time synchronzing `freemocap` with `qualisys`
    - we could use:
        -  the camera flash
        -  Arduino based TTL pulse
            -  e.g. have an arduino attached to QUalisys computer AND freemocap computer, 
            -  attach a trigger to both arduinos
            -  and then have each computer record the timestamp that it recieved teh pulse from the tigger
            -  use that to synchronize the data streams
    - method for resampling things at different framerates

## 2022-09-21

- agenda
    - equipment
        - pad
            - buy from amazon
        - NIH Toolbox
            - Erin Meier has the app
        - iPad
            - Trent said he had one
            - Michael will get one
            - need to figure out the plan 
- Experimental protocol
    - start freemocap recording
        - bonus points - start an screen record with audio and say out loud what is happening
        - assume calibration has been settled
    - stand in A-pose (palms facing forward) for count of 10
    - Range of Motion 
        - Start from head down to feet,  move each joint through full range of motion
            - head
            - lshoulder/rshoulder - swing forward and back
            - elbows - rotate
            - wrists
            - hands - palms to camera and move fingers
            - hips (swing torso around)
            - legs forward/back/left/right
            - knee - lift and swing around 
            - ankle
    - NIH Balance assessment (quiet stance)
        - (ideally would be random order each time, but we can do it in order here)
        - 70 seconds each
        - Eyes open, solid ground
        - eyes closed, solid ground
        - eyes open, foam pad
        - eyes closed, foam pad
    - Star balance (active balance)
        - stand in the middle of an 8-pointed star of tape on the ground
            - arms of the star longer than you reach with your foot
        - touch your toe as far along each arm of the star as you can (lightly touching with toe)
        - do it for both legs (clockwise direction for right leg, anticlockwise for left)
        - repeat x3? 
        - depends on how long the rest of it takes

## 2022-11-02

### Data collection day! :D

sub - ATJ

equipment: 
- 6 webcams
- PC from jon's office


software: 
- fresh install of `freemocap==0.0.54` in `python=3.7` anaconda
    - with `pip install upgrade`

Qualisys Markerset:

- Qualisys Sport marker set  (42)
    - 6 additional
        - 2 medial ankles
        - 4 shoulders
            - one on front of shoulder one on back , for each shoulder
Recording: 

1. 