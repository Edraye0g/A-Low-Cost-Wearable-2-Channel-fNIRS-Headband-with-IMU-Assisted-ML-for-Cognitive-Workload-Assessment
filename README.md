# A-Low-Cost-Wearable-2-Channel-fNIRS-Headband-with-IMU-Assisted-ML-for-Cognitive-Workload-Assessment

## Problems
 The Prefrontal Cortex (PFC) is a special part of the brain connected with higher cognition, attention, and working memory. With standard fNIRS (Functional Near-Infrared Spectroscopy) they use multichannel module on the scalp to detect topological map of BOLD(Brain-Oxygen Level) activity. While doing higher cognitive tasks, our head naturally moves, jaw clenches, eye-brows stretches, practically making the signal noisy with motion artifacts. This is why, low-cost fNIRS faces huge problem dealing with this artifacts. 

## Goal
 The plan is to suppress motion artifact, canceling out a spike from specific time-stamp to reduce noise. Both the photo diode and the motion sensor will be synced to identify the time-stamps. In this project, we will only use a 2-channel model. We will not overkill it by using multichannel model. That would be our future prospect.

## Motivation 
 ### Al-Omairi et al. (2024) and Herff et al. (2014).
  Herff and Causse establish that the Prefrontal Cortex (PFC) is the exact anatomical location for measuring mental workload (n-back/arithmetic). Furthermore, Al-Omairi (2024) directly validates the hardware choice by proving that a cheap IMU improves motion correction compared to relying on optical data alone.

 ### Brigadoi et al. (2013), Cooper et al. (2012), and Al-Omairi et al. (2023).
  Brigadoi and Cooper, who proved across 20 datasets that Wavelet is the strongest baseline for reducing artifact AUC. We are using their mathematical conclusions to design our Python script.

 ### Naseer et al. (2016), Asgher et al. (2019), and Cao et al. (2022).
  Naseer and Asgher already did the hard work of proving that simple time-domain features (Mean and Peak of HbO) fed into an SVM or LDA yield the highest accuracy for mental arithmetic vs. rest. We are importing their exact feature-extraction strategy into our pipeline.

> The Al-Omairi and Brigadoi papers talk endlessly about "Signal-to-Noise Ratio" (SNR), but they don't do Machine Learning. The Naseer and Asgher papers talk endlessly about Machine Learning, but they assume the patient is sitting perfectly still in a lab, ignoring real-world motion.

## Novelty Claim
 **Does the IMU-Wavelet correction method from Al-Omairi actually improve the SVM classification accuracy established by Naseer when deployed on Low-cost DIY hardware?**

## Plan of approach
        Working Memory based task
        Mathmetical calculation (Immediate)
        30s Maths -> 10s Rest
        1 Individual -> 10 trials 
               |
               |
               V
    Headband (Containing 2 channels and a IMU accelerometer) 
               |
               | *sending BOLD signal to photodetector*
               |
               V
    Transimpedance Amplifier (Converting Current to Voltage 
    and Amplify)
               |
               |
               |
               V
    ESP32 for ADC conversion 
               |
               | Sending CSV file to a python script (Wired 
               | or wireless)
               |
               V 
    Python Script with wavelet filtering
               | 
               | RAW data file + 3/6 axis motion filtering 
               |
               V
    Pretrain dataset with 5/6 individual >> Using SVM + LOSO cross validation  
    OR pretrain with a publicly available dataset >> Using deep neural network
               |
               |
               |
               V
    New Reading arrived from a new subject
               |
               |
               |
               V
    Python Script Filters >> SVM models tests with the new 
    subject >> Shows accuracy against RAW data vs Filtered algorithmic data

**Step By Step**
 1. **Headband**
 - ESP32 sends a pulse to 730nm LED. Light enters to forehead and bounces back. 
 - OPT101 photo diode captures the bounced light and converts it into electrical voltage. 
 - At the exact same millisecond, the MPU6050 IMU chip measures the physical tilt and acceleration of your head.

 2. **ESP32 C++**
 - C++ code runs a script that loops a fast flickering session (500 times a second). It turns the 730nm LED ON, takes a reading >> turns the 850nm ON, takes a reading >> then turns OFF to read the ambient darkroom in the insulation. 
 - While also grabbing IMU motion data and timestamp. 
 - It combines all these numbers into a single line of text separated by commas and blasts it to USB cable. 

 3. **Python Script**
 - Python subtracts the room light (the "BOTH OFF" reading) from the LED readings to purify the signal.
 - Python runs the raw voltages through the Modified Beer-Lambert Law (MBLL). The voltages are instantly converted into the actual biological metrics: [HbO] (Oxygenated blood) and [HbR] (Deoxygenated blood).
 - Python checks the IMU data. If it sees a massive spike in the IMU (meaning you moved your head), it triggers the Wavelet Filter to mathematically erase the corresponding noise spike from the [HbO] signal.
 - The brain signal is slow, while the motion signal is very fast. IMU detects sharp (motion signal) spike, IMU detects a sharp physical head jerk at exact time-stamp sends signal to the script >> wavelet filter mathematically deletes the specific fast spike from the specific time-stamp. Other layer like slow brain signal remained untouched. Once the filter deletes the sharp noise layer, Python runs an "Inverse Wavelet Transform." It squishes the remaining layers back together into a single, normal timeline line.

 4. **Machine Learning** 
 - Python looks at the last 30 seconds of clean data. It calculates just two simple numbers: the Mean (average oxygen level) and the Peak (highest oxygen spike) during that window.
 - Python takes those two numbers and feeds them into your pre-trained Support Vector Machine (SVM) file saved on your hard drive. The SVM compares those numbers against what it learned last week.

 5. **Live Dashboard**
 - A Live Graph: A scrolling line chart showing your actual oxygen levels rising and falling.
 -The Verdict: A bold text readout that updates instantly, saying:
 "STATE: HIGH WORKLOAD" (If doing math)
 "STATE: RESTING" (If staring at the wall)

 6. **Real Time Update**
 - First 30 second after the starting, no data shows. From 30th second, data appears. 


## Components
The Brain: 1x ESP32, 1x MPU6050 IMU

The Emitters: 2x Red LEDs (730nm), 2x NIR LEDs (850nm)

The Drivers: 2x 2N2222 Transistors, 2x 4.7k ohm resistors, 2x 100 ohm resistors, 2x 2k ohm potentiometers

The Detectors: 2x OPT101 Sensors

The Amplifiers: 2x LM358 Op-Amps, 2x 15k ohm resistors, 2x 0.1uF capacitors, 4x 10k ohm fixed resistors, 2x 50k ohm potentiometers

## Circuit Design

```
[ESP32 Digital Pin] ──(Control)──> [SN7405 Inverter] ──> [PN2369A Transistor] ──> [LEDs (30-50mA)]
                                                                                       │
[ESP32 ADC1 Input]  <──(Gain Stage)── [LM358 Op-Amp] <──(LPF)── [OPT101 Sensor] <──────┘



THE SIGNAL JOURNEY (fNIRS + IMU)
                        
+------------------+         [ PHASE 1: EMISSION ]              [ PHASE 2: OPTICAL PATH ]
|   THE ESP32      |                                          
| (Microcontroller)|====> 2N2222 Transistors (Switching) ===> 660nm Red & 850nm NIR LEDs
|                  |      - Receives timing pulses            - LEDs flash intensely
|                  |      - Acts as high-speed gates          - Photons enter the skull
| 1. TDM Generator |                                                    ||
| 2. ADC Reader    |                                                    || (Light scatters
| 3. I2C Master    |                                                    ||  in tissue)
+------------------+                                                    \/
     ^        ^                                                 (Hemoglobin Absorption)
     |        |                                                         ||
     |        |          [ PHASE 3: DETECTION ]                         || (Unabsorbed light
     |        |                                                         ||  bounces back up)
     |        |       LM358 Op-Amp Circuit                    OPT101 Photodiodes (1cm & 3cm)
     |        <====== - Filters out 50Hz AC noise     <====== - Catches returning photons
     |   (Clean       - Amplifies weak 3cm signal             - Converts light to raw voltage
     |   Analog)                                                
     |                                                          
     |                   [ PHASE 4: KINEMATIC (MOTION) PATH ]   
     |                                                          
     |                MPU6050 IMU Sensor                        
     <=============== - Tracks X/Y/Z head movement           <====== [ PHYSICAL HEAD MOTION ]
       (I2C Digital)  - Sends digital timestamps
```

## Questions Answer
 ### Question 1
1. SN7405 Hex Inverter & PN2369A NPN Transistors (LED Driver Stage):
Purpose: To drive the 
730 nm 850 nm and LEDs at high pulsed currents (Justification: ESP32 GPIO pins are rated for a maximum of 12 −20 mA 30 −50 mA) safely, which cannot penetrate 1.5 −3 cm) ).of tissue, skull, and scalp. The SN7405 open-collector inverter converts the ESP32's control signals into inverted logic to switch the PN2369A high-speed NPN transistors. The LEDs are driven directly from a regulated 5 V rail, regulated by a 2 kΩ limiting resistor, keeping the microcontroller completely isolated from high currents.
2. OPT101 Monolithic Photodiode & Transimpedance Amplifier (Optical Detection):
Purpose: To detect micro-watt optical fluctuations bouncing back from tissue. trimming potentiometer and a 100Ω Justification: The OPT101 integrates a photodiode and a transimpedance amplifier (TIA) on a single chip with an internal current 1 MΩ feedback resistor. This eliminates parasitic capacitance and stray electromagnetic interference (EMI) that severely degrade discrete photodiode circuits on breadboards.
3. Passive RC
 Filter & LM358 Adjustable Gain Stage (Signal Conditioning):
Purpose: To remove high-frequency noise and stretch tiny millivolt hemodynamic fluctuations across the ESP32's input voltage range. Justification: Cortical hemodynamic changes (R=15 kΩ C =0.1μF, Δ[HbO]2) cause optical intensity shifts of less than 1%. The passive RC low-pass filter () establishes a cutoff frequency.
4. MPU6050 6-Axis Motion Sensor (Inertial Reference):
Purpose: To provide objective, physical ground-truth reference data for head movement and muscle activity.
Justification: Communicating via I2C, it supplies real-time 3-axis acceleration and 3-axis angular velocity to trigger the downstream Wavelet artifact rejection algorithm in Python.

 ### Question 2
Verification of Brain Hemodynamics
Initial Plan (Old Approach)
Depended entirely on placing the headset on the forehead during a cognitive task (mental arithmetic) and assuming that any observed signal shift represented brain activity.
Why this is vulnerable: Cognitive signals are tiny ( ). Without a baseline reference, a reviewer can claim the measured signal is merely forehead sweat, motion artifacts, or superficial scalp blood flow.
```
[TIER 1: Physical Bench Validation] ──> Forearm Ischemia (Cuff Occlusion at 230 mmHg)
                                         (Proves MBLL math & optical front-end)
                                                        │
                                                        v
[TIER 2: Cortical Isolation]         ──> Dual-Channel Subtraction (3cm Long - 1cm Short)
                                         (Isolates Cortex from Scalp Noise)

```
1. Tier 1: Physical Ground-Truth Ischemia (Forearm Cuff Occlusion Test):
Before placing the sensor on the head, the probe is secured to the subject's flexor muscle on the forearm.
A pneumatic cuff is inflated to  for 5 minutes (arterial occlusion) and then rapidly released (reactive hyperemia).
Expected Biological Curve: Arterial occlusion causes an immediate, massive drop in oxygenated hemoglobin ( ) and a
simultaneous rise in deoxygenated hemoglobin ( ). Upon release, a massive hyperemic overshoot occurs.
Verification Proof: Because ischemia produces a mathematically predictable, physiological response, successfully capturing this curve
proves that your hardware, optics, and Modified Beer-Lambert Law (MBLL) code are functioning with 100% accuracy.
2. Tier 2: Dual-Channel Cortical Isolation (Short vs. Long Channel):
Once validated on the arm, the probe is moved to the forehead using a dual-channel configuration:
Short Channel ( separation): Penetrates only skin and scalp tissue; captures systemic vascular noise and cardiac pulsation.
Long Channel ( separation): Penetrates skin, skull, and the outer Prefrontal Cortex.
Verification Proof: By performing spatial regression in Python ( ), superficial scalp blood
flow is subtracted, isolating true cortical hemodynamics.

 ### Question 3: Signal-to-Noise Ratio (SNR)
SNR = 20 log10 ( Vsignal/Vnoise) = 20 log10 (μ/σ)
μ is the mean DC voltage level of the optical channel during a stable baseline state.
σ is the standard deviation (RMS noise) of the signal.

1. Dark Current SNR  ──> Measured with LEDs off (determines photodiode noise floor)
2. Hardware SNR      ──> Measured on static tissue phantom (determines circuit performance)
3. Filtering SNR     ──> Measured pre- vs. post-IMU Wavelet filter during motion
   
**ΔSNRImprovement = SNRPost-Filter − SNRPre-Filter**
1. Dark Noise SNR: Measured with LEDs completely turned off to calculate the baseline noise floor of the OPT101 photodiode and op-amp
stage.
2. Hardware/Optical SNR: Measured on a static solid tissue phantom (or resting muscle) over a 60-second window.
3. Filter Performance (ΔSNR): Calculated during intentional head motion to evaluate the efficacy of the IMU-guided Wavelet filter:

## Future Prospect
 ESP32 has lagging issues. With small memory, it cannot work on 16/32 channel to cover the whole skull. But, using a RasberryPi + Frequency Division Multiplexing. They turn on all the 32 LEDs at the exact same time, but they flicker at slightly different frequency. Sensor sees a giant mass of light but rasberryPi will analyze it using Fourier transform to backpropagate and ultimately finding a topographical map. 

1. It immediately justifies why motion correction is necessary (the subject is moving/exercising).
2. It justifies why 2 channels on the PFC are enough (measuring executive function/working memory rather than full-brain motor mapping).
3. It highlights the socio-economic impact of low-cost hardware (bringing neuro-monitoring out of the lab and into homes/clinics).

## Sources
 
**Al-Omairi, Hayder R., Sebastian J. F. Fudickar, Andreas Hein, and J. Rieger. "Improved Motion Artifact Correction in fNIRS Data by Combining Wavelet and Correlation-Based Signal Improvement." *Sensors (Basel, Switzerland)* 23 (2023). https://doi.org/10.3390/s23083979.**
 
**Al-Omairi, Hayder R., Arkan Al-Zubaidi, Sebastian J. F. Fudickar, Andreas Hein, and Jochem W. Rieger. "Hammerstein–Wiener Motion Artifact Correction for Functional Near-Infrared Spectroscopy: A Novel Inertial Measurement Unit-Based Technique." *Sensors (Basel, Switzerland)* 24 (2024). https://doi.org/10.3390/s24103173.**
 
**Brigadoi, S., Lisa Ceccherini, S. Cutini, F. Scarpa, P. Scatturin, J. Selb, L. Gagnon, D. Boas, and R. Cooper. "Motion artifacts in functional near-infrared spectroscopy: a comparison of motion correction techniques applied to real cognitive data." *NeuroImage* 85 (2013). https://doi.org/10.1016/j.neuroimage.2013.04.082.**
 
Chiarelli, A., E. Maclin, M. Fabiani, and G. Gratton. "A kurtosis-based wavelet algorithm for motion artifact correction of fNIRS data." *NeuroImage* 112 (2015): 128 - 137. https://doi.org/10.1016/j.neuroimage.2015.02.057.
 
**Cooper, R., J. Selb, L. Gagnon, Dorte Phillip, H. Schytz, H. Iversen, M. Ashina, and D. Boas. "A Systematic Comparison of Motion Artifact Correction Techniques for Functional Near-Infrared Spectroscopy." *Frontiers in Neuroscience* 6 (2012). https://doi.org/10.3389/fnins.2012.00147.**
 
Di Lorenzo, Renata, Laura Pirazzoli, A. Blasi, C. Bulgarelli, Yoko Hakuno, Yasuyo Minagawa, and S. Brigadoi. "Recommendations for motion correction of infant fNIRS data applicable to multiple data sets and acquisition systems." *NeuroImage* (2019). https://doi.org/10.1016/j.neuroimage.2019.06.056.
 
Fang, Shuqi, Zhiming Xing, B. Cao, Jun Wang, Xiumin Gao, and Xiangmei Dong. "Motion ArtifactCorrection in Fnirs Signals Based on Spline Interpolation and Locally Weighted Regression." *Research Review* (2024). https://doi.org/10.52845/cs/2024-4-4-3.
 
Fishburn, F., Ruth S. Ludlum, C. Vaidya, and A. Medvedev. "Temporal Derivative Distribution Repair (TDDR): A motion correction method for fNIRS." *NeuroImage* 184 (2018): 171 - 179. https://doi.org/10.1016/j.neuroimage.2018.09.025.
 
Guan, Shuo, Yuhang Li, Yuxi Luo, Haijing Niu, Yuanyuan Gao, Dalin Yang, and Rihui Li. "Disentangling the impact of motion artifact correction algorithms on functional near-infrared spectroscopy–based brain network analysis." *Neurophotonics* 11 (2024). https://doi.org/10.1117/1.nph.11.4.045006.
 
Huang, Weihao, and Jun Li. "Enhancing fNIRS data analysis with a novel motion artifact detection algorithm and improved correction." *Biomed. Signal Process. Control.* 95 (2024): 106496. https://doi.org/10.1016/j.bspc.2024.106496.
 
Jahani, S., S. Setarehdan, D. Boas, and M. Yücel. "Motion artifact detection and correction in functional near-infrared spectroscopy: a new hybrid method based on spline interpolation method and Savitzky–Golay filtering." *Neurophotonics* 5 (2018). https://doi.org/10.1117/1.nph.5.1.015003.
 
Perpetuini, D., D. Cardone, C. Filippini, A. Chiarelli, and A. Merla. "A Motion Artifact Correction Procedure for fNIRS Signals Based on Wavelet Transform and Infrared Thermography Video Tracking." *Sensors (Basel, Switzerland)* 21 (2021). https://doi.org/10.3390/s21155117.
 
Zhao, Yunyi, Haiming Luo, Jianan Chen, R. Loureiro, Shufan Yang, and Hubin Zhao. "Learning based motion artifacts processing in fNIRS: a mini review." *Frontiers in Neuroscience* 17 (2023). https://doi.org/10.3389/fnins.2023.1280590.

Aghajani, H., M. Garbey, and A. Omurtag. "Measuring Mental Workload with EEG+fNIRS." *Frontiers in Human Neuroscience* 11 (2017). https://doi.org/10.3389/fnhum.2017.00359.
 
Asgher, Umer, Riaz Ahmad, Noman Naseer, Y. Ayaz, Muhammad Jawad Khan, and M. K. Amjad. "Date of publication xxxx 00, 0000, date of current version xxxx 00, 0000." (2019).
 
Cao, Jun, E. Garro, and Yifan Zhao. "EEG/fNIRS Based Workload Classification Using Functional Brain Connectivity and Machine Learning." *Sensors (Basel, Switzerland)* 22 (2022). https://doi.org/10.3390/s22197623.
 
Causse, M., Zarrin K. Chua, Vsevolod Peysakhovich, N. Del Campo, and N. Matton. "Mental workload and neural efficiency quantified in the prefrontal cortex using fNIRS." *Scientific Reports* 7 (2017). https://doi.org/10.1038/s41598-017-05378-x.
 
Chen, Jianan, Huixin Yang, Yunjia Xia, Tingchen Gong, Alexander Thomas, Jia Liu, Wei Chen, Tom Carlson, and Hubin Zhao. "Simultaneous Mental Fatigue and Mental Workload Assessment With Wearable High-Density Diffuse Optical Tomography." *Ieee Transactions on Neural Systems and Rehabilitation Engineering* 33 (2025): 1242 - 1251. https://doi.org/10.1109/tnsre.2025.3551676.
 
**Herff, Christian, D. Heger, Ole Fortmann, Johannes Hennrich, F. Putze, and Tanja Schultz. "Mental workload during n-back task—quantified in the prefrontal cortex using fNIRS." *Frontiers in Human Neuroscience* 7 (2014). https://doi.org/10.3389/fnhum.2013.00935.**
 
Karmakar, Subashis, Supreeti Kamilya, P. Dey, P. K. Guhathakurta, M. Dalui, T. Bera, Suman Halder, C. Koley, Tandra Pal, and Anupam Basu. "Real time detection of cognitive load using fNIRS: A deep learning approach." *Biomed. Signal Process. Control.* 80 (2023): 104227. https://doi.org/10.1016/j.bspc.2022.104227.
 
Lim, Lam Ghai, W. Ung, Y. L. Chan, Cheng-Kai Lu, S. Sutoko, T. Funane, M. Kiguchi, and T. Tang. "A Unified Analytical Framework With Multiple fNIRS Features for Mental Workload Assessment in the Prefrontal Cortex." *IEEE Transactions on Neural Systems and Rehabilitation Engineering* 28 (2020): 2367-2376. https://doi.org/10.1109/tnsre.2020.3026991.
 
Liu, Shixian. "Applying antagonistic activation pattern to the single-trial classification of mental arithmetic." *Heliyon* 8 (2022). https://doi.org/10.1016/j.heliyon.2022.e11102.
 
Naseer, Noman, F. Noori, N. Qureshi, and K. Hong. "Determining Optimal Feature-Combination for LDA Classification of Functional Near-Infrared Spectroscopy Signals in Brain-Computer Interface Application." *Frontiers in Human Neuroscience* 10 (2016). https://doi.org/10.3389/fnhum.2016.00237.
 
**Naseer, Noman, N. Qureshi, F. Noori, and K. Hong. "Analysis of Different Classification Techniques for Two-Class Functional Near-Infrared Spectroscopy-Based Brain-Computer Interface." *Computational Intelligence and Neuroscience* 2016 (2016). https://doi.org/10.1155/2016/5480760.**

Scholkmann et al. (2014): "A review on continuous wave functional near-infrared spectroscopy instrumentation." NeuroImage, 85, 6-27.
(Reference for source-detector separation and optode design).



