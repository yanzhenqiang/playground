
# Foo
https://github.com/mbrossar/denoise-imu-gyro
http://www.iri.upc.edu/people/jsola/JoanSola/objectes/notes/kinematics.pdf
https://arxiv.org/pdf/2002.10718.pdf
https://docs.novatel.com/OEM7/Content/SPAN_Logs/CORRIMUDATA.htm
https://hal.archives-ouvertes.fr/hal-02488923/document



rosbag filter 11024_20200901_140751_default_0.bag 11024_20200901_140751_filter.bag "topic == '/pose' or topic == '/localization/ins_pose' or topic == '/gps_imu_data' or '/raw_imu_data'"


python3 convert.py -t 11024_20190730_091616 -r /home/pilot/yanzhenqiang/bags/11024_20190730_091616_default_0.bag.bak -o ./


/home/pilot/yanzhenqiang/denoise-imu-gyro/visualization/visualize_pose.py -t 11024_20190730_091616_0 -r /home/pilot/yanzhenqiang/denoise-imu-gyro/11024_20190730_091616_0.bag -o /home/pilot/yanzhenqiang/denoise-imu-gyro/raw/

# Data
11024_20190730_100503_default_0 # raw imu data
11024_20200901_140751_default_0 # corrimu data


## Installation

```
pip install -r requirements.txt
```

## Training
```
python3 main.py
```

## EuRoC data format[TL;NR]
``` txt
- mav0
    - cam0
        - data: contain the png image, such as timestamp.png.
        - data.csv
            head -n5 data.csv
            #timestamp [ns],filename
            1403636579763555584,1403636579763555584.png
            1403636579813555456,1403636579813555456.png
            1403636579863555584,1403636579863555584.png
            1403636579913555456,1403636579913555456.png
        - sensor.yaml
            # General sensor definitions.
            sensor_type: camera
            comment: VI-Sensor cam0 (MT9M034)

            # Sensor extrinsics wrt. the body-frame.
            T_BS:
              cols: 4
              rows: 4
              data: [0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
                    0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,
                    -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949,
                    0.0, 0.0, 0.0, 1.0]

            # Camera specific definitions.
            rate_hz: 20
            resolution: [752, 480]
            camera_model: pinhole
            intrinsics: [458.654, 457.296, 367.215, 248.375] #fu, fv, cu, cv
            distortion_model: radial-tangential
            distortion_coefficients: [-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05]
    - cam1
        - [As above]
    - imu0
        - data.csv
            head -n5 data.csv 
            #timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]
            1403636579758555392,-0.099134701513277898,0.14730578886832138,0.02722713633111154,8.1476917083333333,-0.37592158333333331,-2.4026292499999999
            1403636579763555584,-0.099134701513277898,0.14032447186034408,0.029321531433504733,8.033280791666666,-0.40861041666666664,-2.4026292499999999
            1403636579768555520,-0.098436569812480182,0.12775810124598494,0.037699111843077518,7.8861810416666662,-0.42495483333333334,-2.4353180833333332
            1403636579773555456,-0.10262536001726656,0.11588986233242347,0.045378560551852569,7.8289755833333325,-0.37592158333333331,-2.4680069166666665
        - sensor.yaml
            cat sensor.yaml 
            #Default imu sensor yaml file
            sensor_type: imu
            comment: VI-Sensor IMU (ADIS16448)

            # Sensor extrinsics wrt. the body-frame.
            T_BS:
              cols: 4
              rows: 4
              data: [1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 1.0]
            rate_hz: 200

            # inertial sensor noise model parameters (static)
            gyroscope_noise_density: 1.6968e-04     # [ rad / s / sqrt(Hz) ]   ( gyro "white noise" )
            gyroscope_random_walk: 1.9393e-05       # [ rad / s^2 / sqrt(Hz) ] ( gyro bias diffusion )
            accelerometer_noise_density: 2.0000e-3  # [ m / s^2 / sqrt(Hz) ]   ( accel "white noise" )
            accelerometer_random_walk: 3.0000e-3    # [ m / s^3 / sqrt(Hz) ].  ( accel bias diffusion )
    - leica0
        - data.csv
            #timestamp [ns],p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m]
            1403636578922881280,4.7822997174796811,-1.8155736277079491,0.84462707370341705
            1403636578968881408,4.7822118752035667,-1.8147864986792011,0.85767100487377101
            1403636579022881280,4.7807530761485442,-1.8131922179613229,0.87462386853895402
            1403636579068881152,4.7791190472344942,-1.812024099627725,0.89192319457420399
        - sensor.yaml
            # General sensor definitions.
            sensor_type: position
            comment: Position measurement from a Leica Nova MS50.

            # Sensor extrinsics wrt. the body-frame. This is the transformation of the
            # tracking prima to the body frame.
            T_BS:
              cols: 4
              rows: 4
              data: [1.0, 0.0, 0.0,  7.48903e-02,
                    0.0, 1.0, 0.0, -1.84772e-02,
                    0.0, 0.0, 1.0, -1.20209e-01,
                    0.0, 0.0, 0.0,  1.0]
    - state_groundtruth_estimate0
        - data.csv
            #timestamp, p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z [], v_RS_R_x [m s^-1], v_RS_R_y [m s^-1], v_RS_R_z [m s^-1], b_w_RS_S_x [rad s^-1], b_w_RS_S_y [rad s^-1], b_w_RS_S_z [rad s^-1], b_a_RS_S_x [m s^-2], b_a_RS_S_y [m s^-2], b_a_RS_S_z [m s^-2]
            1403636580838555648,4.688319,-1.786938,0.783338,0.534108,-0.153029,-0.827383,-0.082152,-0.027876,0.033207,0.800006,-0.003172,0.021267,0.078502,-0.025266,0.136696,0.075593
            1403636580843555328,4.688177,-1.786770,0.787350,0.534640,-0.152990,-0.826976,-0.082863,-0.029272,0.033992,0.804771,-0.003172,0.021267,0.078502,-0.025266,0.136696,0.075593
            1403636580848555520,4.688028,-1.786598,0.791382,0.535178,-0.152945,-0.826562,-0.083605,-0.030043,0.034999,0.808240,-0.003172,0.021267,0.078502,-0.025266,0.136696,0.075593
            1403636580853555456,4.687878,-1.786421,0.795429,0.535715,-0.152884,-0.826146,-0.084391,-0.030230,0.035853,0.810462,-0.003172,0.021267,0.078502,-0.025266,0.136696,0.075593
        - sensor.yaml
            # General sensor definitions.
            sensor_type: visual-inertial
            comment: The nonlinear least-squares batch solution over the Leica position and IMU measurements including time offset estimation. The orientation is only observed by the IMU.

            # Sensor extrinsics wrt. the body-frame. This is the transformation of the
            # tracking prima to the body frame.
            T_BS:
              cols: 4
              rows: 4
              data: [1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 1.0]
    - body.yaml
```