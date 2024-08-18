# MoCap-Robotic-Imitation-MEMD

**Motion Capture-based Robotic Imitation: A Keyframeless Implementation Method using Multivariate Empirical Mode Decomposition**

This repository contains the implementation code for motion capture-based robotic imitation using MEMD (Multivariate Empirical Mode Decomposition). The project allows for the decomposition and application of motion data on NAO robots or simulators.

## Prerequisites

### 1. `MEMD_opt_Punch.py`
- **Python 3.x** environment
- Required Python packages:
  - `numpy`
  - `scipy`
  - `matplotlib`
  
### 2. `apply_motionSim.py`
- **Python 2.7** environment
- NAOqi SDK (download from [Aldebaran NAO 6 Support](https://www.aldebaran.com/en/support/nao-6/downloads-softwares))
- Choregraphe software for NAO 6 robot simulation (required if no physical robot is available)

## File Structure

```plaintext
├── CSVdata/                  # Directory containing input CSV files for motion data
├── punch/                    # Directory containing output data
├── utils/                    # Utility scripts for MEMD and other processes
├── .gitignore                # Git ignore file
├── LICENSE                   # License for the project
├── MEMD_opt_Punch.py         # Script for MEMD motion data processing
├── README.md                 # Project documentation
└── apply_motionSim.py        # Script for applying motion to NAO robot/simulator
```

## Usage

### 1. MEMD_opt_Punch.py

This script processes motion data using MEMD and compares the results with the Fourier Transform (FT). It also optimizes the output using a custom algorithm. The script is designed to work with motion data that has already been converted for use with the NAO robot. The example provided uses "punch" motion data.

#### Command-Line Arguments

- `--output_csv`: Path to the input CSV file containing the motion data.
- `--output_opt`: Path to the output directory where results will be saved.

#### Example

```bash
python MEMD_opt_Punch.py --output_csv "./CSVdata/punch.csv" --output_opt "./punch"
```

### 2. apply_motionSim.py

After generating optimized CSV data, this script is used to apply the data to a NAO robot or simulator. This script runs in a Python 2.7 environment and requires the NAOqi SDK and Choregraphe software (if using a simulator). You need to specify the path to the data and the robot IP address.

#### Command-Line Arguments

- `--ip`: The IP address of the robot ("localhost" for the simulator, "nao.lan" for the real robot).
- `--port`: The port number for the NAO simulator (required if `ip` is "localhost").
- `--motionpath`: The directory containing the motion data.
- `--datapath`: The CSV file containing the motion data to be applied.

#### Examples

**For the Simulator:**

```bash
python apply_motionSim.py --ip "localhost" --port 59477 --motionpath "./punch/" --datapath "out_hhtAgr.csv"
```

**For the Real Robot:**

```bash
python apply_motionSim.py --ip "nao.lan" --motionpath "./punch/" --datapath "out_hhtAgr.csv"
```


### Important Notes:
- **Python 2.7** is required to run `apply_motionSim.py`.
- Ensure that the NAOqi SDK is properly installed and configured.
- If using a simulator, make sure Choregraphe software is installed, and a NAO 6 simulator is running with the correct port number.

## Article Reference

This repository is based on the research paper:

**Ran Dong, Qiong Chang, Meng Joo Er, Junpei Zhong, and Soichiro Ikuno,** "Motion Capture-based Robotic Imitation: A Keyframeless Implementation Method using Multivariate Empirical Mode Decomposition," ASME 2024. [Link to paper]

```bibtex
@article{dong2024motion,
  title={Motion Capture-based Robotic Imitation: A Keyframeless Implementation Method using Multivariate Empirical Mode Decomposition},
  author={Ran Dong and Qiong Chang and Meng Joo Er and Junpei Zhong and Soichiro Ikuno},
  journal={IEEE/ASME Transactions on Mechatronics},
  year={2024},
  note={In Press}
}
```

If you use this code or any part of it in your research, please cite the above paper.


## License

This project is licensed under the MIT License - see the LICENSE file for details.
