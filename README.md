# LXMERT Model Compression for Visual Question Answering
This project implementation is built on the great repo of [LXMERT](https://github.com/airsplay/lxmert) and PyTorch code for the EMNLP 2019 paper ["LXMERT: Learning Cross-Modality Encoder Representations from Transformers"](https://arxiv.org/abs/1908.07490) on VQA v2.0.

**See the complete report [here](https://github.com/ghazaleh-mahmoodi/lxmert_compression/blob/main/report/main.pdf).**  
# Summary
Large-scale pretrained models such as LXMERT are becoming popular for learning cross-modal representations on text-image pairs for vision-language tasks. According to the lottery ticket hypothesis, NLP and computer vision models contain smaller subnetworks capable of being trained in isolation to full performance. In this project, we combine these observations to evaluate whether such trainable subnetworks exist in LXMERT when fine-tuned on the VQA task. In addition, we perform a model size cost-benefit analysis by investigating how much pruning can be done without significant loss in accuracy.

# Run

## Install the required packages
```bash
pip3 install -r requirements.txt
```

## Run All Experiment
to run all experiment ,in lxmert folder run following command:
```bash
bash run/vqa_run.bash
```

# Results 
- The plots are available in `lxmert/result` directory.
- The trained models are available in `lxmert/models` directory.
- The logs are available in `lxmert/logs` directory.

## Plots
### Low Magnitude Pruning Subnetwork 
![alt text](./report/images/Low_Magnitude_experiment_result.PNG)

***

### Random Pruning Subnetwork 
![alt text](./report/images/Random_experiment_result.PNG)

***

### High Magnitude Pruning Subnetwork 
![alt text](./report/images/High_Magnitude_experiment_result.PNG)

*** 

### All Result Based on Pruning Sparcity
![alt text](./report/images/experiment_result.PNG)

***

### All Result Based on Pruning mode
![alt text](./report/images/table.PNG)
