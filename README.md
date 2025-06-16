# Final-year-thesis

## About
This is my senior year thesis at **King Mongkut's University of Technology North Bangkok (KMUTNB)** as an robotic engineering student. This thesis compares two deep learning models, CNN (VGG-16) and RNN (LSTM), for the purpose of classify shrimp DNA sequences.
There are three classes:healthy, infected with White spot disease (WSSV), and infected with Acute Hepatopancreatic Necrosis Disease (AHPND). These are the two crucial virus that imapcted Thailand's shrimp farmimg industry.

## Datasets and Files
Datasets are retrieved by the NCBI databank, and all the details are in METADATA.xlsx
All Full model is in my hugging face https://huggingface.co/SandmanIV/Thesis_ShrimpAnalysis/tree/main

## Abstract 
Shrimp farming is a vital industry in Thailand, contributing significantly to the nation's 
economy and food security. However, this sector faces major challenges due to highly contagious 
diseases such as White Spot Syndrome Virus (WSSV) and Acute Hepatopancreatic Necrosis 
Disease (AHPND). These diseases can cause mass shrimp mortality, leading to severe economic 
losses for farmers and disrupting the aquaculture supply chain. Traditional diagnostic methods, 
such as Polymerase Chain Reaction (PCR) testing, although accurate, are time-consuming, labor
intensive, and require specialized laboratory equipment, limiting their accessibility and 
scalability for real-time disease monitoring. This research focuses on developing an effective 
Deep Learning-based system designed for the detection of WSSV and AHPND infection in 
shrimps using DNA sequence analysis. The proposed system utilizes Convolution Neural 
Networks (CNN) and Recurrent Neural Networks (RNN) to analyze and classify genetic 
sequences, enabling rapid and accurate identification of infections. By employing techniques 
such as K-mer encoding and one-hot encoding, this system enhances the efficiency of data 
preprocessing, allowing for precise pattern recognition in shrimp DNA sequences. The 
experimental result shows that the CNN model with one-hot encoding performed better than the 
RNN model with K-mer encoding. CNN resulted with higher accuracy, lower loss, F-1 score of 
0.999, and average true prediction probability of 0.9996. While RNN resulted in lower accuracy, 
higher loss, F-1 score of 0.95, and average true prediction probability of 0.9942. It can be 
concluded that CNN is the suitable model for shrimp health analysis.

