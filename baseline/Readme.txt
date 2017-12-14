ASVspoof 2017 CHALLENGE:
Audio replay detection challenge for automatic speaker verification anti-spoofing

http://www.spoofingchallenge.org/

===================================================================================================
Matlab implementation of the baseline system for replay detection based on constant Q cepstral coefficients (CQCC) features + Gaussian Mixture Models (GMMs)
===================================================================================================

Contents of the package
=======================

- CQCC_v1.0
------------------
Matlab implementation of constant Q cepstral coefficients.

For further details on CQCC, refer to the following publication:
Todisco, M., Delgado, H., Evans, N., A new feature for automatic speaker verification anti-spoofing: Constant Q cepstral coefficients. ODYSSEY 2016, The Speaker and Language Recognition Workshop, June 21-24, 2016, Bilbao, Spain

http://audio.eurecom.fr/content/software

- vlfeat
-----------------
VLFeat open source library that implements the GMMs.

For further details, refer to:
http://www.vlfeat.org/

- bosaris_toolkit
-----------------
The BOSARIS Toolkit provides MATLAB code for evaluating scores.

For further details, refer to:
https://sites.google.com/site/bosaristoolkit/

- baseline_CM.m
----------------
This script is the baseline countermeasures system for ASVspoof 2017 CHALLENGE - Audio replay detection challenge for automatic speaker verification anti-spoofing. 
Front-end is based on CQCC features, while back-end is based on GMMs. 
The CQCC is applied with a maximum frequency of fs/2, where fs = 16kHz is the sampling frequency.
The minimum frequency is set to fs/2/2^9 ~15Hz (9 being the number of octaves). 
The number of bins per octave is set to 96. Re-sampling is applied with a sampling period of 16.
CQCC features dimension is set to 29 coefficients + 0th, with the static, delta and delta-delta coefficients.
2-class GMMs are trained on the genuine and spoofed speech utterances of the training dataset, respectively. We use 512-component models, trained with an expectation-maximisation (EM) algorithm with random initialisation. The score is computed as the log-likelihood ratio for the test utterance given the natural and the spoofed speech models.
BASELINE PERFORMANCE UPDATE!! Using the originally distributed dataset (file "ASVspoof2017.tar.gz"), performance in terms of equal error rate (EER) on the development set should be close to 7%. However, with the newly distributed data (file "ASVspoof2017_train_dev_update.tar.gz") with the DTFM tones removed/reduced, EER should be around 11%.

Contact information
===================

For any query, please contact:

- asvspoof2017 at cs.uef.fi

- Hector Delgado (delgado at eurecom.fr)
- Massimiliano Todisco (todisco at eurecom.fr)
- Nicholas Evans (evans at eurecom.fr)

