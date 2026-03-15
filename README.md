# CS_129_Code_To_Turn_In
Code for CS 129 Project by Caio Gould

Based on OpenKBP code by Babier et al. (2021)

This repository contains code used for the OpenKBP dose prediction project.

Edited files:
- two_convolutions_per_level_network_architectures.py
  
  Implements a U-Net with two convolutions per level. This is based on the OpenKBP one-convolution-per-level U-Net located in OpenKBP_provided_code/network_architectures.py.

- cascade_network_architectures.py
  
  Implements a cascade U-Net with two U-Nets in sequence, each using two convolutions per level. This is also based on the OpenKBP one-convolution-per-level U-Net located in OpenKBP_provided_code/network_architectures.py.

- network_functions_modified_to_save_losses.py
  
  A modified copy of the OpenKBP network functions that calculates and saves training and validation losses.

- plotting.ipynb
  
  Notebook used to generate the plots in the report and poster.


All other files come from the original OpenKBP challenge repository:
Babier A., Zhang B., Mahmood R., Moore K.L., Purdie T.G., McNiven A.L., Chan T.C.Y., "OpenKBP: The open-access knowledge-based planning grand challenge and dataset." Med. Phys. 2021; 48: 5549-5561. doi.org/10.1002/mp.14845.

provided_code is the code provided by the OpenKBP challenge repository.

The OpenKBP dataset and generated results are not included due to size.
