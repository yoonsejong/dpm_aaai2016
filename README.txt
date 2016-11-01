This is a reference MATLAB implementation of Distirubted Bayesian Principal Component Analysis (D-BPCA). The code is provided under GPLv2. 

Please note that this is a preliminary release for proof of concept purpose. We are working on to release much faster (~ 6x), full fledged (using OpenMP / MPI), MEX-based implementation and demo scripts on github in the near future.

===
GPL
===

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 2 of the License.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.

========
CITATION
========

If you used this program for any of your work, please cite the publication below:

B. Gholami, S. Yoon and V. Pavlovic. "Decentralized Approximate Bayesian Inference for Distributed Sensor Network". The 30th AAAI Conference on Artificial Intelligence (AAAI), Phoenix, Arizona, USA, 2016.

===================
HOW TO RUN THE CODE
===================

Requires a reasonably recent version of MATLAB (2014-). Just hit 

>> test1_cbpca

and

>> test2_dbpca

from command window.

=======================
OTHER DATASET AND TOOLS
=======================

For Caltech Turntable dataset, please refer: http://www.vision.caltech.edu/Image_Datasets/3D_objects/

For Hopkins 155 dataset, please refer: http://www.vision.jhu.edu/data/hopkins155/

For Voodoo Camera Tracker, please see: http://www.viscoda.com/index.php/en/products/non-commercial/voodoo-camera-tracker
