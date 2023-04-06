#!/bin/bash

#
#   Copyright 2021 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

downloadSS () {
	if [ ! -f ${1}.mtx ]; then
		if [ ! -f ${1}.tar.gz ]; then
			wget ${2} || exit 1
		fi
		if [ ! -f ${1}/${1}.mtx ]; then
			tar xf ${1}.tar.gz || exit 1
		fi
		ln ${1}/${1}.mtx ./ || exit 1
	fi
}

downloadSN () {
	if [ ! -f ${1}.txt ]; then
		if [ ! -f ${1}.txt.gz ]; then
			wget https://snap.stanford.edu/data/${1}.txt.gz || exit 1
		fi
		gunzip -k ${1}.txt.gz || exit 1
	fi
}

DATASETS_DIR="$(pwd)/datasets"

echo "This script will download matrices from the SuiteSparse Matrix Collection [1], which "
echo "is maintained by Tim Davis, Yifan Hu, and Scott Kolodziej. It also downloads two "
echo "matrices from the SNAP collection maintained by Jure Leskovec [2]."
echo " "
echo "The matrices downloaded from the SuiteSparse Matrix Collection are:"
echo " - west0497, Chemical Process Simulation Problem [3]"
echo " - gyro_m, Duplicate Model Reduction Problem [4,5]"
echo " - dwt_59, Structural Problem [3]"
echo " - EPA, Web Link Matrix [8]"
echo " "
echo "The matrices downloaded from SNAP are:"
echo " - cit-HepTh, high energy physics theory citation graph [6]"
echo " - facebook_combined, social network circles [7]"
echo " "
echo "[1] Timothy A. Davis and Yifan Hu. 2011. The University of Florida Sparse Matrix"
echo "    Collection."
echo "    ACM Transactions on Mathematical Software 38, 1, Article 1."
echo "[2] Leskovec, J. and Sosic, R. 2016. SNAP: A General-Purpose Network Analysis"
echo "    and Graph-Mining Library. ACM Transactions on Intelligent Systems and Technology"
echo "[3] Duff, I. S. and R. G. Grimes and J. G. Lewis, 1989. Sparse Matrix Problems."
echo "    ACM Trans. on Mathematical Software, vol 14, no. 1, pp 1-14, 1989."
echo "[4] Oberwolfach model reduction benchmark collection, 2004."
echo "[5] Jan Lienemann, Dag Billger, Evgenii B. Rudnyi, Andreas Greiner, and"
echo "    Jan G. Korvink, 2004. MEMS Compact Modeling Meets Model Order Reduction:"
echo "    Examples of the Application of Arnoldi Methods to Microsystem Devices."
echo "    Proc. '04 Nanotechnology Conference and Trade Show, Boston, Massachusetts, USA"
echo "[6] J. Leskovec, J. Kleinberg and C. Faloutsos. 2005. Graphs over Time: Densification"
echo "    Laws, Shrinking Diameters and Possible Explanations."
echo "    ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD)"
echo "[7] J. McAuley and J. Leskovec. 2012. Learning to Discover Social Circles in Ego"
echo "     Networks. NIPS."
echo "[8] V. Batagelj and A. Mrvar, Pajek datasets (Creative Commons BY-NC-SA 2.5),"
echo "    http://vlado.fmf.uni-lj.si/pub/networks/data/, 2006."
echo " "
echo "Please take note of the attributions to SuiteSparse, west0497, gyro_m, and EPA."
echo " "
echo "Please ensure the download you initiate is in line with applicable terms of use, "
echo "laws, and regulations."
echo " "
echo "Please use this script once and keep the datasets for future use."
echo ""
echo "On confirmation, the datasets will be downloaded to: ${DATASETS_DIR}"
echo ""
read -p "I have taken note and agree [yes/no] " -r
echo ""
if [[ "$REPLY" = "yes" ]]; then
	if [[ ! -d "${DATASETS_DIR}" ]]; then
		mkdir "${DATASETS_DIR}" || exit 1
	fi
	cd "${DATASETS_DIR}" || exit 1
	downloadSS "west0497" "https://suitesparse-collection-website.herokuapp.com/MM/HB/west0497.tar.gz"
	downloadSS "gyro_m" "https://suitesparse-collection-website.herokuapp.com/MM/Oberwolfach/gyro_m.tar.gz"
	downloadSS "dwt_59" "https://suitesparse-collection-website.herokuapp.com/MM/HB/dwt_59.tar.gz"
	downloadSS "EPA" "https://suitesparse-collection-website.herokuapp.com/MM/Pajek/EPA.tar.gz"
	downloadSN "cit-HepTh"
	downloadSN "facebook_combined"
	echo ""
	echo "The datasets are available in: ${DATASETS_DIR}"
	echo ""
	exit 0
else
	echo "'yes' is required to continue."
	exit 2
fi

