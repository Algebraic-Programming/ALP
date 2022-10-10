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

# Author: A. N. Yzelman
# Date: 6th of October, 2022

echo "Detecting suspicious spacing errors in the current directory, `pwd`"
printf "\t spaces, followed by end-of-line...\n"
find . -type f | xargs grep -I ' $'
printf "\t tabs, followed by end-of-line...\n"
find . -type f | xargs grep -IP '\t$'
printf "\t spaces followed by a tab...\n"
find . -type f | xargs grep -IP ' \t'

