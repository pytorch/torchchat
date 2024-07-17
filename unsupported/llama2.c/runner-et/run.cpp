/* 
* Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
*/

/* Inference for Llama-2 Transformer model in pure C */
/* this uses the same logic regardless of AOTI OR ET */
/* but requires different data types - ATen vs ETen  */

#define __ET__MODEL
#include "../runner/run.cpp"
