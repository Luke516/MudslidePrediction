/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs';

const INPUT_NODE_NAME = 'images';

export class NasNet {
  constructor() {}

  async load() {
    this.model = await tf.loadModel("https://storage.googleapis.com/mudslide/my_model_2/model.json", false);
    this.model.summary();
    /*this.model = await tf.loadFrozenModel(
        GOOGLE_CLOUD_STORAGE_DIR + MODEL_FILE_URL,
        GOOGLE_CLOUD_STORAGE_DIR + WEIGHT_MANIFEST_FILE_URL);*/
  }

  dispose() {
    if (this.model) {
      this.model.dispose();
    }
  }
  /**
   * Infer through MobileNet. This does standard ImageNet pre-processing before
   * inferring through the model. This method returns named activations as well
   * as softmax logits.
   *
   * @param input un-preprocessed input Array.
   * @return The softmax logits.
   */
  predict(input) {
    /*const preprocessedInput = tf.div(
        tf.sub(input.asType('float32'), PREPROCESS_DIVISOR),
        PREPROCESS_DIVISOR);*/
    const preprocessedInput = tf.image.resizeBilinear(input.asType('float32'), [224,224]);
    //console.log(preprocessedInput);
    const reshapedInput =
        preprocessedInput.reshape([1, ...preprocessedInput.shape]);
    //return this.model.execute({[INPUT_NODE_NAME]: reshapedInput}, OUTPUT_NODE_NAME);
    return this.model.predict(reshapedInput).dataSync();
  }

  getTopKClasses(logits, topK) {
    const predictions = tf.tidy(() => {
      return tf.softmax(logits);
    });

    const values = predictions.dataSync();
    predictions.dispose();

    let predictionList = [];
    for (let i = 0; i < values.length; i++) {
      predictionList.push({value: values[i], index: i});
    }
    predictionList = predictionList
                         .sort((a, b) => {
                           return b.value - a.value;
                         })
                         .slice(0, topK);

    return predictionList.map(x => {
      return {label: IMAGENET_CLASSES[x.index], value: x.value};
    });
  }
}
