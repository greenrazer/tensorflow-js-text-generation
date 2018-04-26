class LSTM {
  constructor(options) {
    if (options.seqLength &&
        options.hiddenSize &&
        options.numLayers &&
        options.vocabSize){
      this.seqLength = options.seqLength;
      this.hiddenSize = options.hiddenSize;
      this.numLayers = options.numLayers;
      this.vocabSize = options.vocabSize;
      this.outputKeepProb = options.outputKeepProb;
    }
    else {
      throw new Error("Missing some needed parameters");
    }
  }
  async init(options) {
    const logger = options && options.logger ? options.logger : console.log;

    logger("setting up model...");

    let cells = [];
    for(let i = 0; i < this.numLayers; i++) {
      const cell = await tf.layers.lstmCell({
        units: this.hiddenSize
      });
      cells.push(cell);
    }

    const multiLstmCellLayer = await tf.layers.rnn({
      cell: cells,
      returnSequences: true,
      inputShape: [this.seqLength, this.vocabSize]
    });

    const dropoutLayer = await tf.layers.dropout({
      rate: 0.2
    });

    const flattenLayer = tf.layers.flatten();

    const denseLayer = await tf.layers.dense({
      units: this.vocabSize,
      activation: 'softmax',
      useBias: true
    });

    const model = tf.sequential();
    model.add(multiLstmCellLayer);
    model.add(dropoutLayer);
    model.add(flattenLayer);
    model.add(denseLayer);

    logger("compiling...");

    model.compile({
      loss: 'categoricalCrossentropy', 
      optimizer: 'adam'
    });

    logger("done.");

    this.model = await model;
  }
  async train(inData, outData, options) {
    const logger = options && options.logger ? options.logger : console.log;
    const batchSize = options.batchSize;
    const epochs = options && options.epochs ? options.epochs : 1;
    for(let i = 0; i < epochs; i++){
      const modelFit = await this.model.fit(inData, outData, {
        batchSize: batchSize,
        epochs: 1,
      });
      logger("Loss after epoch " + (i+1) + ": " + modelFit.history.loss[0]);
    }
  }
  async predict(primer, amnt){
    let output = tf.tensor(primer);
    for(let i = 0; i < amnt; i++){
      let slicedVec = await output.slice(i,this.seqLength);
      slicedVec = slicedVec.reshape([1, slicedVec.shape[0], slicedVec.shape[1]]);
      let next = await this.model.predict(slicedVec, {
        batchSize: 1,
      });
      output = output.concat(next);
    }
    return output;
  }
}