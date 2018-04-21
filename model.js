class LSTM {
  constructor(options) {
    this.batchSize = options.batchSize;
    this.seqLength = options.seqLength;
    this.hiddenSize = options.hiddenSize;
    this.numLayers = options.numLayers;
    this.vocabSize = options.vocabSize;
    this.outputKeepProb = options.outputKeepProb;
  }
  async init(options) {
    const logger = options.logger || console.log;
    const training = options.training == null ? true : options.training;

    logger("setting up model...");

    if(!training){
      this.batchSize = 1;
      this.seqLength = 1;
    }

    const inputData = await tf.input({
      shape:[this.seqLength]
    });

    const labels = await tf.input({
      shape:[this.seqLength]
    });

    logger("creating lstm cells.");
    let cells = [];
    for(let i = 0; i < this.numLayers; i++) {
      const cell = await tf.layers.lstmCell({
        units: this.hiddenSize
      });
      cells.push(cell);
    }

    logger("creating multi-lstm layer.");
    const multiLstmCellLayer = await tf.layers.rnn({
      cell: cells,
      returnSequences: true
    });

    logger("creating embedding layer.");
    const embeddingsLayer = await tf.layers.embedding({
      inputDim: this.vocabSize,
      outputDim: this.hiddenSize
    });

    logger("creating dense layer.");
    const denseLayer = await tf.layers.dense({
      units: this.vocabSize,
      activation: 'relu',
      useBias: true,
      inputDim: this.hiddenSize
    })

    // logger("creating reshape layer.");
    // const reshapeLayer = tf.layers.reshape({targetShape:[this.seqLength]})//{
    //   targetShape = [this.seqLength, this.vocabSize]
    // });

    logger("applying embeddings.");
    let inputs = await embeddingsLayer.apply(inputData);

    if(training && this.outputKeepProb) {
      logger("applying dropout.");
      inputs = await tf.layers.dropout({
        rate: this.outputKeepProb, 
        noiseShape: [this.seqLength, this.vocabSize]
      }).apply(inputs)
    }

    logger("applying lstm layer.");
    let outputs = await multiLstmCellLayer.apply(inputs);
    logger("applying dense layer.")
    outputs = await denseLayer.apply(outputs);
    // outputs = reshapeLayer.apply(outputs);

    logger("creating model.");
    const model = await tf.model({inputs:inputData, outputs:outputs});

    logger("compiling...");

    model.compile({
      loss: 'categoricalCrossentropy', 
      optimizer: 'adam'
    });

    logger("done.");

    return model;
  }
}