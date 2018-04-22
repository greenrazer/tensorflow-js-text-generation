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
    const logger = options.logger || console.log;
    const training = options.training == null ? true : options.training;

    logger("setting up model...");

    if(!training){
      this.seqLength = 1;
    }

    const inputData = await tf.input({
      shape:[this.seqLength]
    });

    let cells = [];
    for(let i = 0; i < this.numLayers; i++) {
      const cell = await tf.layers.lstmCell({
        units: this.hiddenSize
      });
      cells.push(cell);
    }

    const multiLstmCellLayer = await tf.layers.rnn({
      cell: cells,
      returnSequences: true
    });

    const embeddingsLayer = await tf.layers.embedding({
      inputDim: this.vocabSize,
      outputDim: this.hiddenSize
    });

    const dropoutLayer = await tf.layers.dropout({
      rate: this.outputKeepProb, 
      noiseShape: [this.hiddenSize]
    });

    const denseLayer = await tf.layers.dense({
      units: 1,
      activation: 'relu',
      useBias: true
    });

    // logger("creating reshape layer.");
    // const reshapeLayer = tf.layers.reshape({targetShape:[this.seqLength]})//{
    //   targetShape = [this.seqLength, this.vocabSize]
    // });

    let inputs = await embeddingsLayer.apply(inputData);

    if(training && this.outputKeepProb) {
      inputs = await dropoutLayer.apply(inputs);
    }

    let outputs = await multiLstmCellLayer.apply(inputs);

    outputs = await denseLayer.apply(outputs);
    // outputs = reshapeLayer.apply(outputs);

    const model = await tf.model({inputs:inputData, outputs:outputs});

    logger("compiling...");

    model.compile({
      loss: 'categoricalCrossentropy', 
      optimizer: 'adam'
    });

    logger("done.");

    this.model = await model;
  }
  async train(inData, outData, options) {
    options = options || {};
    const logger = options.logger || console.log;
    const batchSize = options.batchSize || 1;
    const epochs = options.epochs || 1;
    for (let i = 1; i < epochs+1; ++i) {
      const modelFit = await this.model.fit(inData, outData, {
        batchSize: batchSize,
        epochs: epochs
      });
      logger("Loss after epoch " + i + ": " + modelFit.history.loss[0]);
    }
  }
}

function prepareData(text) {
  let data = text.split("");
  let vocab = getVocab(data);
  data = data.map((value) => {
    let place = vocab.get(value);
    // return oneHot(vocab.size, place);
    return place;
  });
  let labels = data.slice();
  labels = labels.rotateLeft();
  return [data, labels, vocab];
}

function divideIntoSequences(data, seqLength, batchSize){
  let seqs = Math.floor(data.length / seqLength);
  let output = [];
  for(let i = 0; i < seqs; i++){
    let seqPacket = [];
    for(let j = 0; j < seqLength; j++){
      seqPacket.push(data[seqLength*i + j]);
    }
    output.push(seqPacket);
  }
  return output;
  // let batches = Math.floor(seqs / batchSize);
  // let outoutput = []
  // for(let i = 0; i < batches; i++){
  //   let batchPacket = [];
  //   for(let j = 0; j < batchSize; j++){
  //     batchPacket.push(output[batchSize*i + j]);
  //   }
  //   outoutput.push(batchPacket);
  // }
  // return outoutput;
}

function oneHot(size, at){
  let vector = [];
  for(let i = 0; i < size; i++){
    if(at == i){
      vector.push(1);
    }
    else{
      vector.push(0);
    }
  }
  return vector;
}

function getVocab(arr) {
  //get letter mapped to amount of occurances
  let counts = new Map();
  for(let i of arr){
    if(counts.has(i)){
      const value = counts.get(i);
      counts.set(i, value+1);
    }
    else {
      counts.set(i, 1);
    }
  }
  // here we are taking those occurances and turning it in
  // into a map from letter to how frequetly it appears relative to other letters
  return new Map(Array.from(counts).sort((a, b) => {
    return b[1] - a[1];
  }).map((value, i) => {
    return [value[0], i];
  }));
}

Array.prototype.rotateLeft = function() {
  let firstItem = this.shift();
  this.push(firstItem);
  return this;
}